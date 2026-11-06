import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from xiconwam.data.lerobot.processors.window_packer import WindowPacker
from xiconwam.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from xiconwam.data.mixture import MixtureDataset
from xiconwam.engine.evaluate import EvaluateMixin
from xiconwam.engine.metrics import action_error_metrics
from xiconwam.kit.samplers import ResumableEpochSampler


def processor(scale=1):
    meta = {
        "images": [],
        "action": [{"key": "q", "shape": 2, "raw_shape": 2}],
        "state": [{"key": "q", "shape": 2, "raw_shape": 2}],
    }
    result = WindowPacker(
        meta,
        num_obs_steps=5,
        action_state_merger=ConcatLeftAlign(unify_action_map=[5, 17]),
    )
    result.set_normalizer_from_stats(
        {
            g: {
                "q": {
                    "global_mean": torch.zeros(2),
                    "global_std": torch.tensor([2.0, 4.0]) * scale,
                }
            }
            for g in ("action", "state")
        }
    )
    return result.eval()


def sample():
    return {
        "video": -torch.ones(3, 5, 16, 16),
        "action": torch.zeros(4, 48),
        "proprio": torch.zeros(4, 48),
        "prompt": "test",
        "context": None,
        "context_mask": None,
        "action_is_pad": torch.tensor([False, False, True, True]),
        "image_is_pad": torch.tensor([False, False, False, True, True]),
        "action_dim_is_pad": ~torch.isin(torch.arange(48), torch.tensor([5, 17])),
    }


def test_physical_errors_ignore_unify_slots_and_padded_time():
    data = sample()
    pred = torch.full((4, 48), float("nan"))
    pred[:2, [5, 17]] = 1
    metrics, weights = action_error_metrics(
        processor(),
        pred,
        data["action"],
        data["proprio"][None],
        data["action_is_pad"],
        data["action_dim_is_pad"],
    )
    assert metrics["action_l1"] == pytest.approx(3)
    assert metrics["action_l2"] == pytest.approx(10)
    assert weights["action_l1"] == 4
    assert metrics["action/q/0/l1"] == pytest.approx(2)
    assert metrics["action/q/1/l1"] == pytest.approx(4)
    data["action_is_pad"][:] = True
    assert action_error_metrics(
        processor(),
        pred,
        data["action"],
        data["proprio"][None],
        data["action_is_pad"],
        data["action_dim_is_pad"],
    ) == ({}, {})


class Windows(torch.utils.data.Dataset):
    def __init__(self, scale=1, is_training_set=True, norm_stats_output=None):
        self.lerobot_dataset = SimpleNamespace(processor=processor(scale))
        self.is_training_set = is_training_set

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return sample()


class Model(torch.nn.Module):
    device = "cpu"
    torch_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(1, 1)

    def training_loss(self, data):
        assert data["action_is_pad"].shape == (1, 4)
        assert data["action_dim_is_pad"].shape == (1, 48)
        assert data["image_is_pad"].shape == (1, 5)
        torch.rand(1)  # Evaluation must restore training RNG.
        return torch.tensor(1.0), {}

    def infer(self, **kwargs):
        pred = torch.ones(4, 48)
        pred[2:] = 1000
        frames = [
            Image.fromarray(np.full((16, 16, 3), 0 if i < 3 else 255, np.uint8))
            for i in range(5)
        ]
        return {"video": frames, "action": pred}

    def _encode_video_latents(self, video):
        return video

    def _decode_latents(self, video):
        return [[Image.fromarray(np.zeros((16, 16, 3), np.uint8)) for _ in range(5)]]


class Harness(EvaluateMixin):
    def _set_dit_only_train_mode(self):
        self.model.dit.train()


def harness(tmp_path, dataset):
    run = Harness()
    run.model, run.val_dataset = Model(), dataset
    run.cfg = OmegaConf.create({"eval_num_samples": 3, "eval_seed": 9})
    run.eval_num_inference_steps, run.eval_save_video = 1, False
    run.global_step, run.eval_dir = 10, str(tmp_path)
    run.accelerator = SimpleNamespace(
        unwrap_model=lambda m: m,
        num_processes=1,
        process_index=0,
        is_main_process=True,
        autocast=nullcontext,
    )
    return run


def test_mixture_full_evaluate_uses_source_stats_fixed_windows_and_masks(tmp_path):
    run = harness(tmp_path, MixtureDataset([Windows(), Windows(2)], names=["a", "b"]))
    rng = torch.get_rng_state().clone()
    first = run.evaluate()
    assert torch.equal(rng, torch.get_rng_state())
    assert run.model.dit.training
    assert first["num_windows"] == 6
    assert first["sources/a/action_l1"] == pytest.approx(3)
    assert first["sources/b/action_l1"] == pytest.approx(6)
    assert first["psnr_rg"] == pytest.approx(80)  # Corrupt padded frames ignored.
    before = json.loads((tmp_path / "step_000010.json").read_text())
    run.global_step = 20
    assert run.evaluate() == first
    after = json.loads((tmp_path / "step_000020.json").read_text())
    assert before["selected_indices"] == after["selected_indices"]
    assert before["metric_counts"]["action_l1"] == 24


def test_eval_restores_training_mode_on_error(tmp_path):
    run = harness(tmp_path, Windows())

    def broken(**kwargs):
        raise ValueError("bad inference")

    run.model.infer = broken
    with pytest.raises(ValueError, match="bad inference"):
        run.evaluate()
    assert run.model.dit.training


def test_eval_with_no_valid_target_frames_can_be_logged(tmp_path):
    from xiconwam.engine.session import TrainSession

    run = harness(tmp_path, Windows())
    # A backbone evaluating only the final frame sees a padded target here.
    run.model.video_dit = SimpleNamespace(evaluation_metric_frames=[-1])
    metrics = run.evaluate()
    assert "psnr_rd" not in metrics
    payloads = []
    logger = SimpleNamespace(global_step=run.global_step, _wandb_log=payloads.append)
    TrainSession._log_evaluation(logger, metrics)
    assert payloads[0]["eval/val_loss"] == 1
    assert not any("psnr" in key for key in payloads[0])


@pytest.mark.parametrize("explicit_validation", [False, True])
def test_mixture_computed_stats_are_isolated_and_reused_for_validation(
    tmp_path, monkeypatch, explicit_validation
):
    from pathlib import Path
    from xiconwam.launch import build_datasets
    from xiconwam.data.lerobot.utils.normalizer import load_dataset_stats_from_json

    computations = []

    class RawDataset:
        def __init__(self, dataset_dirs, is_training_set, **kwargs):
            self.name = dataset_dirs[0]
            self.training = is_training_set

        def __len__(self):
            return 10

        def _set_return_images(self, value):
            pass

        def set_processor(self, value):
            self.processor = value

        def get_dataset_stats(self, processor):
            assert self.training  # Never compute stats on the validation split.
            computations.append(self.name)
            return {
                group: {
                    "q": {
                        "global_mean": torch.zeros(2),
                        "global_std": torch.full(
                            (2,), 2.0 if self.name == "a" else 7.0
                        ),
                    }
                }
                for group in ("action", "state")
            }

    monkeypatch.setattr(
        "xiconwam.data.lerobot.robot_video_dataset.BaseLerobotDataset", RawDataset
    )
    monkeypatch.setattr("xiconwam.kit.misc.get_work_dir", lambda: str(tmp_path))
    meta = processor().shape_meta

    def spec(name):
        return {
            "config": {
                "_target_": "xiconwam.data.lerobot.robot_video_dataset.EpisodeWindowDataset",
                "dataset_dirs": [name],
                "shape_meta": meta,
                "num_frames": 5,
                "video_size": [16, 16],
                "context_len": 8,
                "text_encoder_id": "test",
                "preflight_text_cache": False,
                "pretrained_norm_stats": None,
                "processor": {
                    "_target_": "xiconwam.data.lerobot.processors.window_packer.WindowPacker",
                    "shape_meta": meta,
                    "num_obs_steps": 5,
                    "action_state_merger": {
                        "_target_": "xiconwam.data.lerobot.transforms.action_state_merger.ConcatLeftAlign",
                        "unify_action_map": [5, 17],
                    },
                },
            }
        }

    cfg = OmegaConf.create(
        {
            "sources": {name: spec(name)["config"] for name in ("a", "b")},
            "train": {
                "_target_": "xiconwam.data.mixture.build_mixture_dataset",
                "_recursive_": False,
                "sources": {
                    name: {"config": "${sources." + name + "}"} for name in ("a", "b")
                },
            },
        }
    )
    if explicit_validation:
        cfg.val = OmegaConf.to_container(cfg.train, resolve=True)
        cfg.val.is_training_set = False
    OmegaConf.set_struct(cfg, True)
    train, val = build_datasets(cfg)
    assert computations == ["a", "b"]
    assert train.datasets[0].norm_stats_path != train.datasets[1].norm_stats_path
    for i, name in enumerate(("a", "b")):
        path = train.datasets[i].norm_stats_path
        assert val.datasets[i].norm_stats_path == path
        assert cfg.train.sources[name].config.pretrained_norm_stats == path
        assert Path(path).is_file()
        stats = load_dataset_stats_from_json(path)
        assert (
            stats["action"]["q"]["global_std"].tolist()
            == [2.0 if name == "a" else 7.0] * 2
        )
        metrics, _ = action_error_metrics(
            val.datasets[i].lerobot_dataset.processor,
            torch.ones(4, 48),
            torch.zeros(4, 48),
            torch.zeros(1, 4, 48),
        )
        assert metrics["action_l1"] == pytest.approx(2.0 if name == "a" else 7.0)


def test_source_coverage_resume_and_persistent_workers():
    class Indices(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, i):
            return {"index": i}

    ds = MixtureDataset([Indices(), Indices()], weights=[0.1, 0.9])
    covered = set()
    for epoch in range(5):
        ds.set_epoch(epoch)
        covered.update(ds._index_map[ds._index_map[:, 0] == 0, 1].tolist())
    assert covered == set(range(100))
    restored = MixtureDataset([Indices(), Indices()], weights=[0.1, 0.9])
    restored.set_epoch(4)
    np.testing.assert_array_equal(restored._index_map, ds._index_map)
    sampler = ResumableEpochSampler(ds, 42, 2, 1)
    sampler.set_epoch_offset(3)
    list(iter(sampler))
    assert ds._epoch == 3
    # A shallow worker copy shares the epoch tensor but has its own index map.
    import copy

    worker = copy.copy(ds)
    ds.set_epoch(7)
    worker[0]
    np.testing.assert_array_equal(worker._index_map, ds._index_map)


def test_mixture_hydra_build_instantiates_each_source_with_correct_split():
    from hydra.utils import instantiate

    cfg = OmegaConf.create(
        {
            "_target_": "xiconwam.data.mixture.build_mixture_dataset",
            "_recursive_": False,
            "is_training_set": False,
            "sources": {"one": {"config": {"_target_": "test_validation.Windows"}}},
        }
    )
    mixed = instantiate(cfg)
    assert len(mixed) == 10
    assert mixed.names == ["one"]
    assert not mixed.datasets[0].is_training_set


class IndexedWindows(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return {"index": index}


def test_persistent_dataloader_workers_observe_epoch_changes():
    ds = MixtureDataset([IndexedWindows(), IndexedWindows()], weights=[0.1, 0.9])
    sampler = ResumableEpochSampler(ds, 42, 2, 1)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, num_workers=1, persistent_workers=True, sampler=sampler
    )
    # Return source-local indices as data, independent of shuffle order.
    try:
        first = {
            int(index)
            for batch in loader
            for source, index in zip(batch["_dataset_index"], batch["index"])
            if source == 0
        }
        sampler.set_epoch(1)
        second = {
            int(index)
            for batch in loader
            for source, index in zip(batch["_dataset_index"], batch["index"])
            if source == 0
        }
        assert ds._epoch == 1
        assert first != second
        assert second == set(ds._index_map[ds._index_map[:, 0] == 0, 1])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
