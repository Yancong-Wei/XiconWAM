import json
import shutil
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytest.importorskip("xicon_engine")
from xicon_engine import PolicyInfo

from xiconwam.data.lerobot.prompts import DEFAULT_PROMPT
from xiconwam.data.lerobot.text_embedding_cache import (
    build_text_embedding_payload,
    prompt_hash,
    text_embedding_cache_filename,
)
from xiconwam.deployment.bundle import export_bundle, load_bundle_config, verify_bundle
from xiconwam.kit.config_resolvers import register_default_resolvers

register_default_resolvers()


@pytest.fixture
def inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="task/robodojo", overrides=["mixed_precision='no'"])
    OmegaConf.resolve(cfg)
    assets = tmp_path / "input"
    assets.mkdir()
    (assets / "backbone").mkdir()
    (assets / "backbone/weights.bin").write_bytes(b"fixture-backbone")
    (assets / "expert.pt").write_bytes(b"fixture-action-initializer")
    cfg.model.backbone.model_id = "backbone"
    cfg.model.backbone.text_dim = 8
    cfg.model.backbone.action_dit_pretrained_path = "expert.pt"
    cfg.model.action_dit_pretrained_path = "expert.pt"
    cfg.data.train.pretrained_norm_stats = "stats.json"
    cfg.data.train.text_embedding_cache_dir = "text"
    cfg.data.train.dataset_dirs = []
    (assets / "stats.json").write_text(
        json.dumps(
            {
                group: {
                    "default": {"global_mean": [0.0] * 14, "global_std": [1.0] * 14}
                }
                for group in ("action", "state")
            }
        )
    )
    (assets / "text").mkdir()
    prompt = DEFAULT_PROMPT.format(task="pick")
    filename = text_embedding_cache_filename(
        prompt, 128, "wan22ti2v5b", dtype=torch.float32
    )
    torch.save(
        build_text_embedding_payload(
            context=torch.zeros(128, 8),
            mask=torch.ones(128, dtype=torch.bool),
            context_len=128,
            encoder_id="wan22ti2v5b",
            prompt_digest=prompt_hash(prompt),
            dtype=torch.float32,
        ),
        assets / "text" / filename,
    )
    config = assets / "config.yaml"
    OmegaConf.save(cfg, config)
    checkpoint = assets / "weights.pt"
    checkpoint.write_bytes(b"fixture-trained-checkpoint")
    contract = assets / "contract.json"
    info = PolicyInfo(
        "test-policy",
        32,
        14,
        14,
        20,
        "fixture/joint14",
        camera_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        action_names=tuple(f"q{i}" for i in range(14)),
        action_units=("rad",) * 14,
    )
    contract.write_text(json.dumps(info.to_payload()))
    return dict(
        config=config, checkpoint=checkpoint, contract=contract, base_dir=assets
    )


def test_bundle_moves_without_original_paths_and_loads_through_service(
    inputs, tmp_path, monkeypatch
):
    output = tmp_path / "bundle"
    manifest = export_bundle(**inputs, output=output, instructions=["pick"])
    assert manifest["instructions"] == ["pick"]
    assert manifest["code"]["python_sources_sha256"]
    assert len(manifest["text_prompts"]) == 1
    assert manifest["policy"]["schema_id"] == "fixture/joint14"
    moved = tmp_path / "moved"
    shutil.move(output, moved)
    shutil.rmtree(inputs["base_dir"])
    cfg, checkpoint, contract = load_bundle_config(moved)
    assert Path(cfg.model.backbone.model_id).is_dir()
    assert Path(cfg.data.train.pretrained_norm_stats).is_file()
    assert cfg.data.train.dataset_dirs == []
    assert checkpoint.read_bytes() == b"fixture-trained-checkpoint"
    assert contract.is_file()
    assert verify_bundle(moved) == manifest
    # Real loading/preprocessing/text-cache paths, with only the multi-GB model replaced.
    import hydra.utils
    from types import SimpleNamespace
    from xiconwam.deployment.serve import main
    import xiconwam.deployment.serve as service
    from test_engine_policy import observation

    instantiate = hydra.utils.instantiate

    class FakeModel:
        vae = SimpleNamespace()

        def load_checkpoint(self, path):
            assert Path(path).is_file()

        def eval(self):
            return self

        def infer_action(self, **kw):
            assert kw["context"].shape == (128, 8)
            return {"action": torch.zeros(32, 48)}

    monkeypatch.setattr(
        hydra.utils,
        "instantiate",
        lambda spec, **kw: (
            FakeModel()
            if spec.get("_target_") == "xiconwam.launch.build_model"
            else instantiate(spec, **kw)
        ),
    )
    monkeypatch.setattr(
        service,
        "serve",
        lambda policy, **kw: policy.infer(
            type(observation())(torch.zeros(14).numpy(), observation().images, "pick")
        ),
    )
    monkeypatch.chdir(tmp_path)
    main(["--bundle", str(moved), "--device", "cpu"])


@pytest.mark.parametrize(
    "resource", ["checkpoint.pt", "contract.json", "config.yaml", "stats"]
)
def test_corruption_fails_verification(inputs, tmp_path, resource):
    output = tmp_path / "bundle"
    manifest = export_bundle(**inputs, output=output)
    if resource == "stats":
        resource = manifest["resource_bindings"]["data.train.pretrained_norm_stats"]
    path = output / resource
    content = path.read_bytes()
    path.write_bytes(
        bytes([content[0] ^ 1]) + content[1:]
    )  # Same size: hash must catch it.
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(output)


def test_failed_export_is_atomic_and_does_not_overwrite(inputs, tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="required instruction"):
        export_bundle(**inputs, output=output, instructions=["unknown"])
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_bundle(**inputs, output=output)


def test_inventory_and_traversal(inputs, tmp_path):
    output = tmp_path / "bundle"
    export_bundle(**inputs, output=output)
    (output / "extra").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(output)
    (output / "extra").unlink()
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["resource_bindings"]["model.backbone.model_id"] = "../input"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid bundle member"):
        verify_bundle(output)


def test_code_provenance_survives_missing_git(monkeypatch):
    from xiconwam.deployment.bundle import _provenance

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("xiconwam.deployment.bundle.subprocess.run", unavailable)
    record = _provenance()
    assert record["git_commit"] is None
    assert "deployment/bundle.py" in record["python_sources_sha256"]


def test_mixture_export_requires_explicit_source_and_preserves_its_stats(
    inputs, tmp_path
):
    cfg = OmegaConf.load(inputs["config"])
    source = OmegaConf.to_container(cfg.data.train, resolve=True)
    cfg.data = {
        "train": {
            "_target_": "xiconwam.data.mixture.build_mixture_dataset",
            "sources": {"robot": {"enabled": True, "config": source}},
        }
    }
    OmegaConf.save(cfg, inputs["config"])
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="--data-source"):
        export_bundle(**inputs, output=output)
    manifest = export_bundle(**inputs, output=output, data_source="robot")
    assert manifest["data_source"] == "robot"
    bundled, _, _ = load_bundle_config(output)
    assert bundled.data.train.processor.action_state_merger.unify_action_map == ["0-13"]
    assert Path(bundled.data.train.pretrained_norm_stats).is_file()
