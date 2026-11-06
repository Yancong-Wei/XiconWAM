"""Physical-unit adapter, real RPC and Engine's own open-loop evaluator."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

engine = pytest.importorskip("xicon_engine")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from xicon_engine import Observation, PolicyClient, PolicyInfo, PolicyServer

from xiconwam.data.lerobot.processors.window_packer import WindowPacker
from xiconwam.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from xiconwam.data.lerobot.transforms.relative_action import RelativeJointTransform
from xiconwam.deployment.policy import XiconWAMPolicy
from xiconwam.data.lerobot.prompts import DEFAULT_PROMPT

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def make_policy(relative=False):
    meta = {
        "images": [
            dict(key=k, shape=[3, h, w])
            for k, (h, w) in zip(CAMERAS, [(256, 320), (128, 160), (128, 160)])
        ],
        "state": [{"key": "default", "raw_shape": 2, "shape": 2}],
        "action": [{"key": "default", "raw_shape": 2, "shape": 2}],
    }
    processor = WindowPacker(
        meta,
        num_obs_steps=33,
        action_output_dim=48,
        proprio_output_dim=48,
        action_state_merger=ConcatLeftAlign(unify_action_map=[5, 17]),
        action_state_transforms=(
            [RelativeJointTransform(["default"])] if relative else None
        ),
    )
    stats = {
        group: {
            "default": {
                "global_mean": torch.tensor([10.0, 20.0]),
                "global_std": torch.tensor([2.0, 4.0]),
            }
        }
        for group in ("action", "state")
    }
    processor.set_normalizer_from_stats(stats)
    info = PolicyInfo(
        "fixture-model",
        4,
        2,
        2,
        20.0,
        "fixture/v1",
        camera_keys=CAMERAS,
        action_names=("q0", "q1"),
        action_units=("rad", "rad"),
    )
    calls = []

    def infer_action(**kw):
        calls.append(kw)
        assert kw["input_image"].shape == (3, 384, 320)
        assert kw["input_image"].min() == -1
        assert kw["input_image"].max() == 1
        assert kw["proprio"].shape == (48,)
        action = torch.zeros((4, 48))
        action[:, 5], action[:, 17] = 1.0, -1.0
        return {"action": action}

    policy = XiconWAMPolicy(
        SimpleNamespace(infer_action=infer_action), processor, info, num_video_frames=9
    )
    return policy, calls


def observation(state=(100.0, 200.0)):
    return Observation(
        state,
        {
            k: np.full((8, 8, 3), i * 127 + (i == 2), np.uint8)
            for i, k in enumerate(CAMERAS)
        },
        "pick",
    )


@pytest.mark.parametrize("relative", [False, True])
def test_full_chunk_inverse_mapping_normalization_and_physical_reference(relative):
    policy, calls = make_policy(relative)
    result = policy.infer(observation())
    expected = [112.0, 216.0] if relative else [12.0, 16.0]
    np.testing.assert_allclose(result.values, np.tile(expected, (4, 1)))
    assert calls[0]["prompt"] == DEFAULT_PROMPT.format(task="pick")
    # Training normalizer clips an outlier state; inverse delta uses exact raw state.
    assert calls[0]["proprio"][5] == 5
    assert calls[0]["proprio"][17] == 5
    assert result.values.shape == (4, 2)  # no accidental 32-row history crop


def test_real_websocket_adapter_and_engine_open_loop(tmp_path):
    from xicon_engine.validation import LeRobotEpisode, evaluate

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import io
    import json

    from PIL import Image

    policy, _calls = make_policy()
    root = tmp_path / "dataset"
    (root / "meta/episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 20})
    )
    pq.write_table(
        pa.Table.from_pylist([{"episode_index": 0}]),
        root / "meta/episodes/file.parquet",
    )
    images = {}
    for key, value in observation().images.items():
        buffer = io.BytesIO()
        Image.fromarray(value).save(buffer, format="PNG")
        images[key] = {"bytes": buffer.getvalue()}
    rows = [
        dict(
            episode_index=0,
            frame_index=i,
            timestamp=i / 20,
            **{"observation.state": [100.0, 200.0], "action": [13.0, 18.0], **images},
        )
        for i in range(7)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/file.parquet")
    report = tmp_path / "report"

    def client_work(url):
        dataset = LeRobotEpisode(root)
        try:
            with PolicyClient(url) as client:
                return evaluate(
                    dataset,
                    client,
                    output=report,
                    image_map={k: k for k in CAMERAS},
                    instruction="pick",
                )
        finally:
            dataset.close()

    async def run():
        server = await PolicyServer(policy, port=0).start()
        try:
            return await asyncio.to_thread(client_work, f"ws://127.0.0.1:{server.port}")
        finally:
            await server.close()

    summary = asyncio.run(run())
    assert summary["chunks"] == 2
    assert summary["valid_samples"] == 7
    assert summary["mae_per_axis"] == [1.0, 2.0]
    arrays = np.load(report / "predictions.npz")
    assert arrays["predictions"].shape == (2, 4, 2)
    assert arrays["valid_mask"].sum() == 7
    for name in (
        "actions_pred_vs_gt.png",
        "actions_error.png",
        "cameras.png",
        "open_loop_timeline.png",
    ):
        assert (report / name).stat().st_size > 1000


def test_cached_text_is_passed_without_online_encoder():
    policy, calls = make_policy()
    context = torch.ones(4, 8)
    mask = torch.ones(4, dtype=torch.bool)
    prompts = []

    def get_context(prompt):
        prompts.append(prompt)
        return {"context": context, "mask": mask}

    policy.text_context = get_context
    policy.infer(observation())
    assert prompts == [DEFAULT_PROMPT.format(task="pick")]
    assert calls[0]["prompt"] is None
    assert calls[0]["context"] is context
    assert calls[0]["context_mask"] is mask


def test_wrong_camera_contract_fails_before_model_call():
    policy, calls = make_policy()
    with pytest.raises(engine.ProtocolError, match="camera keys"):
        policy.infer(Observation([1.0, 2.0]))
    assert not calls


def test_training_instruction_override_is_preserved_in_serving():
    policy, calls = make_policy()
    policy.override_instruction = "抓取@pick"
    policy.infer(observation())
    assert calls[0]["prompt"] == DEFAULT_PROMPT.format(task="抓取@pick")
