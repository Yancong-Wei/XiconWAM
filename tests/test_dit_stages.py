from types import SimpleNamespace

import pytest
import torch
from test_step import _sample, _tiny

from scripts.split_dit_checkpoint import split_joint_weights
from xiconwam.actor.mot import MoTPolicy
from xiconwam.actor.idm import IDMPolicy
from xiconwam.actor.hide import HidePolicy
from xiconwam.engine.session import TrainSession
from xiconwam.actor.video import VideoModel


def assert_same_weights(first, second):
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, param in first.state_dict().items():
        torch.testing.assert_close(param, second.state_dict()[name], rtol=0, atol=0)


def checkpoint_policy(architecture):
    if architecture != "hide":
        return _tiny(MoTPolicy if architecture == "mot" else IDMPolicy)
    actor = _tiny(MoTPolicy)
    return HidePolicy(
        video_dit=actor.video_expert,
        action_dit=actor.action_expert,
        vae=actor.vae,
        text_dim=32,
        state_dim=14,
        state_position="sequence",
        projector_hidden_dim=16,
        video_hidden_layer=0,
        device="cpu",
        torch_dtype=torch.float32,
    )


def lora_policy(architecture):
    from xiconwam.actor.lora import inject_video_dit_lora

    policy = checkpoint_policy(architecture)
    inject_video_dit_lora(
        policy.video_dit, {"r": 2, "lora_alpha": 2, "target_modules": ["self_attn.q"]}
    )
    TrainSession._apply_dit_only_train_mode(policy)
    return policy


@pytest.mark.parametrize("architecture", ["mot", "idm", "hide"])
def test_saved_hydra_target_keeps_legacy_peft_lora(monkeypatch, architecture):
    from hydra.utils import instantiate
    from xiconwam.actor.lora import has_lora

    policy = checkpoint_policy(architecture)
    monkeypatch.setattr(
        type(policy), "from_backbone_pretrained", lambda **kwargs: policy
    )
    restored = instantiate(
        {
            "_target_": f"xiconwam.launch.build_{architecture}",
            "backbone": {"name": "wan22"},
            "state_dim": 14,
            "action_dit_config": {"action_dim": 14},
            "lora": {"r": 2, "lora_alpha": 2, "target_modules": ["self_attn.q"]},
        }
    )
    assert has_lora(restored.video_dit)


@pytest.mark.parametrize("architecture", ["mot", "idm", "hide"])
def test_lora_checkpoint_roundtrip_and_integrity(tmp_path, architecture):
    original = lora_policy(architecture)
    path = tmp_path / "lora.pt"
    original.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    restored = lora_policy(architecture)
    restored.load_checkpoint(path)
    restored_state = restored.state_dict()
    for name, weight in payload["trainable_state"].items():
        torch.testing.assert_close(weight, restored_state[name], rtol=0, atol=0)

    del payload["trainable_state"][next(iter(payload["trainable_state"]))]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trainable tensors"):
        lora_policy(architecture).load_checkpoint(path)


@pytest.mark.parametrize("architecture", ["mot", "idm", "hide"])
def test_full_policy_checkpoint_roundtrip(tmp_path, architecture):
    original = checkpoint_policy(architecture)
    path = tmp_path / "complete.pt"
    original.save_checkpoint(path)
    restored = checkpoint_policy(architecture)
    restored.load_checkpoint(path)
    assert_same_weights(original.dit, restored.dit)
    assert_same_weights(original.state_encoder, restored.state_encoder)


@pytest.mark.parametrize("architecture", ["mot", "idm", "hide"])
@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "state"])
def test_incomplete_policy_checkpoint_is_rejected(tmp_path, architecture, damage):
    original = checkpoint_policy(architecture)
    path = tmp_path / "damaged.pt"
    original.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    weights = payload["dit" if architecture == "hide" else "streams"]
    key = next(iter(weights))
    if damage == "missing":
        del weights[key]
    elif damage == "extra":
        weights["unknown_parameter"] = torch.zeros(1)
    elif damage == "shape":
        weights[key] = torch.zeros(1)
    elif architecture == "hide":
        for name in list(weights):
            if name.startswith("state_encoder."):
                del weights[name]
    else:
        del payload["state_encoder"]
    torch.save(payload, path)
    with pytest.raises((RuntimeError, ValueError)):
        checkpoint_policy(architecture).load_checkpoint(path)


@pytest.mark.parametrize("architecture", ["mot", "hide"])
@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
def test_video_weights_require_explicit_component_loading(
    tmp_path, architecture, suffix
):
    original = checkpoint_policy(architecture)
    path = tmp_path / ("video" + suffix)
    original.save_video_checkpoint(path)
    restored = checkpoint_policy(architecture)
    with pytest.raises(ValueError, match="load_video_checkpoint"):
        restored.load_checkpoint(path)
    restored.load_video_checkpoint(path)
    assert_same_weights(original.video_dit, restored.video_dit)


def test_joint_split_and_independent_load(tmp_path):
    original = _tiny(MoTPolicy)
    joint = tmp_path / "joint.pt"
    video = tmp_path / "video.pt"
    action = tmp_path / "action.pt"
    original.save_checkpoint(joint, step=9)
    split_joint_weights(joint, video_path=video, action_path=action)
    restored = _tiny(MoTPolicy)
    initial_action = {
        key: value.clone() for key, value in restored.action_expert.state_dict().items()
    }
    restored.load_video_checkpoint(video)
    torch.testing.assert_close(
        restored.action_expert.state_dict(), initial_action, rtol=0, atol=0
    )
    assert_same_weights(original.video_expert, restored.video_expert)
    restored.load_action_checkpoint(action)
    assert_same_weights(original.action_expert, restored.action_expert)
    assert_same_weights(original.state_encoder, restored.state_encoder)
    original.save_action_checkpoint(action)
    restored.load_checkpoint(action)
    assert_same_weights(original.action_expert, restored.action_expert)


@pytest.mark.parametrize("freeze_video", [True, False])
def test_train_freezes_the_requested_dit(freeze_video):
    actor = _tiny(MoTPolicy)
    TrainSession._apply_dit_only_train_mode(actor, freeze_video, not freeze_video)
    if freeze_video:
        actor.loss_lambda_video = 0
    loss, metrics = actor(_sample())
    loss.backward()
    assert any(p.grad is not None for p in actor.video_expert.parameters()) == (
        not freeze_video
    )
    assert (
        any(p.grad is not None for p in actor.action_expert.parameters())
        == freeze_video
    )
    assert (
        any(p.grad is not None for p in actor.state_encoder.parameters())
        == freeze_video
    )
    assert actor.video_expert.training == (not freeze_video)
    assert actor.action_expert.training == freeze_video
    if freeze_video:
        assert metrics["weighted_loss_video"].item() == 0


def test_standalone_video_backward_and_roundtrip(tmp_path):
    actor = _tiny(MoTPolicy)
    actor.vae.upsampling_factor = 8
    components = SimpleNamespace(
        dit=actor.video_expert,
        vae=actor.vae,
        text_encoder=None,
        tokenizer=None,
    )
    model = VideoModel(
        components, {"train_shift": 3.0}, device="cpu", torch_dtype=torch.float32
    )
    TrainSession._apply_dit_only_train_mode(model)
    sample = _sample()
    sample = {key: sample[key] for key in ("video", "context", "context_mask")}
    loss, metrics = model(sample)
    loss.backward()
    assert torch.isfinite(metrics["loss_video"])
    assert any(p.grad is not None for p in model.dit.parameters())
    assert model.train_scheduler.shift == 3.0
    checkpoint = tmp_path / "video.safetensors"
    model.save_checkpoint(checkpoint)
    restored = _tiny(MoTPolicy)
    restored.load_video_checkpoint(checkpoint)
    assert_same_weights(model.dit, restored.video_expert)
