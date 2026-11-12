"""Regression checks for backbone composition and one-path training resume."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_step import _sample, _tiny

from xiconwam.actor.mot import MoTPolicy
from xiconwam.actor.video import VideoModel
from xiconwam.engine.configuration import resolve_training_config
from xiconwam.engine.launch_config import launcher_config
from xiconwam.model_config import (
    action_dit,
    resolve_action_config,
    validate_architecture,
    video_backbone,
)

ROOT = Path(__file__).resolve().parents[1]


def task(name, overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name=f"task/{name}", overrides=list(overrides))


@pytest.mark.parametrize(
    "name,layers,heads",
    [
        ("robodojo_wan", 30, 24),
        ("robodojo_cosmos", 28, 16),
        ("robodojo_flux", 25, 24),
    ],
)
@pytest.mark.parametrize("architecture", ["mot", "idm", "hide", "video"])
def test_each_architecture_selects_independent_backbones(
    name, layers, heads, architecture
):
    cfg = task(name, [f"model={architecture}"])
    OmegaConf.resolve(cfg)
    assert "action_expert" not in cfg.model.video_dit
    assert "action_dit_pretrained_path" not in cfg.model.video_dit
    assert "action_backbone" not in cfg.model
    if architecture == "video":
        assert action_dit(cfg.model) is None
        assert cfg.data.train.processor.action_output_dim == 48
        return
    action = resolve_action_config(cfg.model.video_dit, cfg.model.action_dit)
    assert action["num_heads"] == heads
    if name == "robodojo_flux":
        assert action["num_layers_double"] + action["num_layers_single"] == layers
        if architecture == "hide":
            with pytest.raises(ValueError, match="cross-attention"):
                validate_architecture(architecture, action)
            return
    else:
        assert action["num_layers"] == layers
    validate_architecture(architecture, action)


@pytest.mark.parametrize("legacy", ["current", "backbone", "nested", "separate"])
def test_resume_restores_the_saved_pair_and_freeze_policy(tmp_path, legacy):
    run = tmp_path / "original"
    state = run / "checkpoints/state/step_000010"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text('{"global_step": 10}')
    saved = task(
        "robodojo_cosmos",
        ["model=idm", "model.training.mode=action_only", "zero=3", "zero_offload=true"],
    )
    if legacy != "current":
        saved = OmegaConf.to_container(saved, resolve=True)
        if legacy in {"backbone", "nested"}:
            saved["model"]["backbone"] = saved["model"].pop("video_dit")
            if legacy == "nested":
                saved["model"]["backbone"]["action_dit"] = saved["model"].pop(
                    "action_dit"
                )
        else:
            video = saved["model"].pop("video_dit")
            saved["model"]["action_backbone"] = saved["model"].pop("action_dit")
            saved["model"]["video_backbone"] = video
        saved["acceleration"] = {
            "framework": "deepspeed",
            "zero": saved.pop("zero"),
            "offload": saved.pop("zero_offload"),
        }
    OmegaConf.save(saved, run / "config.yaml")
    # An incomplete later checkpoint must not become the automatic resume target.
    (state.parent / "step_000011").mkdir()
    current = task("robodojo_wan", [f"resume={run}", "max_steps=20"])
    restored = resolve_training_config(current, ["max_steps=20"])
    assert video_backbone(restored.model).name == "cosmos25"
    assert restored.model.architecture == "idm"
    assert restored.model.training.mode == "action_only"
    assert video_backbone(restored.model).from_scratch
    assert action_dit(restored.model).checkpoint is None
    assert "acceleration" not in restored
    launch = launcher_config(restored, 8, ROOT)
    assert launch["distributed_type"] == "DEEPSPEED"
    assert (
        Path(launch["deepspeed_config"]["deepspeed_config_file"]).name
        == "ds_zero3_offload_config.json"
    )
    assert restored.resume == str(state)
    assert restored.max_steps == 20
    with pytest.raises(ValueError, match="saved model"):
        resolve_training_config(current, ["model=mot"])


@pytest.mark.parametrize(
    "name,steps,batch,workers",
    [
        ("robodojo_wan", 90000, 16, 4),
        ("robodojo_cosmos", 30000, 16, 8),
        ("robodojo_flux", 30000, 16, 8),
        ("robodojo_a800", 200000, 10, 0),
        ("wan_video", 1000, 1, 2),
    ],
)
def test_shared_training_defaults_preserve_task_schedules(name, steps, batch, workers):
    from xiconwam.engine.configuration import validate_training_config

    cfg = task(name)
    validate_training_config(cfg)
    assert (cfg.max_steps, cfg.batch_size, cfg.num_workers) == (steps, batch, workers)
    assert cfg.wandb.entity is None


@pytest.mark.parametrize(
    "override,reason",
    [
        ("+model.typo=1", "Unknown model"),
        ("+model.video_dit.typo=1", "video_dit configuration"),
        ("batch_size=0", "batch_size"),
        ("model.training.freeze_video=true", "trainable"),
        ("data.train.window_steps=32", "divisible"),
        ("model.video_dit.lora.rank=0", "rank"),
    ],
)
def test_invalid_config_is_rejected_before_loading_assets(override, reason):
    from xiconwam.engine.configuration import validate_training_config

    extra = []
    if "freeze_video" in override:
        extra = ["model.training.freeze_action=true"]
    if ".lora.rank" in override:
        extra = ["model.video_dit.lora.enable=true"]
    with pytest.raises(ValueError, match=reason):
        validate_training_config(task("robodojo_wan", [override, *extra]))


def test_mixture_seed_and_horizon_follow_their_configuration():
    from xiconwam.data.lerobot.robot_video_dataset import (
        configured_window_steps,
        trained_window,
    )

    cfg = task("robodojo_wan", ["data=mixture", "seed=13"])
    assert cfg.data.train.seed == 13
    data = cfg.data.train.sources.robodojo.config
    assert trained_window(
        configured_window_steps(data), data.action_video_freq_ratio
    ) == (9, 32)
    assert configured_window_steps({"num_frames": 33}) == 33


def test_hide_learning_rate_scale_tracks_the_optimizer_rate():
    actor = _tiny(MoTPolicy)
    from xiconwam.actor.hide import HidePolicy

    model = HidePolicy(
        video_dit=actor.video_dit,
        action_dit=actor.action_expert,
        vae=actor.vae,
        state_dim=14,
        text_dim=32,
        video_hidden_layer=0,
        learning_rate_scale={"video": 0.1, "action": 1.0},
    )
    first = model.optimizer_param_groups(1e-4, 0.01)
    second = model.optimizer_param_groups(2e-4, 0.01)
    assert [group["lr"] for group in first] == pytest.approx([1e-5, 1e-4])
    assert [group["lr"] for group in second] == pytest.approx([2e-5, 2e-4])


def test_action_shape_override_is_checked_before_loading_weights():
    cfg = task("robodojo_wan", ["model.action_dit.dit_config.num_heads=16"])
    with pytest.raises(ValueError, match="must match"):
        resolve_action_config(cfg.model.video_dit, cfg.model.action_dit)


def test_cosmos_standalone_video_backward():
    from xiconwam.vision.cosmos.cosmos_video_dit import (
        Cosmos25DiTConfig,
        Cosmos25VideoDiT,
    )

    config = Cosmos25DiTConfig(
        hidden_size=128,
        num_heads=1,
        num_layers=1,
        intermediate_size=256,
        context_dim=32,
        reason_context_dim=64,
        adaln_lora_dim=8,
        attention_backend="sdpa",
        video_attention_mask_mode="first_frame_causal",
    )
    actor = _tiny(MoTPolicy)
    actor.vae.upsampling_factor = 8
    components = SimpleNamespace(
        dit=Cosmos25VideoDiT(config),
        vae=actor.vae,
        text_encoder=None,
        tokenizer=None,
    )
    model = VideoModel(components, {}, device="cpu", torch_dtype=torch.float32)
    sample = {
        key: value
        for key, value in _sample().items()
        if key in {"video", "context", "context_mask"}
    }
    loss, _ = model(sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None for parameter in model.dit.blocks.parameters()
    )


def test_flux_video_endpoint_training_and_inference():
    from xiconwam.vision.flux.flux2_video_dit import Flux2, Flux2Params, Flux2VideoDiT

    params = Flux2Params(
        in_channels=16,
        context_in_dim=32,
        hidden_size=32,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        mlp_ratio=2,
        use_guidance_embed=False,
    )

    class Codec(torch.nn.Module):
        def encode(self, images):
            value = torch.nn.functional.adaptive_avg_pool2d(images, (4, 4))
            return value.repeat(1, 6, 1, 1)[:, :16]

        def decode(self, latents):
            return torch.nn.functional.interpolate(latents[:, :3], (32, 32))

    components = SimpleNamespace(
        dit=Flux2VideoDiT(Flux2(params), "tiny", "sdpa"),
        vae=Codec(),
        text_encoder=None,
        tokenizer=None,
    )
    model = VideoModel(components, {}, device="cpu", torch_dtype=torch.float32)
    sample = _sample()
    loss, _ = model(
        {
            key: value
            for key, value in sample.items()
            if key in {"video", "context", "context_mask"}
        }
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.dit.parameters())
    model.encode_prompt = lambda prompts: (sample["context"], sample["context_mask"])
    frames = model.infer_video(
        torch.zeros(3, 32, 32), "Move it.", frames=2, steps=2, seed=1
    )
    assert len(frames) == 2 and frames[1].size == (32, 32)


def test_hide_component_roundtrip_includes_its_state_and_video_projector(tmp_path):
    from scripts.split_dit_checkpoint import split_joint_weights
    from xiconwam.actor.hide import HidePolicy
    from xiconwam.engine.session import TrainSession

    def model():
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

    original = model()
    TrainSession._apply_dit_only_train_mode(original, freeze_action=True)
    loss, _ = original(_sample())
    loss.backward()
    assert all(
        parameter.grad is None for parameter in original.state_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in original.video_context_projector.parameters()
    )
    joint = tmp_path / "hide.pt"
    action = tmp_path / "hide-action.pt"
    video = tmp_path / "hide-video.safetensors"
    original.save_checkpoint(joint)
    split_joint_weights(joint, video_path=video, action_path=action)
    restored = model()
    restored.load_action_checkpoint(action)
    restored.load_video_checkpoint(video)
    torch.testing.assert_close(
        original.dit.state_dict(), restored.dit.state_dict(), rtol=0, atol=0
    )


@pytest.mark.parametrize("architecture", ["mot", "idm", "hide", "video"])
def test_model_factory_uses_the_selected_component_configs(monkeypatch, architecture):
    from hydra.utils import instantiate

    from xiconwam.vision.contract import BackboneComponents

    cfg = task("robodojo_wan", [f"model={architecture}"])
    video = cfg.model.video_dit
    video.attention_backend = "sdpa"
    for key, value in {
        "num_heads": 4,
        "attn_head_dim": 8,
        "num_layers": 1,
        "hidden_dim": 32,
        "freq_dim": 16,
        "text_dim": 32,
    }.items():
        video[key] = value
    if architecture != "video":
        cfg.model.action_dit.pretrained_path = None
        cfg.model.action_dit.dit_config.hidden_dim = 32
        cfg.model.action_dit.dit_config.action_dim = 14
        cfg.model.action_dit.dit_config.ffn_dim = 64
        cfg.model.action_dit.dit_config.projector_hidden_dim = 16
        cfg.model.state_dim = 14
        cfg.model.state_encoder_hidden_dim = 16
        if architecture == "hide":
            cfg.model.feature_layer = 0

    def components(*args, **kwargs):
        actor = _tiny(MoTPolicy)
        actor.vae.upsampling_factor = 8
        return BackboneComponents(
            name="wan22",
            dit=actor.video_expert,
            vae=actor.vae,
            text_encoder=None,
            tokenizer=None,
            text_dim=32,
        )

    monkeypatch.setattr("xiconwam.vision.registry.load_components", components)
    monkeypatch.setattr("xiconwam.actor.video.load_components", components)
    model = instantiate(cfg.model, device="cpu", model_dtype=torch.float32)
    sample = _sample()
    if architecture == "video":
        sample = {
            key: value
            for key, value in sample.items()
            if key in {"video", "context", "context_mask"}
        }
    loss, _ = model(sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.training_policy["mode"] == (
        "video_only" if architecture == "video" else "joint"
    )


@pytest.mark.parametrize("name", ["wan", "cosmos", "flux"])
def test_backbone_structure_drives_its_loader_and_portable_checkpoint(tmp_path, name):
    from xiconwam.actor.dit_checkpoint import save_video_weights
    from xiconwam.vision.registry import load_dit

    cfg = task(f"robodojo_{name}").model.video_dit
    cfg.attention_backend = "sdpa"
    if name == "wan":
        cfg.dit_config.hidden_dim = 32
        cfg.dit_config.ffn_dim = 64
        cfg.dit_config.in_dim = 16
        cfg.dit_config.out_dim = 16
        cfg.dit_config.num_heads = 4
        cfg.dit_config.attn_head_dim = 8
        cfg.dit_config.num_layers = 1
        cfg.dit_config.freq_dim = 16
        cfg.dit_config.text_dim = 32
        expected_width = 32
    elif name == "cosmos":
        cfg.dit_config.hidden_size = 128
        cfg.dit_config.num_heads = 1
        cfg.dit_config.num_layers = 1
        cfg.dit_config.intermediate_size = 256
        cfg.dit_config.context_dim = 32
        cfg.dit_config.reason_context_dim = 64
        cfg.dit_config.adaln_lora_dim = 8
        expected_width = 128
    else:
        cfg.dit_config.hidden_size = 32
        cfg.dit_config.num_heads = 4
        cfg.dit_config.context_in_dim = 32
        cfg.dit_config.in_channels = 16
        cfg.dit_config.depth = 1
        cfg.dit_config.depth_single_blocks = 1
        cfg.dit_config.axes_dim = [2, 2, 2, 2]
        cfg.attn_head_dim = 8
        expected_width = 32
    cfg.from_scratch = True
    model = load_dit(cfg, device="cpu", torch_dtype=torch.float32)
    assert model.hidden_dim == expected_width
    if name == "wan":
        checkpoint = tmp_path / "wan" / "diffusion_pytorch_model.safetensors"
        checkpoint.parent.mkdir()
        save_video_weights(model, checkpoint)
        cfg.model_id = str(checkpoint.parent)
    else:
        from scripts.split_dit_checkpoint import export_video

        checkpoint = tmp_path / "trained.safetensors"
        save_video_weights(model, checkpoint)
        config = OmegaConf.to_container(cfg, resolve=True)
        if name == "cosmos":
            base = tmp_path / "base"
            base.mkdir()
            (base / "tokenizer.pth").write_bytes(b"vae fixture")
            config["model_id"] = str(base)
            output = tmp_path / "cosmos-model"
            export_video(
                config,
                model.state_dict(),
                {"backbone_name": cfg.name},
                output,
                base_model=base,
            )
            assert (output / "tokenizer.pth").read_bytes() == b"vae fixture"
            cfg.model_id = str(output)
        else:
            output = tmp_path / "flux.safetensors"
            export_video(
                config, model.state_dict(), {"backbone_name": cfg.name}, output
            )
            cfg.model_path = str(output)
    cfg.from_scratch = False
    restored = load_dit(cfg, device="cpu", torch_dtype=torch.float32)
    torch.testing.assert_close(
        model.state_dict(), restored.state_dict(), rtol=0, atol=0
    )


def test_checkpoint_completion_marker_follows_all_rank_writes(tmp_path):
    from xiconwam.engine.checkpoint import CheckpointMixin

    events = []
    accelerator = SimpleNamespace(
        is_main_process=True,
        wait_for_everyone=lambda: events.append("barrier"),
        save_state=lambda **kwargs: events.append("save_state"),
    )
    session = SimpleNamespace(
        global_step=1,
        state_dir=str(tmp_path),
        accelerator=accelerator,
        _save_weights_checkpoint=lambda **kwargs: "weights.pt",
        _stagger_zero_checkpoint_writes=lambda: None,
        _save_trainer_state=lambda path: events.append("complete"),
        _prune_saved_checkpoints=lambda: None,
    )
    CheckpointMixin.save_checkpoint(session)
    write = events.index("save_state")
    complete = events.index("complete")
    assert "barrier" in events[write + 1 : complete]


def test_failed_checkpoint_rewrite_invalidates_the_old_completion_marker(tmp_path):
    from xiconwam.engine.checkpoint import CheckpointMixin
    from xiconwam.engine.configuration import resolve_resume_path

    state = tmp_path / "step_000001"
    state.mkdir()
    (state / "trainer_state.json").write_text('{"global_step": 1}')

    def fail(**kwargs):
        raise OSError("simulated disk failure")

    accelerator = SimpleNamespace(
        is_main_process=True,
        wait_for_everyone=lambda: None,
        save_state=fail,
    )
    session = SimpleNamespace(
        global_step=1,
        state_dir=str(tmp_path),
        accelerator=accelerator,
        _save_weights_checkpoint=lambda **kwargs: "weights.pt",
        _stagger_zero_checkpoint_writes=lambda: None,
    )
    with pytest.raises(OSError, match="disk failure"):
        CheckpointMixin.save_checkpoint(session)
    assert not (state / "trainer_state.json").exists()
    with pytest.raises(ValueError, match="No complete"):
        resolve_resume_path(tmp_path)


def test_old_video_checkpoint_override_is_rejected_but_action_checkpoint_is_valid():
    from xiconwam.vision.registry import normalize_spec

    model = task("robodojo_wan").model
    model.action_dit.checkpoint = "/tmp/action.pt"
    assert action_dit(model)["checkpoint"] == "/tmp/action.pt"
    cfg = model.video_dit
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.checkpoint = "/tmp/old-video.safetensors"
    with pytest.raises(ValueError, match="export a complete model directory"):
        normalize_spec(cfg)


def test_one_command_splits_joint_into_loadable_wan_and_action(tmp_path):
    import os
    import subprocess
    import sys

    from xiconwam.vision.registry import load_dit

    original = _tiny(MoTPolicy)
    joint = tmp_path / "joint.pt"
    original.save_checkpoint(joint, step=9)
    cfg = task("robodojo_wan")
    video_cfg = cfg.model.video_dit
    video_cfg.attention_backend = "sdpa"
    for key, value in {
        "hidden_dim": 32,
        "ffn_dim": 64,
        "in_dim": 16,
        "out_dim": 16,
        "num_heads": 4,
        "attn_head_dim": 8,
        "num_layers": 1,
        "freq_dim": 16,
        "text_dim": 32,
    }.items():
        video_cfg.dit_config[key] = value
    saved_config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, saved_config, resolve=True)
    base = tmp_path / "base"
    tokenizer = base / "google/umt5-xxl"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}")
    (tokenizer / "tokenizer.json").write_text("{}")
    (base / "Wan2.2_VAE.pth").write_bytes(b"vae")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"encoder")
    video = tmp_path / "video-model"
    action = tmp_path / "action.pt"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/split_dit_checkpoint.py"),
            str(joint),
            "--backbone-config",
            str(saved_config),
            "--base-model",
            str(base),
            "--video-model",
            str(video),
            "--action-output",
            str(action),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    video_cfg.model_id = str(video)
    video_cfg.from_scratch = False
    restored_video = load_dit(video_cfg, device="cpu", torch_dtype=torch.float32)
    torch.testing.assert_close(
        original.video_expert.state_dict(), restored_video.state_dict(), rtol=0, atol=0
    )
    restored = _tiny(MoTPolicy)
    restored.load_action_checkpoint(action)
    torch.testing.assert_close(
        original.action_expert.state_dict(),
        restored.action_expert.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.state_encoder.state_dict(),
        restored.state_encoder.state_dict(),
        rtol=0,
        atol=0,
    )
    assert (video / "Wan2.2_VAE.pth").read_bytes() == b"vae"
    assert (video / "models_t5_umt5-xxl-enc-bf16.pth").read_bytes() == b"encoder"


def test_legacy_flat_action_resume_uses_original_backbone(tmp_path):
    run = tmp_path / "old-run"
    state = run / "checkpoints/state/step_000010"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text('{"global_step": 10}')
    saved = OmegaConf.to_container(task("robodojo_wan"), resolve=True)
    model = saved["model"]
    model["backbone"] = model.pop("video_dit")
    action = model.pop("action_dit")
    model["action_dit_config"] = action["dit_config"]
    model["action_dit_checkpoint"] = "old-action.pt"
    OmegaConf.save(saved, run / "config.yaml")
    restored = resolve_training_config(task("robodojo_wan", [f"resume={run}"]))
    assert restored.model.skip_dit_load_from_pretrain
    assert restored.model.backbone.video_dit_checkpoint is None
    assert restored.model.action_dit_checkpoint is None
