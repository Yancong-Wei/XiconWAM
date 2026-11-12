"""Compose a run and recover its saved model and optimizer configuration."""

import logging
from pathlib import Path

from omegaconf import OmegaConf

from xiconwam.model_config import action_dit, video_backbone

logger = logging.getLogger(__name__)

_RUNTIME_OVERRIDES = {
    "output_dir",
    "max_steps",
    "max_epochs",
    "num_workers",
    "log_every",
    "save_every",
    "save_every_epochs",
    "checkpoint_save_limit",
    "eval_every",
    "eval_num_samples",
    "eval_seed",
    "eval_save_video",
    "eval_num_inference_steps",
    "dataloader_prefetch_factor",
    "dataloader_persistent_workers",
    "dataloader_pin_memory",
    "dataloader_worker_threads",
}


def validate_training_config(config):
    """Reject invalid or ineffective settings before loading model/data assets."""
    import inspect
    import math

    from xiconwam.launch import build_model
    from xiconwam.model_config import (
        as_dict,
        resolve_action_config,
        resolve_feature_layer,
        validate_architecture,
    )
    from xiconwam.data.lerobot.robot_video_dataset import (
        configured_window_steps,
        trained_window,
    )
    from xiconwam.vision.registry import normalize_spec

    def positive(name, value, *, zero=False, integer=False):
        number = float(value)
        if (
            not math.isfinite(number)
            or (number < 0 if zero else number <= 0)
            or (integer and not number.is_integer())
        ):
            raise ValueError(
                f"{name} must be {'nonnegative' if zero else 'positive'}, got {value}"
            )

    model = config.model
    if model.get("_target_") == "xiconwam.launch.build_model":
        unknown = (
            set(model)
            - set(inspect.signature(build_model).parameters)
            - {"_target_", "_recursive_", "_convert_", "_partial_"}
        )
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
    video = normalize_spec(video_backbone(model))
    architecture = model.get("architecture", "mot")
    if architecture not in {"mot", "idm", "hide", "video"}:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    action = action_dit(model)
    if action is not None:
        action_config = resolve_action_config(video, action)
        validate_architecture(architecture, action_config)
        positive("action_dit.action_dim", action_config["action_dim"])
        if action.get("from_scratch") and action.get("checkpoint"):
            raise ValueError(
                "action_dit.from_scratch conflicts with action_dit.checkpoint"
            )
    training = as_dict(model.get("training"))
    mode = training.get("mode", "video_only" if architecture == "video" else "joint")
    if mode not in {"joint", "action_only", "video_only"} or (
        (architecture == "video") != (mode == "video_only")
    ):
        raise ValueError("model.training.mode must match the selected architecture")
    freeze_video = training.get("freeze_video", False) or mode == "action_only"
    freeze_action = training.get("freeze_action", False)
    if (freeze_video and freeze_action) or (architecture == "video" and freeze_video):
        raise ValueError("At least one DiT must remain trainable")
    lora = video.get("lora", model.get("lora", {})) or {}
    if lora.get("enable", False):
        if architecture == "video":
            raise ValueError("Standalone video training does not support LoRA")
        positive("video_dit.lora.rank", lora.get("rank", 16))
        positive("video_dit.lora.alpha", lora.get("alpha", lora.get("rank", 16)))
    if architecture == "hide":
        resolve_feature_layer(
            video, model.get("feature_layer", model.get("extract_layer", 17))
        )
        if not 0 <= float(model.get("feature_timestep", 1)) <= 1:
            raise ValueError("HIDE feature_timestep must be in [0, 1]")
        for key in (
            "repeated_diffusion_steps",
            "action_inference_steps",
            "noise_beta_alpha",
            "noise_beta_beta",
            "noise_s",
        ):
            positive(f"model.{key}", model[key], integer=key.endswith("steps"))
        for key, value in (model.get("learning_rate_scale") or {}).items():
            if key not in {"video", "action"}:
                raise ValueError(f"Unknown HIDE learning_rate_scale group: {key}")
            positive(f"learning_rate_scale.{key}", value, zero=True)
    if (
        architecture == "idm"
        and not 0 <= float(model.get("video_cond_noise_prob", 0.5)) <= 1
    ):
        raise ValueError("IDM video_cond_noise_prob must be in [0, 1]")
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "log_every",
        "save_every",
        "max_grad_norm",
    ):
        positive(
            key,
            config[key],
            integer=key
            in {"batch_size", "gradient_accumulation_steps", "log_every", "save_every"},
        )
    for key in ("num_workers", "eval_every", "eval_num_samples", "weight_decay"):
        positive(key, config.get(key, 0), zero=True, integer=key != "weight_decay")
    if not 0 <= float(config.warmup_ratio) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config.lr_scheduler_type not in {"constant", "cosine"}:
        raise ValueError("lr_scheduler_type must be constant or cosine")
    if config.get("max_steps") is None and config.get("max_epochs") is None:
        raise ValueError("Set max_steps or max_epochs")
    for key in (
        "max_steps",
        "max_epochs",
        "save_every_epochs",
        "checkpoint_save_limit",
    ):
        if config.get(key) is not None:
            positive(
                key, config[key], integer=key in {"max_steps", "checkpoint_save_limit"}
            )
    sources = [config.data.train]
    if str(config.data.train.get("_target_", "")).endswith("build_mixture_dataset"):
        sources = [
            entry.config
            for entry in config.data.train.sources.values()
            if entry.get("enabled", True)
        ]
        if not sources:
            raise ValueError("Mixture needs at least one enabled source")
    for data in sources:
        target = str(data.get("_target_", ""))
        if target.endswith("VideoManifestDataset") and architecture != "video":
            raise ValueError("VideoManifestDataset has no actions; select model=video")
        if target.endswith("EpisodeWindowDataset"):
            trained_window(configured_window_steps(data), data.action_video_freq_ratio)
            if not 0 <= float(data.val_set_proportion) < 1:
                raise ValueError("data.val_set_proportion must be in [0, 1)")
            positive("data.global_sample_stride", data.global_sample_stride)


def resolve_resume_path(resume):
    path = Path(str(resume)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    if path.is_file() or (path / "trainer_state.json").is_file():
        return path
    state_root = (
        path / "checkpoints/state" if (path / "checkpoints/state").is_dir() else path
    )
    states = [
        p
        for p in state_root.glob("step_*")
        if p.is_dir() and p.name[5:].isdigit() and (p / "trainer_state.json").is_file()
    ]
    if not states:
        raise ValueError(f"No complete training state found under {path}")
    return max(states, key=lambda p: int(p.name[5:]))


def resume_run_config(path):
    for parent in path.parents:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def resolve_training_config(config, overrides=()):
    if not config.get("resume"):
        return config
    path = resolve_resume_path(config.resume)
    saved_path = resume_run_config(path)
    if saved_path is None:
        # Standalone weight files can initialize the architecture selected in YAML.
        if path.is_file():
            config.resume = str(path)
            return config
        raise FileNotFoundError(
            f"Full training resume needs the run config.yaml above {path}"
        )
    restored = OmegaConf.load(saved_path)
    OmegaConf.resolve(restored)
    legacy = restored.get("acceleration")
    if legacy is not None and legacy.get("framework", "deepspeed") == "deepspeed":
        restored.zero = legacy.get("zero", restored.get("zero", 2))
        restored.zero_offload = legacy.get(
            "offload", restored.get("zero_offload", False)
        )
        restored.pop("acceleration")
    for raw in overrides:
        key = raw.lstrip("+").split("=", 1)[0]
        if key == "resume":
            continue
        if key in _RUNTIME_OVERRIDES or key.startswith("wandb."):
            OmegaConf.update(
                restored, key, OmegaConf.select(config, key), force_add=True
            )
        elif key.startswith("hydra."):
            continue
        elif OmegaConf.select(restored, key) != OmegaConf.select(config, key):
            raise ValueError(
                f"Full resume restores the saved {key}. Use backbone checkpoint paths to start a new experiment."
            )
    restored.resume = str(path)
    if path.is_dir():
        # All DiT and optimizer weights will come from the complete training state.
        model = restored.model
        video = video_backbone(model)
        if (
            model.get("video_backbone") is not None
            or model.get("action_dit") is not None
            or "action_dit" in (video or {})
            or model.get("architecture") == "video"
        ):
            video.pop("checkpoint", None)
            video.pop("video_dit_checkpoint", None)
            video.from_scratch = True
            action = action_dit(model)
            if action is not None:
                action.checkpoint = None
                action.pretrained_path = None
                action.from_scratch = True
        else:
            restored.model.skip_dit_load_from_pretrain = True
            if restored.model.get("backbone") is not None:
                restored.model.backbone.video_dit_checkpoint = None
            if "action_dit_checkpoint" in restored.model:
                restored.model.action_dit_checkpoint = None
    logger.info("Restoring run configuration from %s with resume=%s", saved_path, path)
    return restored
