import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from xiconwam.data.lerobot.resources import reuse_during_construction
from xiconwam.model_config import as_dict as _as_dict
from xiconwam.kit.logging_config import get_logger, setup_logging
from xiconwam.kit.pytorch_utils import (
    mixed_precision_to_dtype as _mixed_precision_to_model_dtype,
    normalize_mixed_precision as _normalize_mixed_precision,
)

logger = get_logger(__name__)


def _lora_payload(lora, backbone):
    if lora is None:
        return None
    cfg = _as_dict(lora)
    if not bool(cfg.get("enable", False)):
        return None
    target = cfg.get("target", "video")
    if target in (None, "video"):
        modules = backbone.get("lora_target_modules")
    elif isinstance(target, (list, tuple)):
        modules = list(target)
    else:
        modules = [str(target)]
    if not modules:
        raise ValueError(
            "LoRA is enabled, but `lora.target` did not match any modules."
        )
    rank = int(cfg.get("rank", 16))
    return {
        "r": rank,
        "lora_alpha": int(cfg.get("alpha", rank)),
        "lora_dropout": float(cfg.get("dropout", 0.0)),
        "bias": "none",
        "target_modules": list(modules),
    }


def _apply_lora(model, lora):
    if lora is None:
        return model
    from xiconwam.actor.lora import inject_video_dit_lora

    video = getattr(model, "video_dit", None)
    if video is None:
        video = model.dit["video_dit"]
    inject_video_dit_lora(video, lora)
    logger.info(
        "Video tower LoRA enabled: rank=%s alpha=%s targets=%s",
        lora["r"],
        lora["lora_alpha"],
        lora["target_modules"],
    )
    return model


def build_model(
    architecture: str = "mot",
    backbone=None,
    attention_backend: str = "sdpa",
    state_dim: int | None = None,
    state_position: str = "context",
    projector_hidden_dim: int = 64,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    action_dit_checkpoint: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    lora=None,
    video_cond_noise_prob: float = 0.5,
    video_hidden_layer: int = 17,
    action_dim: int | None = None,
    feature_timestep: float = 1.0,
    extract_layer: int | None = None,
    repeated_diffusion_steps: int = 1,
    action_inference_steps: int = 4,
    noise_beta_alpha: float = 1.5,
    noise_beta_beta: float = 1.0,
    noise_s: float = 0.999,
    group_learning_rate=None,
    learning_rate_scale=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    video_backbone=None,
    action_backbone=None,
    action_dit=None,
    video_dit=None,
    state_encoder_hidden_dim=None,
    feature_layer=None,
    training=None,
    **unused,
):
    """Build a joint policy or a standalone video model."""
    del attention_backend
    if unused:
        raise TypeError(f"Unknown model configuration keys: {sorted(unused)}")
    if state_encoder_hidden_dim is not None:
        projector_hidden_dim = int(state_encoder_hidden_dim)
    name = str(architecture).strip().lower()
    if name not in {"mot", "idm", "hide", "video"}:
        raise ValueError(
            f"`architecture` must be mot, idm, hide, or video, got {architecture!r}."
        )
    from xiconwam.model_config import (
        resolve_action_config,
        resolve_feature_layer,
        validate_architecture,
    )

    video_specs = [
        spec for spec in (video_dit, video_backbone, backbone) if spec is not None
    ]
    if len(video_specs) != 1:
        raise ValueError(
            "Specify exactly one video_dit (legacy: backbone/video_backbone)"
        )
    backbone_cfg = _as_dict(video_specs[0])
    component_lora = backbone_cfg.pop("lora", None)
    if lora is not None and component_lora is not None:
        raise ValueError("Specify LoRA once, under model.video_dit.lora")
    lora = component_lora if component_lora is not None else lora
    paired_action = backbone_cfg.pop("action_dit", None)
    if action_dit is not None:
        paired_action = _as_dict(action_dit)
    elif action_backbone is not None:
        paired_action = _as_dict(action_backbone)
    if name != "video" and paired_action is not None:
        action_dit_config = resolve_action_config(backbone_cfg, paired_action)
        validate_architecture(name, action_dit_config)
        action_dit_checkpoint = paired_action.get("checkpoint")
        action_dit_pretrained_path = paired_action.get("pretrained_path")
        if paired_action.get("from_scratch", False):
            action_dit_pretrained_path = None
        action_scheduler = paired_action.get("scheduler")
    video_scheduler = video_scheduler or backbone_cfg.get("video_scheduler")
    training_cfg = _as_dict(training)
    if name == "video":
        from xiconwam.actor.video import VideoModel

        if (
            action_dit_checkpoint is not None
            or _lora_payload(lora, backbone_cfg) is not None
        ):
            raise ValueError(
                "Standalone video training requires full Video DiT weights"
            )
        model = VideoModel.from_backbone(
            backbone_cfg,
            device=device,
            torch_dtype=model_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        model.training_policy = training_cfg or {"mode": "video_only"}
        return model
    action_cfg = _as_dict(action_dit_config) if action_dit_config is not None else None
    common = dict(
        backbone=backbone_cfg,
        state_dim=state_dim,
        state_position=state_position,
        projector_hidden_dim=projector_hidden_dim,
        action_dit_config=action_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
        action_dit_checkpoint=action_dit_checkpoint,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        model_dtype=model_dtype,
        device=device,
    )
    if name == "hide":
        from xiconwam.actor.hide import HidePolicy

        if state_dim is None:
            raise ValueError("`state_dim` is required.")
        if action_cfg is None:
            raise ValueError("hide requires `action_dit_config`.")
        if action_dim is None:
            action_dim = action_cfg["action_dim"]
        if int(action_dim) != int(action_cfg["action_dim"]):
            raise ValueError("HIDE action_dim must match Action DiT action_dim")
        if feature_layer is not None:
            extract_layer = resolve_feature_layer(backbone_cfg, feature_layer)
        video_scheduler_cfg = _as_dict(video_scheduler)
        action_scheduler_cfg = _as_dict(action_scheduler)
        loss_cfg = _as_dict(loss)
        model = HidePolicy.from_backbone_pretrained(
            backbone=backbone_cfg,
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            action_dit_checkpoint=action_dit_checkpoint,
            action_dim=int(action_dim),
            state_dim=int(state_dim),
            state_position=state_position,
            projector_hidden_dim=int(projector_hidden_dim),
            video_hidden_layer=int(
                extract_layer if extract_layer is not None else video_hidden_layer
            ),
            feature_timestep=float(feature_timestep),
            repeated_diffusion_steps=int(repeated_diffusion_steps),
            action_inference_steps=int(action_inference_steps),
            noise_beta_alpha=float(noise_beta_alpha),
            noise_beta_beta=float(noise_beta_beta),
            noise_s=float(noise_s),
            group_learning_rate=(
                _as_dict(group_learning_rate) if group_learning_rate else None
            ),
            learning_rate_scale=(
                _as_dict(learning_rate_scale) if learning_rate_scale else None
            ),
            skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
            device=device,
            torch_dtype=model_dtype,
            video_train_shift=float(video_scheduler_cfg.get("train_shift", 5.0)),
            video_infer_shift=float(video_scheduler_cfg.get("infer_shift", 5.0)),
            video_num_train_timesteps=int(
                video_scheduler_cfg.get("num_train_timesteps", 1000)
            ),
            action_train_shift=float(action_scheduler_cfg.get("train_shift", 5.0)),
            action_infer_shift=float(action_scheduler_cfg.get("infer_shift", 5.0)),
            action_num_train_timesteps=int(
                action_scheduler_cfg.get("num_train_timesteps", 1000)
            ),
            loss_lambda_video=float(loss_cfg.get("lambda_video", 1.0)),
            loss_lambda_action=float(loss_cfg.get("lambda_action", 1.0)),
        )
    elif name == "idm":
        from xiconwam.actor.idm import IDMPolicy

        model = _build_policy(
            IDMPolicy, video_cond_noise_prob=video_cond_noise_prob, **common
        )
    else:
        from xiconwam.actor.mot import MoTPolicy

        model = _build_policy(MoTPolicy, **common)
    model.training_policy = training_cfg
    return _apply_lora(model, _lora_payload(lora, backbone_cfg))


def _build_policy(policy_cls, **kwargs):
    if kwargs.get("backbone") is None:
        raise ValueError("`backbone` configuration is required.")
    if kwargs.get("action_dit_config") is None:
        raise ValueError("`action_dit_config` is required.")
    video_scheduler = _as_dict(kwargs.get("video_scheduler"))
    action_scheduler = _as_dict(kwargs.get("action_scheduler"))
    loss = _as_dict(kwargs.get("loss"))
    init_kwargs = {}
    if (
        "video_cond_noise_prob" in kwargs
        and kwargs["video_cond_noise_prob"] is not None
    ):
        init_kwargs["video_cond_noise_prob"] = float(kwargs["video_cond_noise_prob"])
    return policy_cls.from_backbone_pretrained(
        backbone=_as_dict(kwargs["backbone"]),
        device=kwargs.get("device", "cuda"),
        torch_dtype=kwargs.get("model_dtype", torch.bfloat16),
        state_dim=kwargs.get("state_dim"),
        state_position=kwargs.get("state_position", "context"),
        projector_hidden_dim=int(kwargs.get("projector_hidden_dim", 64)),
        action_dit_config=_as_dict(kwargs["action_dit_config"]),
        action_dit_pretrained_path=kwargs.get("action_dit_pretrained_path"),
        action_dit_checkpoint=kwargs.get("action_dit_checkpoint"),
        skip_dit_load_from_pretrain=bool(
            kwargs.get("skip_dit_load_from_pretrain", False)
        ),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(
            action_scheduler.get("num_train_timesteps", 1000)
        ),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        **init_kwargs,
    )


def build_mot(**kwargs):
    """Compatibility target for saved Hydra configs; prefer build_model."""
    return _build_legacy_policy("mot", kwargs)


def build_idm(**kwargs):
    """Compatibility target for saved Hydra configs; prefer build_model."""
    return _build_legacy_policy("idm", kwargs)


def build_hide(**kwargs):
    """Compatibility target for saved Hydra configs; prefer build_model."""
    kwargs.setdefault(
        "action_dim", _as_dict(kwargs.get("action_dit_config")).get("action_dim")
    )
    return _build_legacy_policy("hide", kwargs)


def _build_legacy_policy(architecture, kwargs):
    # Historical targets take PEFT kwargs directly; build_model uses enable/rank.
    lora = kwargs.pop("lora", None)
    return _apply_lora(build_model(architecture=architecture, **kwargs), lora)


def build_actor(**kwargs):
    """Compatibility target preserving the original actor checkpoint variant."""
    from xiconwam.actor.world import XiconActor

    return _build_policy(XiconActor, **kwargs)


def validation_dataset_config(data_cfg: DictConfig) -> DictConfig:
    """Val uses the train spec with training-set sampling turned off.

    The node stays attached to the root config so `${model...}` interpolations
    still resolve. A written `data.val` block replaces this copy.
    """
    if data_cfg.get("val") is not None:
        return data_cfg.val
    copied = OmegaConf.create(OmegaConf.to_container(data_cfg.train, resolve=False))
    # Hydra loads this node in struct mode, so the derived key has to be opened.
    was_struct = OmegaConf.is_struct(data_cfg)
    OmegaConf.set_struct(data_cfg, False)
    data_cfg.val = copied
    data_cfg.val.is_training_set = False
    if was_struct:
        OmegaConf.set_struct(data_cfg, True)
    return data_cfg.val


@reuse_during_construction
def build_datasets(data_cfg: DictConfig):
    from accelerate import PartialState
    from xiconwam.kit import misc

    train_ds = instantiate(data_cfg.train)
    PartialState().wait_for_everyone()  # Rank 0 must finish writing statistics first.
    # Bind the run config to the exact saved training statistics. Mixtures need
    # one path per source, including when statistics were computed on this run.
    source_stats = {}
    if hasattr(train_ds, "datasets") and hasattr(train_ds, "names"):
        for name, source in zip(train_ds.names, train_ds.datasets):
            path = getattr(source, "norm_stats_path", None)
            if path:
                source_stats[name] = path
                OmegaConf.update(
                    data_cfg.train.sources[name].config,
                    "pretrained_norm_stats",
                    path,
                    force_add=True,
                )
    elif getattr(train_ds, "norm_stats_path", None):
        OmegaConf.update(
            data_cfg.train,
            "pretrained_norm_stats",
            train_ds.norm_stats_path,
            force_add=True,
        )
    if getattr(train_ds, "is_video_manifest", False):
        return train_ds, instantiate(data_cfg.get("val"))
    val_cfg = validation_dataset_config(data_cfg)
    train_stats_path = data_cfg.train.get("pretrained_norm_stats")
    default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
    val_stats_path = val_cfg.get("pretrained_norm_stats")
    target = str(data_cfg.train.get("_target_", ""))
    is_mixture = target.endswith("build_mixture_dataset") or target.endswith(
        "MixtureDataset"
    )
    kwargs = {}
    if is_mixture and source_stats:
        kwargs["source_norm_stats"] = source_stats
    if val_stats_path or train_stats_path:
        kwargs["pretrained_norm_stats"] = val_stats_path or train_stats_path
    elif not is_mixture:
        kwargs["pretrained_norm_stats"] = default_stats_path
    if kwargs.get("pretrained_norm_stats") is not None:
        logger.info(
            "Building val dataset with pretrained_norm_stats: %s",
            kwargs["pretrained_norm_stats"],
        )
    elif is_mixture:
        logger.info(
            "Building mixture val dataset; each source keeps its own pretrained_norm_stats."
        )
    val_ds = instantiate(data_cfg.val, **kwargs)
    return train_ds, val_ds


def run_training(cfg: DictConfig):
    from accelerate import Accelerator

    from xiconwam.actor.helpers.inference import configure_model_execution
    from xiconwam.engine.session import TrainSession
    from xiconwam.kit import misc

    from hydra.core.hydra_config import HydraConfig
    from xiconwam.engine.configuration import (
        resolve_training_config,
        validate_training_config,
    )

    overrides = HydraConfig.get().overrides.task if HydraConfig.initialized() else []
    cfg = resolve_training_config(cfg, overrides)
    validate_training_config(cfg)
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    setup_logging(
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )
    misc.register_work_dir(cfg.output_dir)
    if accelerator.is_main_process:
        config_payload = OmegaConf.to_container(cfg, resolve=True)
        with open(Path(cfg.output_dir) / "config.yaml", "w") as handle:
            OmegaConf.save(config_payload, handle)

    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(
        cfg.model, model_dtype=model_dtype, device=str(accelerator.device)
    )
    model = configure_model_execution(
        model,
        vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1),
        inference_cross_kv_reuse=cfg.get("inference_cross_kv_reuse", True),
    )
    train_ds, val_ds = build_datasets(cfg.data)
    if accelerator.is_main_process:
        # Include derived normalization bindings in the deployable run config.
        OmegaConf.save(
            OmegaConf.to_container(cfg, resolve=True),
            Path(cfg.output_dir) / "config.yaml",
        )
    trainer = TrainSession(
        cfg=cfg,
        accelerator=accelerator,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    try:
        trainer.train()
    finally:
        accelerator.end_training()
