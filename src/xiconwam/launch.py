import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from xiconwam.data.lerobot.resources import reuse_during_construction
from xiconwam.kit.logging_config import get_logger, setup_logging
from xiconwam.kit.pytorch_utils import (
    mixed_precision_to_dtype as _mixed_precision_to_model_dtype,
    normalize_mixed_precision as _normalize_mixed_precision,
)

logger = get_logger(__name__)


def _as_dict(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value or {})


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
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **unused,
):
    """Build mot, idm, or hide. The visual tower comes from `backbone`."""
    del attention_backend, unused
    name = str(architecture).strip().lower()
    if name not in {"mot", "idm", "hide"}:
        raise ValueError(
            f"`architecture` must be mot, idm, or hide, got {architecture!r}."
        )
    backbone_cfg = _as_dict(backbone)
    action_cfg = _as_dict(action_dit_config) if action_dit_config is not None else None
    common = dict(
        backbone=backbone_cfg,
        state_dim=state_dim,
        state_position=state_position,
        projector_hidden_dim=projector_hidden_dim,
        action_dit_config=action_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
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
            raise ValueError("hide requires `action_dim`.")
        video_scheduler_cfg = _as_dict(video_scheduler)
        action_scheduler_cfg = _as_dict(action_scheduler)
        loss_cfg = _as_dict(loss)
        model = HidePolicy.from_backbone_pretrained(
            backbone=backbone_cfg,
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
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


def build_mot(
    backbone=None,
    attention_backend: str = "sdpa",
    state_dim: int | None = None,
    state_position: str = "context",
    projector_hidden_dim: int = 64,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **unused,
):
    """Action stream reads the full video stream while both are denoised."""
    del attention_backend
    from xiconwam.actor.mot import MoTPolicy

    model = _build_policy(
        MoTPolicy,
        backbone=backbone,
        state_dim=state_dim,
        state_position=state_position,
        projector_hidden_dim=projector_hidden_dim,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        model_dtype=model_dtype,
        device=device,
    )
    return _apply_lora(model, unused.get("lora"))


def build_idm(
    video_cond_noise_prob: float = 0.5,
    backbone=None,
    attention_backend: str = "sdpa",
    state_dim: int | None = None,
    state_position: str = "context",
    projector_hidden_dim: int = 64,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **unused,
):
    """Teacher-forced inverse dynamics: predict the action from the video future."""
    del attention_backend
    from xiconwam.actor.idm import IDMPolicy

    model = _build_policy(
        IDMPolicy,
        backbone=backbone,
        state_dim=state_dim,
        state_position=state_position,
        projector_hidden_dim=projector_hidden_dim,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        model_dtype=model_dtype,
        device=device,
        video_cond_noise_prob=video_cond_noise_prob,
    )
    return _apply_lora(model, unused.get("lora"))


def build_hide(
    backbone=None,
    attention_backend: str = "sdpa",
    state_dim: int | None = None,
    state_position: str = "context",
    projector_hidden_dim: int = 64,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_hidden_layer: int = 17,
    feature_timestep: float = 1.0,
    repeated_diffusion_steps: int = 1,
    action_inference_steps: int = 4,
    noise_beta_alpha: float = 1.5,
    noise_beta_beta: float = 1.0,
    noise_s: float = 0.999,
    group_learning_rate=None,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    lora=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **unused,
):
    """Video DiT hidden states condition a separate action DiT."""
    del attention_backend, unused
    from xiconwam.actor.hide import HidePolicy

    if backbone is None or state_dim is None or action_dit_config is None:
        raise ValueError(
            "`backbone`, `state_dim`, and `action_dit_config` are required."
        )
    action_cfg = _as_dict(action_dit_config)
    video_scheduler = _as_dict(video_scheduler)
    action_scheduler = _as_dict(action_scheduler)
    loss = _as_dict(loss)
    model = HidePolicy.from_backbone_pretrained(
        backbone=_as_dict(backbone),
        action_dit_config=action_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
        action_dim=int(action_cfg["action_dim"]),
        state_dim=int(state_dim),
        state_position=state_position,
        projector_hidden_dim=int(projector_hidden_dim),
        video_hidden_layer=int(video_hidden_layer),
        feature_timestep=float(feature_timestep),
        repeated_diffusion_steps=int(repeated_diffusion_steps),
        action_inference_steps=int(action_inference_steps),
        noise_beta_alpha=float(noise_beta_alpha),
        noise_beta_beta=float(noise_beta_beta),
        noise_s=float(noise_s),
        group_learning_rate=(
            _as_dict(group_learning_rate) if group_learning_rate else None
        ),
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
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
    )
    return _apply_lora(model, lora)


def build_actor(
    backbone=None,
    attention_backend: str = "sdpa",
    state_dim: int | None = None,
    state_position: str = "context",
    projector_hidden_dim: int = 64,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build the dual-stream world-action actor from a Wan2.2 tower."""
    del attention_backend
    from xiconwam.actor.world import XiconActor

    if backbone is None:
        raise ValueError("`backbone` configuration is required.")
    if action_dit_config is None:
        raise ValueError("`action_dit_config` is required.")
    video_scheduler = _as_dict(video_scheduler)
    action_scheduler = _as_dict(action_scheduler)
    loss = _as_dict(loss)
    return XiconActor.from_backbone_pretrained(
        backbone=_as_dict(backbone),
        device=device,
        torch_dtype=model_dtype,
        state_dim=state_dim,
        state_position=state_position,
        projector_hidden_dim=projector_hidden_dim,
        action_dit_config=_as_dict(action_dit_config),
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
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
    )


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
    from xiconwam.kit import misc

    train_ds = instantiate(data_cfg.train)
    val_cfg = validation_dataset_config(data_cfg)
    train_stats_path = data_cfg.train.get("pretrained_norm_stats")
    default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
    val_stats_path = val_cfg.get("pretrained_norm_stats")
    target = str(data_cfg.train.get("_target_", ""))
    is_mixture = target.endswith("build_mixture_dataset") or target.endswith(
        "MixtureDataset"
    )
    kwargs = {}
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


def run_inference(cfg: DictConfig):
    import numpy as np
    from einops import repeat
    from PIL import Image

    from xiconwam.actor.helpers.inference import (
        configure_inference_compile_from_config,
        configure_model_execution,
    )
    from xiconwam.kit.video_io import save_mp4

    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(
        cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device)
    )
    model = configure_model_execution(
        model,
        vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1),
        inference_cross_kv_reuse=cfg.get("inference_cross_kv_reuse", True),
    )
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    model = configure_inference_compile_from_config(model, inference_cfg)

    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize(
            (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
        )
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(
        input_image, width=inference_cfg.width, height=inference_cfg.height
    )
    arr = np.array(input_image, dtype=np.float32)
    pixels = torch.from_numpy(arr).to(device=model.device, dtype=model.torch_dtype)
    pixels = pixels * (2.0 / 255.0) - 1.0
    pixels = repeat(pixels, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)
    infer_out = model.infer(
        prompt=str(inference_cfg.prompt),
        negative_prompt=str(inference_cfg.negative_prompt),
        text_cfg_scale=float(inference_cfg.text_cfg_scale),
        action_cfg_scale=float(inference_cfg.action_cfg_scale),
        input_image=pixels,
        num_frames=int(inference_cfg.num_frames),
        num_inference_steps=int(inference_cfg.num_inference_steps),
        sigma_shift=(
            None
            if inference_cfg.get("sigma_shift") is None
            else float(inference_cfg.sigma_shift)
        ),
        seed=int(inference_cfg.seed),
        rand_device=str(inference_cfg.rand_device),
    )
    save_mp4(infer_out["video"], output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
