from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

TOWERS = ("wan22", "cosmos25", "flux2")


def normalize_spec(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError(f"backbone config must be dict-like, got {type(config)}")
    result = dict(config)
    for key in ("checkpoint", "video_dit_checkpoint"):
        if result.get(key) is not None:
            raise ValueError(
                f"Video backbone {key!r} is no longer supported; export a complete "
                "model directory and set model.video_dit.model_id instead. "
                "Action checkpoint and full training resume remain supported."
            )
    name = str(result["name"]).strip().lower()
    if name not in TOWERS:
        raise ValueError(f"Xicon vision tower must be one of {TOWERS}, got {name!r}.")
    common = {
        "name",
        "from_scratch",
        "tokenizer_max_len",
        "load_text_encoder",
        "text_encoder_id",
        "text_dim",
        "attention_backend",
        "use_gradient_checkpointing",
        "hidden_dim",
        "freq_dim",
        "num_heads",
        "attn_head_dim",
        "dit_config",
        "video_scheduler",
        "lora_target_modules",
        "lora",
        # Read historical configs without changing their checkpoint format.
        "video_hidden_layer",
        "action_dit",
        "action_dit_pretrained_path",
        "checkpoint",
        "video_dit_checkpoint",
    }
    backend_keys = {
        "wan22": {"model_id", "num_layers"},
        "cosmos25": {
            "model_id",
            "reason_model_id",
            "num_layers",
            "video_attention_mask_mode",
        },
        "flux2": {
            "variant",
            "model_path",
            "ae_model_path",
            "qwen3_model_spec",
            "qwen3_output_layers",
            "num_layers_double",
            "num_layers_single",
        },
    }
    unknown = set(result) - common - backend_keys[name]
    if unknown:
        raise ValueError(
            f"Unknown {name} video_dit configuration keys: {sorted(unknown)}"
        )
    result["name"] = name
    return result


def _backend(name: str):
    if name == "wan22":
        from xiconwam.vision import wan as backend
    elif name == "cosmos25":
        from xiconwam.vision import cosmos as backend
    elif name == "flux2":
        from xiconwam.vision import flux as backend
    else:
        raise ValueError(f"Unknown vision tower {name!r}.")
    return backend


def load_components(config, **kwargs):
    cfg = normalize_spec(config)
    kwargs["skip_dit_load_from_pretrain"] = bool(
        kwargs.get("skip_dit_load_from_pretrain", False)
    ) or bool(cfg.get("from_scratch", False))
    return _backend(cfg["name"]).load_components(cfg, **kwargs)


def load_dit(config, **kwargs):
    cfg = normalize_spec(config)
    kwargs["skip_dit_load_from_pretrain"] = bool(
        kwargs.get("skip_dit_load_from_pretrain", False)
    ) or bool(cfg.get("from_scratch", False))
    return _backend(cfg["name"]).load_dit(cfg, **kwargs)


def load_text_encoder(config, **kwargs):
    cfg = normalize_spec(config)
    return _backend(cfg["name"]).load_text_encoder(cfg, **kwargs)


def initialize_action(config, action, visual, *, alpha=True):
    cfg = normalize_spec(config)
    return _backend(cfg["name"]).initialize_action(action, visual, alpha=alpha)
