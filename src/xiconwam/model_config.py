"""Resolve the video backbone and its paired Action DiT."""

from omegaconf import DictConfig, OmegaConf


def as_dict(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value or {})


def video_backbone(model_config):
    """Read current model configs and saved configs from earlier runs."""
    keys = [
        key
        for key in ("video_dit", "video_backbone", "backbone")
        if model_config.get(key) is not None
    ]
    if len(keys) != 1:
        raise ValueError(
            "Specify exactly one model.video_dit (legacy: backbone/video_backbone)"
        )
    return model_config[keys[0]]


def action_dit(model_config):
    if model_config.get("architecture") == "video":
        return None
    current = model_config.get("action_dit")
    if current is not None:
        return current
    legacy = model_config.get("action_backbone")
    if legacy is not None:
        return legacy
    return (video_backbone(model_config) or {}).get("action_dit")


def resolve_action_config(video, action):
    video, action = as_dict(video), as_dict(action)
    if action.get("name") != "actiondit":
        raise ValueError("action_dit.name must be actiondit")
    config = dict(action.get("dit_config", {}))
    preset = config.get("preset", action.get("preset", "auto"))
    expected_preset = "staged" if video["name"] == "flux2" else "standard"
    if preset == "auto":
        preset = expected_preset
    if preset != expected_preset:
        raise ValueError(
            f"{video['name']} requires Action DiT preset={expected_preset}"
        )
    config.update(action.get(preset, {}))
    config["preset"] = preset
    automatic = {
        "num_heads": video["num_heads"],
        "attn_head_dim": video["attn_head_dim"],
        "text_dim": video["text_dim"],
        "attention_backend": video.get("attention_backend", "auto"),
    }
    if preset == "staged":
        automatic.update(
            num_layers_double=video["num_layers_double"],
            num_layers_single=video["num_layers_single"],
        )
    else:
        automatic.update(num_layers=video["num_layers"], freq_dim=video["freq_dim"])
    for key, expected in automatic.items():
        if config.get(key, "auto") == "auto":
            config[key] = expected
        elif (
            key
            in {
                "num_heads",
                "attn_head_dim",
                "num_layers",
                "num_layers_double",
                "num_layers_single",
                "text_dim",
            }
            and config[key] != expected
        ):
            raise ValueError(
                f"Action DiT {key}={config[key]} must match {video['name']} {expected}"
            )
    return config


def validate_architecture(architecture, action_config):
    if architecture == "hide" and action_config.get("preset") == "staged":
        raise ValueError(
            "HIDE needs an Action DiT with cross-attention. Select Wan or Cosmos."
        )


def resolve_feature_layer(video, value):
    if value == "auto":
        value = {"wan22": 17, "cosmos25": 16}.get(video["name"])
    if value is None or not 0 <= int(value) < int(video["num_layers"]):
        raise ValueError("HIDE feature_layer must index a video transformer block")
    return int(value)


def resolve_model_paths(model_config, root):
    from pathlib import Path

    video = video_backbone(model_config)
    keys = (
        "model_id",
        "model_path",
        "ae_model_path",
        "reason_model_id",
        "action_dit_pretrained_path",
        "qwen3_model_spec",
    )
    for key in keys:
        value = video.get(key)
        if not value:
            continue
        if key == "qwen3_model_spec" and not str(value).startswith((".", "/", "~")):
            continue
        path = Path(str(value)).expanduser()
        video[key] = str(path if path.is_absolute() else (root / path).resolve())
    action = action_dit(model_config)
    if action is None:
        action = model_config
        keys = ("action_dit_checkpoint", "action_dit_pretrained_path")
    else:
        keys = ("checkpoint", "pretrained_path")
    for key in keys:
        value = action.get(key)
        if value:
            path = Path(str(value)).expanduser()
            action[key] = str(path if path.is_absolute() else (root / path).resolve())
