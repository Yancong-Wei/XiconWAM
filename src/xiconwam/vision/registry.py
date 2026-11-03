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
    name = str(result["name"]).strip().lower()
    if name not in TOWERS:
        raise ValueError(f"Xicon vision tower must be one of {TOWERS}, got {name!r}.")
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
    return _backend(cfg["name"]).load_components(cfg, **kwargs)


def load_dit(config, **kwargs):
    cfg = normalize_spec(config)
    return _backend(cfg["name"]).load_dit(cfg, **kwargs)


def load_text_encoder(config, **kwargs):
    cfg = normalize_spec(config)
    return _backend(cfg["name"]).load_text_encoder(cfg, **kwargs)


def initialize_action(config, action, visual, *, alpha=True):
    cfg = normalize_spec(config)
    return _backend(cfg["name"]).initialize_action(action, visual, alpha=alpha)
