"""Build an ActionExpert initialization file from a video tower.

The output is a resized copy of the video DiT, not a trained checkpoint.
mot, idm, and hide load it through action_dit_pretrained_path.
"""

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from xiconwam.actor.action_expert import ActionExpert
from xiconwam.kit.config_resolvers import register_default_resolvers
from xiconwam.vision.registry import initialize_action, load_dit

register_default_resolvers()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compose_task(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(project_root() / "configs"), version_base="1.3"
    ):
        return compose(
            config_name=config_name,
            overrides=["output_dir=./runs/prepare", *overrides],
        )


def action_expert_spec(cfg):
    from xiconwam.model_config import (
        action_dit,
        as_dict,
        resolve_action_config,
        video_backbone,
    )

    video = as_dict(video_backbone(cfg.model))
    action = action_dit(cfg.model)
    if action is not None:
        return video, resolve_action_config(video, action)
    if cfg.model.get("action_dit_config") is None:
        raise ValueError("The selected model has no Action DiT")
    return video, as_dict(cfg.model.action_dit_config)


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def build_initialization(cfg, *, device: str, dtype: torch.dtype, alpha: bool):
    backbone, action_cfg = action_expert_spec(cfg)
    visual = load_dit(backbone, device=device, torch_dtype=dtype)
    action = ActionExpert(**action_cfg).to(device=device, dtype=dtype)
    return initialize_action(backbone, action, visual, alpha=alpha)


def configured_output(cfg) -> Path:
    from xiconwam.model_config import action_dit, video_backbone

    action = action_dit(cfg.model)
    raw = (
        action.get("pretrained_path")
        if action is not None
        else video_backbone(cfg.model).get("action_dit_pretrained_path")
    )
    text = "" if raw is None else str(raw).strip()
    if text.lower() in {"", "none", "null"}:
        raise SystemExit(
            "model.action_dit.pretrained_path is empty. Pass --output."
        )
    path = Path(text)
    if not path.is_absolute():
        path = project_root() / path
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Resize a video DiT into an ActionExpert initialization file."
    )
    parser.add_argument("--config-name", default="task/robodojo_wan")
    parser.add_argument(
        "--output", help="Override model.action_dit.pretrained_path"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument("--alpha-scaling", default="true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    cfg = compose_task(args.config_name, list(args.overrides))
    output = Path(args.output).expanduser() if args.output else configured_output(cfg)
    backbone_cfg, _ = action_expert_spec(cfg)
    weight = backbone_cfg.get("model_path") or backbone_cfg.get("model_id")
    print(
        f"backbone={backbone_cfg.get('name')} weights={weight} output={output}",
        flush=True,
    )
    payload = build_initialization(
        cfg,
        device=args.device,
        dtype=_dtype(args.dtype),
        alpha=_bool(args.alpha_scaling),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    counts = payload.get("initialization", {})
    print(
        f"Saved ActionExpert initialization to {output}: "
        f"backbone={payload['policy']['source_backbone']} counts={counts}"
    )


if __name__ == "__main__":
    main()
