"""Resolve a task and write the Accelerate launcher configuration."""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from xiconwam.engine.configuration import (
    resolve_training_config,
    validate_training_config,
)
from xiconwam.kit.config_resolvers import register_default_resolvers
from xiconwam.model_config import (
    action_dit,
    as_dict,
    resolve_action_config,
    validate_architecture,
    video_backbone,
)


def deepspeed_path(config, root):
    zero = str(config.get("zero", 2))
    offload = bool(config.get("zero_offload", False))
    if zero.endswith(".json") or "/" in zero:
        path = Path(zero).expanduser()
        path = path if path.is_absolute() else root / path
    else:
        filename = {
            (1, False): "ds_zero1_config.json",
            (2, False): "ds_zero2_config.json",
            (2, True): "ds_zero2_offload_config.json",
            (3, False): "ds_zero3_config.json",
            (3, True): "ds_zero3_offload_config.json",
        }.get((int(zero), offload))
        if filename is None:
            raise ValueError(
                f"Unsupported DeepSpeed zero={zero}, zero_offload={offload}"
            )
        path = root / "configs/deepspeed" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def launcher_config(config, processes, root):
    if processes < 1:
        raise ValueError("Number of processes must be positive")
    # Older saved runs may have used the temporary launcher config group.
    framework = OmegaConf.select(config, "acceleration.framework", default="deepspeed")
    if framework == "single" and processes != 1:
        raise ValueError("This run requires GPU_IDS=0 or NPROC=1")
    payload = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": "DEEPSPEED",
        "mixed_precision": None,
        "machine_rank": 0,
        "num_machines": 1,
        "num_processes": processes,
        "main_training_function": "main",
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False,
    }
    if framework in {"ddp", "single"}:
        payload["distributed_type"] = "MULTI_GPU" if processes > 1 else "NO"
        payload["mixed_precision"] = "no"
    else:
        payload["deepspeed_config"] = {
            "deepspeed_config_file": str(deepspeed_path(config, root)),
            "zero3_init_flag": False,
        }
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--launcher-file")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[3]
    register_default_resolvers()
    overrides = [f"output_dir={args.output}", *overrides]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name=f"task/{args.task}", overrides=overrides)
    cfg = resolve_training_config(cfg, overrides)
    validate_training_config(cfg)
    launch = launcher_config(cfg, args.processes, root)
    if args.dry_run:
        model = cfg.model
        video = video_backbone(model)
        action = action_dit(model)
        action_config = (
            resolve_action_config(video, action)
            if action is not None
            else as_dict(model.get("action_dit_config"))
        )
        validate_architecture(model.architecture, action_config)
        print(
            json.dumps(
                {
                    "architecture": model.architecture,
                    "video_dit": video["name"],
                    "video_assets": video.get("model_id", video.get("model_path")),
                    "data": cfg.data.train.get("_target_"),
                    "action_initialization": action.get("pretrained_path")
                    if action is not None
                    else None,
                    "action_checkpoint": action.get("checkpoint")
                    if action is not None
                    else None,
                    "action_config": action_config,
                    "training": as_dict(model.get("training")),
                    "resume": cfg.get("resume"),
                    "launcher": launch,
                },
                indent=2,
                default=str,
            )
        )
        return
    if not args.launcher_file:
        parser.error("--launcher-file is required")
    OmegaConf.save(OmegaConf.create(launch), args.launcher_file)


if __name__ == "__main__":
    main()
