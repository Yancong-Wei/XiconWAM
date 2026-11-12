#!/usr/bin/env python3
"""Compute XiconWAM ``dataset_stats.json`` from a task YAML.

Uses the same episode aggregation as training
(``BaseLerobotDataset.get_dataset_stats``). Do not rename LeRobot
``meta/stats.json`` into this file: the normalizer expects
``action/state -> default -> global_*`` fields.

Example:
  python scripts/compute_dataset_stats.py
  python scripts/compute_dataset_stats.py --config-name task/robodojo_wan
  python scripts/compute_dataset_stats.py --output playground/datasets/robodojo/dataset_stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="task/robodojo_wan",
        help="Hydra config name under configs/",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path; default is data.train.pretrained_norm_stats",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides, e.g. data.train.dataset_dirs=[./playground/datasets/robodojo]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from xiconwam.data.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from xiconwam.data.discovery import discover_dataset_dirs
    from xiconwam.data.lerobot.robot_video_dataset import (
        configured_window_steps,
        trained_window,
    )
    from xiconwam.data.lerobot.utils.normalizer import save_dataset_stats_to_json
    from xiconwam.kit.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))

    train_cfg = cfg.data.train
    dataset_dirs = discover_dataset_dirs(train_cfg.dataset_dirs)
    for dataset_dir in dataset_dirs:
        root = Path(dataset_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {root.resolve()}")

    output = Path(
        args.output
        or train_cfg.get("pretrained_norm_stats")
        or (Path(dataset_dirs[0]) / "dataset_stats.json")
    )
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    num_frames = configured_window_steps(train_cfg)
    ratio = int(train_cfg.action_video_freq_ratio)
    _, action_steps = trained_window(num_frames, ratio)
    shape_meta = OmegaConf.to_container(train_cfg.shape_meta, resolve=True)

    processor = instantiate(train_cfg.processor)
    dataset = BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        obs_size=num_frames,
        action_size=action_steps,
        val_set_proportion=float(train_cfg.get("val_set_proportion", 0.0)),
        is_training_set=True,
        global_sample_stride=int(train_cfg.get("global_sample_stride", 1)),
        image_obs_indices=tuple(range(0, num_frames, ratio)),
    )

    print(
        f"Computing stats | datasets={dataset_dirs} episodes={dataset.multi_dataset.num_episodes} "
        f"frames={dataset.multi_dataset.num_frames} output={output}"
    )
    stats = dataset.get_dataset_stats(processor)
    save_dataset_stats_to_json(stats, str(output))
    action_mean = stats["action"]["default"]["global_mean"]
    print(
        f"Wrote {output} | action_global_mean_dim={tuple(action_mean.shape)} "
        f"num_episodes={stats['num_episodes']}"
    )


if __name__ == "__main__":
    main()
