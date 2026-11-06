"""XiconWAM joint-policy adapter for RoboDojo websocket evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

CORE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CORE_ROOT.parents[1]
EVAL_ROOT = CORE_ROOT.parent
for path in (str(REPO_ROOT), str(EVAL_ROOT), str(CORE_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from deployment.model_server import XiconServeRuntime
from eval.robodojo_io import action_vector_to_xpolicylab, pack_joint_state


def _resolve_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _camera_color(observation: dict[str, Any], name: str) -> np.ndarray:
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        raise KeyError("observation is missing the 'vision' mapping")
    camera = vision.get(name)
    if not isinstance(camera, dict) or "color" not in camera:
        raise KeyError(f"observation is missing vision.{name}.color")
    return np.asarray(camera["color"], dtype=np.uint8)


class Policy:
    """Websocket ModelTemplate-compatible policy for absolute joint control."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        dry_zeros = bool(self.model_cfg.get("dry_zeros", False))
        action_type = str(self.model_cfg.get("action_type") or "joint")
        if action_type != "joint":
            raise ValueError(
                f"XiconWAM RoboDojo eval only supports action_type=joint, got {action_type!r}"
            )

        run_dir = _resolve_path(
            self.model_cfg.get("run_dir"),
            base=REPO_ROOT,
        )
        if run_dir is None:
            if not dry_zeros:
                raise ValueError("deploy.yml must set run_dir (or pass XICON_RUN_DIR)")
            run_dir = REPO_ROOT  # Unused by the fixture; no run or weights needed.

        checkpoint = _resolve_path(
            self.model_cfg.get("checkpoint"),
            base=REPO_ROOT,
        )
        ckpt_name = self.model_cfg.get("ckpt_name")
        if checkpoint is None and ckpt_name and not dry_zeros:
            name = str(ckpt_name).strip()
            if name:
                weights_dir = run_dir / "checkpoints" / "weights"
                candidate = weights_dir / (
                    name if name.endswith(".pt") else f"{name}.pt"
                )
                if candidate.is_file():
                    checkpoint = candidate
                else:
                    raise FileNotFoundError(
                        f"ckpt_name={name!r} not found under {weights_dir}"
                    )
        if checkpoint is None and not dry_zeros:
            weights = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
            if not weights:
                raise FileNotFoundError(
                    f"No step_*.pt under {run_dir / 'checkpoints' / 'weights'}"
                )
            checkpoint = weights[-1]
        if checkpoint is None:
            checkpoint = run_dir / "dry_zeros_checkpoint"

        dataset_stats = _resolve_path(
            self.model_cfg.get("dataset_stats"),
            base=REPO_ROOT,
        )
        if dataset_stats is None:
            candidate = run_dir / "dataset_stats.json"
            if candidate.is_file():
                dataset_stats = candidate

        self.replan_steps = int(self.model_cfg.get("replan_steps") or 10)
        if self.replan_steps <= 0:
            raise ValueError(f"replan_steps must be positive, got {self.replan_steps}")

        action_horizon = int(self.model_cfg.get("action_horizon") or 32)
        action_dim = int(self.model_cfg.get("action_dim") or 14)
        model_action_dim = int(self.model_cfg.get("model_action_dim") or 48)
        num_inference_steps = int(self.model_cfg.get("num_inference_steps") or 10)
        num_video_frames = int(self.model_cfg.get("num_video_frames") or 9)
        device = str(self.model_cfg.get("device") or "cuda")
        default_instruction = self.model_cfg.get("default_instruction")

        self.runtime = XiconServeRuntime(
            config_path=run_dir / "config.yaml",
            checkpoint_path=checkpoint,
            dataset_stats_path=dataset_stats,
            default_instruction=(
                None if default_instruction is None else str(default_instruction)
            ),
            action_horizon=action_horizon,
            action_dim=action_dim,
            model_action_dim=model_action_dim,
            num_inference_steps=num_inference_steps,
            num_video_frames=num_video_frames,
            dry_zeros=dry_zeros,
            device=device,
        )
        self._obs: dict[str, Any] | None = None

    def update_obs(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            raise TypeError(f"obs must be a dict, got {type(obs)!r}")
        self._obs = obs

    def update_obs_batch(self, obs_list):
        raise NotImplementedError(
            "XiconWAM RoboDojo eval uses eval_batch=false; call update_obs"
        )

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._obs is None:
            raise RuntimeError("get_action called before update_obs")

        horizon = int(self.runtime.action_horizon)
        if self.runtime.dry_zeros:
            chunk = np.zeros((horizon, int(self.runtime.action_dim)), dtype=np.float32)
        else:
            instruction = self._obs.get(
                "instruction", self._obs.get("task_instruction")
            )
            request = {
                "language_instruction": instruction,
                "observation.state": pack_joint_state(self._obs),
                "observation.images.cam_high": _camera_color(self._obs, "cam_head"),
                "observation.images.cam_left_wrist": _camera_color(
                    self._obs, "cam_left_wrist"
                ),
                "observation.images.cam_right_wrist": _camera_color(
                    self._obs, "cam_right_wrist"
                ),
            }
            chunk = self.runtime._infer_xicon(request, horizon=horizon)
        if chunk.ndim != 2 or chunk.shape[-1] != int(self.runtime.action_dim):
            raise RuntimeError(
                f"Unexpected action chunk shape {chunk.shape}; "
                f"expected (?, {self.runtime.action_dim})"
            )
        n_exec = min(self.replan_steps, int(chunk.shape[0]))
        return [action_vector_to_xpolicylab(chunk[i]) for i in range(n_exec)]

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError(
            "XiconWAM RoboDojo eval uses eval_batch=false; call get_action"
        )

    def reset(self) -> None:
        self._obs = None
