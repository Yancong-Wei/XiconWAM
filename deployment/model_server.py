#!/usr/bin/env python3
"""XiconWAM ZMQ policy server for robot_client (protocol v4).

Default mode is serial open-loop: no RTC leftover, no soft guidance. Pair with
``deployment/robot_client.py`` or ``deployment/fake_lerobot_client.py`` using
``--execute-steps`` equal to ``--action-horizon`` for serial chunk execution.

Examples:
  python deployment/model_server.py \\
    --run-dir runs/robodojo/<run_id> \\
    --bind tcp://*:5556 --warmup

  python deployment/model_server.py \\
    --config runs/.../config.yaml \\
    --checkpoint runs/.../checkpoints/weights/step_003000.pt \\
    --dataset-stats playground/datasets/robodojo/dataset_stats.json \\
    --dry-zeros
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SERVER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SERVER_ROOT / "src"
for path in (SERVER_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deployment.protocol import (
    PROTOCOL_VERSION,
    make_action_response,
    make_error_response,
    policy_connect_targets,
    should_clear_rtc,
    validate_zmq_tcp_endpoint,
)

logger = logging.getLogger(__name__)

# RoboDojo / EasyWAM-style L-canvas used by configs/data/robodojo.yaml.
DEFAULT_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_high": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
ROBOTWIN_SIZES = ((256, 320), (128, 160), (128, 160))  # (H, W) per camera
CANVAS_HW = (384, 320)
DEFAULT_UNIFY_MAP = ["0-13"]


def _resolve_run_paths(
    *,
    run_dir: str | None,
    config: str | None,
    checkpoint: str | None,
    dataset_stats: str | None,
) -> tuple[Path, Path, Path | None]:
    if run_dir:
        root = Path(run_dir).expanduser().resolve()
        cfg_path = root / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Run config not found: {cfg_path}")
        weights_dir = root / "checkpoints" / "weights"
        if checkpoint:
            ckpt_path = Path(checkpoint).expanduser().resolve()
        else:
            candidates = sorted(weights_dir.glob("step_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No step_*.pt under {weights_dir}")
            ckpt_path = candidates[-1]
        stats_path = (
            Path(dataset_stats).expanduser().resolve()
            if dataset_stats
            else _guess_stats_from_config(cfg_path)
        )
        return cfg_path, ckpt_path, stats_path

    if not config or not checkpoint:
        raise ValueError("Provide --run-dir, or both --config and --checkpoint")
    cfg_path = Path(config).expanduser().resolve()
    ckpt_path = Path(checkpoint).expanduser().resolve()
    stats_path = (
        Path(dataset_stats).expanduser().resolve()
        if dataset_stats
        else _guess_stats_from_config(cfg_path)
    )
    return cfg_path, ckpt_path, stats_path


def _guess_stats_from_config(cfg_path: Path) -> Path | None:
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        stats = OmegaConf.select(cfg, "data.train.pretrained_norm_stats")
        if stats:
            path = Path(str(stats)).expanduser()
            if not path.is_absolute():
                path = (SERVER_ROOT / path).resolve()
            return path if path.is_file() else None
    except Exception:
        return None
    return None


def _resize_chw(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize HWC uint8 RGB to target (H, W)."""
    from PIL import Image

    h, w = size_hw
    if image.shape[0] == 3 and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.shape[:2] == (h, w):
        return arr
    return np.asarray(
        Image.fromarray(arr, mode="RGB").resize((w, h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def compose_robotwin_canvas(
    images: dict[str, np.ndarray],
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
) -> np.ndarray:
    """Compose three views into a 384x320 RGB canvas (train-time robotwin layout)."""
    if len(camera_keys) != 3:
        raise ValueError(f"robotwin layout needs 3 cameras, got {camera_keys}")
    tiles = []
    for key, size in zip(camera_keys, ROBOTWIN_SIZES):
        if key not in images:
            raise KeyError(f"Missing camera image for {key!r}")
        tiles.append(_resize_chw(images[key], size))
    canvas = np.zeros((CANVAS_HW[0], CANVAS_HW[1], 3), dtype=np.uint8)
    canvas[:256] = tiles[0]
    canvas[256:, :160] = tiles[1]
    canvas[256:, 160:] = tiles[2]
    return canvas


def extract_images(request: dict[str, Any]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    nested = request.get("observations")
    if isinstance(nested, dict) and isinstance(nested.get("images"), dict):
        for key, value in nested["images"].items():
            images[str(key)] = np.asarray(value)
    raw = request.get("images")
    if isinstance(raw, dict):
        for key, value in raw.items():
            images[str(key)] = np.asarray(value)
    for key, value in request.items():
        if not isinstance(key, str) or not key.startswith("observation.images."):
            continue
        cam = key.removeprefix("observation.images.")
        images[cam] = np.asarray(value)
    normalized: dict[str, np.ndarray] = {}
    for key, value in images.items():
        canon = CAMERA_ALIASES.get(key, key)
        normalized[canon] = value
    return normalized


def extract_state(request: dict[str, Any]) -> np.ndarray:
    nested = request.get("observations")
    if isinstance(nested, dict):
        state = nested.get("state", nested.get("proprio"))
        if state is not None:
            return np.asarray(state, dtype=np.float32).reshape(-1)
    for key in ("observation.state", "proprio", "state"):
        if key in request and request[key] is not None:
            return np.asarray(request[key], dtype=np.float32).reshape(-1)
    raise KeyError("Request is missing joint state (observation.state / proprio)")


def extract_instruction(request: dict[str, Any], default: str | None) -> str:
    for key in ("language_instruction", "instruction", "prompt", "task"):
        value = request.get(key)
        if value:
            return str(value)
    if default:
        return str(default)
    return "Complete the manipulation task."


@dataclass
class NormBundle:
    """Train-time normalize / denormalize for unified action and state."""

    shape_meta: dict[str, Any]
    unify_action_map: list[str]
    unify_dim: int
    normalizer: Any
    dst_index: np.ndarray

    @classmethod
    def from_stats(
        cls,
        stats_path: Path,
        *,
        unify_action_map: list[str] | None = None,
        unify_dim: int = 48,
        raw_dim: int = 14,
    ) -> "NormBundle":
        from xiconwam.data.lerobot.utils.normalizer import (
            LinearNormalizer,
            load_dataset_stats_from_json,
        )
        from xiconwam.data.unify_action import parse_unify_spec

        shape_meta = {
            "action": [{"key": "default", "raw_shape": raw_dim, "shape": raw_dim}],
            "state": [{"key": "default", "raw_shape": raw_dim, "shape": raw_dim}],
        }
        stats = load_dataset_stats_from_json(str(stats_path))
        normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=False,
            default_mode="z-score",
            exception_mode=None,
            stats=stats,
        )
        unify_map = list(unify_action_map or DEFAULT_UNIFY_MAP)
        dst_index = parse_unify_spec(unify_map, unify_dim=unify_dim)
        return cls(
            shape_meta=shape_meta,
            unify_action_map=unify_map,
            unify_dim=int(unify_dim),
            normalizer=normalizer,
            dst_index=dst_index,
        )

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Match WindowPacker: normalize raw joints, then scatter into unify slots."""
        import torch

        from xiconwam.data.unify_action import map_to_unify

        raw = np.asarray(state, dtype=np.float32).reshape(-1)
        if raw.shape[0] != len(self.dst_index):
            raise ValueError(
                f"State width {raw.shape[0]} != unify map size {len(self.dst_index)}"
            )
        batch = {
            "state": {"default": torch.as_tensor(raw[None, :], dtype=torch.float32)}
        }
        batch = self.normalizer.forward(batch)
        normalized = batch["state"]["default"][0].cpu().numpy().astype(np.float32)
        unified, _ = map_to_unify(
            normalized, self.dst_index, unify_dim=self.unify_dim
        )
        return np.asarray(unified, dtype=np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Match training inverse: unmap unified slots, then denormalize raw joints."""
        import torch

        from xiconwam.data.unify_action import unmap_from_unify

        values = np.asarray(action, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        # Training: raw -> normalize(raw) -> map_to_unify.
        # Inverse: unmap -> denormalize(raw).
        raw = unmap_from_unify(values, self.dst_index)
        batch = {
            "action": {"default": torch.as_tensor(raw, dtype=torch.float32)},
            "state": {
                "default": torch.zeros(
                    (raw.shape[0], raw.shape[1]), dtype=torch.float32
                )
            },
        }
        batch = self.normalizer.backward(batch)
        return batch["action"]["default"].cpu().numpy().astype(np.float32)


@dataclass
class XiconServeRuntime:
    """One ZMQ bind: parse protocol obs, run XiconWAM, return joint action chunk."""

    config_path: Path
    checkpoint_path: Path
    dataset_stats_path: Path | None
    default_instruction: str | None = None
    action_horizon: int = 32
    action_dim: int = 14
    model_action_dim: int = 48
    action_fps: float = 25.0
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS
    num_inference_steps: int = 10
    num_video_frames: int = 9
    dry_zeros: bool = False
    device: str = "cuda"
    _model: Any = field(default=None, init=False, repr=False)
    _norm: NormBundle | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dry_zeros:
            logger.warning("dry-zeros mode: returning zero actions (protocol smoke only)")
            if self.dataset_stats_path and self.dataset_stats_path.is_file():
                self._norm = NormBundle.from_stats(
                    self.dataset_stats_path,
                    unify_dim=self.model_action_dim,
                    raw_dim=self.action_dim,
                )
            return
        self._load_model()
        if self.dataset_stats_path is None or not self.dataset_stats_path.is_file():
            raise FileNotFoundError(
                "dataset_stats.json is required for real inference "
                "(pass --dataset-stats or set data.train.pretrained_norm_stats)"
            )
        self._norm = NormBundle.from_stats(
            self.dataset_stats_path,
            unify_dim=self.model_action_dim,
            raw_dim=self.action_dim,
        )

    def _load_model(self) -> None:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from xiconwam.actor.helpers.inference import configure_model_execution

        cfg = OmegaConf.load(self.config_path)
        mixed = str(OmegaConf.select(cfg, "mixed_precision") or "bf16").lower()
        dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }.get(mixed, torch.float32)
        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU")
            device = "cpu"
        model = instantiate(cfg.model, model_dtype=dtype, device=device)
        model = configure_model_execution(
            model,
            vae_micro_batch_size=OmegaConf.select(cfg, "vae_micro_batch_size") or 1,
            inference_cross_kv_reuse=bool(
                OmegaConf.select(cfg, "inference_cross_kv_reuse", default=True)
            ),
        )
        logger.info("Loading checkpoint %s", self.checkpoint_path)
        model.load_checkpoint(str(self.checkpoint_path))
        model.eval()
        self._model = model
        if hasattr(model, "action_dim"):
            self.model_action_dim = int(model.action_dim)
        logger.info(
            "XiconWAM ready | architecture=%s action_horizon=%d raw_dim=%d model_dim=%d",
            type(model).__name__,
            self.action_horizon,
            self.action_dim,
            self.model_action_dim,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "server": "xiconwam",
            "backend": "dry_zeros" if self.dry_zeros else "xiconwam",
            "action_horizon": int(self.action_horizon),
            "action_dim": int(self.action_dim),
            "action_fps": float(self.action_fps),
            "pi05_rtc": False,
            "rtc_soft_guidance": False,
            "open_loop": True,
            "camera_keys": list(self.camera_keys),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
        }

    def warmup(self) -> None:
        if self.dry_zeros:
            return
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "warmup",
            "chunk_index": 0,
            "execute_steps": 0,
            "action_horizon": self.action_horizon,
            "language_instruction": self.default_instruction
            or "Complete the manipulation task.",
            "observation.state": np.zeros(self.action_dim, dtype=np.float32),
            "proprio": np.zeros(self.action_dim, dtype=np.float32),
        }
        for key, size in zip(self.camera_keys, ROBOTWIN_SIZES):
            request[f"observation.images.{key}"] = np.zeros(
                (size[0], size[1], 3), dtype=np.uint8
            )
        t0 = time.monotonic()
        self.infer(request)
        logger.info("Warmup done in %.2fs", time.monotonic() - t0)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        forbidden = ("inference_delay", "prev_action_chunk", "prev_chunk_left_over")
        present = [key for key in forbidden if key in request]
        if present:
            raise ValueError(
                "Client-side RTC fields are not part of protocol v4 open-loop mode: "
                f"{present}"
            )
        if "chunk_index" not in request or "execute_steps" not in request:
            raise ValueError("Protocol v4 requires chunk_index and execute_steps")
        chunk_index = int(request["chunk_index"])
        execute_steps = int(request["execute_steps"])
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        horizon = int(request.get("action_horizon") or request.get("steps") or self.action_horizon)
        if horizon != self.action_horizon:
            raise ValueError(
                f"Request action_horizon={horizon} does not match server {self.action_horizon}"
            )
        if not 0 <= execute_steps <= horizon:
            raise ValueError(f"execute_steps must be in [0,{horizon}], got {execute_steps}")
        if chunk_index == 0 and execute_steps != 0:
            raise ValueError("initial chunk_index=0 requires execute_steps=0")
        # Serial open-loop: no leftover state. should_clear_rtc is accepted but unused.
        _ = should_clear_rtc(request)

        t0 = time.monotonic()
        if self.dry_zeros:
            action = np.zeros((horizon, self.action_dim), dtype=np.float32)
        else:
            action = self._infer_xicon(request, horizon=horizon)
        infer_ms = (time.monotonic() - t0) * 1000.0
        if action.shape != (horizon, self.action_dim):
            raise ValueError(
                f"Infer returned shape {action.shape}, expected {(horizon, self.action_dim)}"
            )
        if not np.isfinite(action).all():
            raise ValueError("Infer returned non-finite actions")
        return {
            "action": action,
            "policy_timing": {"xiconwam_infer_ms": infer_ms},
            "rtc": {
                "enabled": False,
                "chunk_index": chunk_index,
                "cleared": True,
                "soft_guidance": False,
                "backend": "xiconwam",
            },
        }

    def _infer_xicon(self, request: dict[str, Any], *, horizon: int) -> np.ndarray:
        import torch

        assert self._model is not None and self._norm is not None
        images = extract_images(request)
        missing = [key for key in self.camera_keys if key not in images]
        if missing:
            raise KeyError(f"Missing camera images: {missing}")
        canvas = compose_robotwin_canvas(images, self.camera_keys)
        pixels = torch.from_numpy(canvas).to(
            device=self._model.device, dtype=self._model.torch_dtype
        )
        pixels = pixels * (2.0 / 255.0) - 1.0
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).contiguous()

        state = extract_state(request)
        if state.shape[0] != self.action_dim:
            raise ValueError(
                f"Joint state width {state.shape[0]} != action_dim {self.action_dim}"
            )
        proprio = torch.as_tensor(
            self._norm.normalize_state(state),
            device=self._model.device,
            dtype=self._model.torch_dtype,
        )
        prompt = extract_instruction(request, self.default_instruction)

        with torch.inference_mode():
            result = self._model.infer_action(
                prompt=prompt,
                input_image=pixels,
                action_horizon=horizon,
                proprio=proprio,
                num_inference_steps=int(self.num_inference_steps),
                num_video_frames=int(self.num_video_frames),
                text_cfg_scale=1.0,
                seed=0,
            )
        action = result["action"]
        if hasattr(action, "detach"):
            action = action.detach().float().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 3:
            action = action[0]
        if action.shape[-1] == self.model_action_dim:
            action = self._norm.denormalize_action(action)
        elif action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Unexpected action width {action.shape[-1]}; "
                f"expected model_dim={self.model_action_dim} or raw_dim={self.action_dim}"
            )
        return np.asarray(action, dtype=np.float32)


def build_runtime(args: argparse.Namespace) -> XiconServeRuntime:
    camera_keys = tuple(
        key.strip() for key in str(args.camera_keys).split(",") if key.strip()
    )
    if len(camera_keys) != 3:
        raise ValueError("--camera-keys must list exactly three cameras for robotwin layout")

    if args.dry_zeros and not args.run_dir and not args.config:
        cfg_path = Path("dry_zeros_config")
        ckpt_path = Path("dry_zeros_checkpoint")
        stats_path = (
            Path(args.dataset_stats).expanduser().resolve()
            if args.dataset_stats
            else None
        )
    else:
        cfg_path, ckpt_path, stats_path = _resolve_run_paths(
            run_dir=args.run_dir,
            config=args.config,
            checkpoint=args.checkpoint,
            dataset_stats=args.dataset_stats,
        )
        if args.dry_zeros:
            ckpt_path = Path("dry_zeros_checkpoint")

    runtime = XiconServeRuntime(
        config_path=cfg_path,
        checkpoint_path=ckpt_path,
        dataset_stats_path=stats_path,
        default_instruction=args.default_instruction,
        action_horizon=int(args.action_horizon),
        action_dim=int(args.action_dim),
        model_action_dim=int(args.model_action_dim),
        action_fps=float(args.action_fps),
        camera_keys=camera_keys,
        num_inference_steps=int(args.num_inference_steps),
        num_video_frames=int(args.num_video_frames),
        dry_zeros=bool(args.dry_zeros),
        device=str(args.device),
    )
    if args.warmup:
        runtime.warmup()
    return runtime


def _make_response(
    request: dict[str, Any],
    backend_result: dict[str, Any],
    *,
    runtime: XiconServeRuntime,
    receive_time: float,
) -> dict[str, Any]:
    action = np.asarray(backend_result["action"], dtype=np.float32)
    timing = {"server_total_ms": (time.monotonic() - receive_time) * 1000.0}
    timing.update(backend_result.get("policy_timing") or {})
    meta = dict(runtime.metadata)
    response = make_action_response(
        request,
        action,
        action_key="action",
        metadata={
            **meta,
            "model_metadata": meta,
            "robot_execution_fps": request.get("execution_fps"),
            "rtc": backend_result.get("rtc"),
        },
    )
    response["timing"] = timing
    return response


def serve(args: argparse.Namespace) -> None:
    import zmq

    args.bind = validate_zmq_tcp_endpoint(args.bind, bind=True)
    runtime = build_runtime(args)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    port = int(args.bind.rsplit(":", 1)[1])
    local_target, lan_target = policy_connect_targets(port)
    logger.info(
        "XiconWAM policy server listening | bind=%s open_loop=True dry_zeros=%s",
        args.bind,
        args.dry_zeros,
    )
    logger.info("Policy target (same machine): %s", local_target)
    if lan_target:
        logger.info("Policy target (robot machine): %s", lan_target)
    try:
        while True:
            request = socket.recv_pyobj()
            receive_time = time.monotonic()
            try:
                if not isinstance(request, dict):
                    raise TypeError(f"Request must be a dict, got {type(request)!r}")
                version = request.get("protocol_version")
                if version is not None and version != PROTOCOL_VERSION:
                    raise ValueError(
                        f"Unsupported protocol_version {version!r}; expected {PROTOCOL_VERSION!r}"
                    )
                backend_result = runtime.infer(request)
                response = _make_response(
                    request,
                    backend_result,
                    runtime=runtime,
                    receive_time=receive_time,
                )
                logger.info(
                    "request_id=%s action_shape=%s ok",
                    request.get("request_id"),
                    response["action"].shape,
                )
            except Exception as exc:
                logger.exception("request failed")
                response = make_error_response(request, str(exc))
                response["timing"] = {
                    "server_total_ms": (time.monotonic() - receive_time) * 1000.0
                }
            socket.send_pyobj(response)
    except KeyboardInterrupt:
        logger.info("XiconWAM policy server stopped")
    finally:
        socket.close(linger=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XiconWAM ZMQ policy server (serial open-loop default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None, help="Training run directory with config.yaml")
    parser.add_argument("--config", default=None, help="Resolved Hydra config.yaml")
    parser.add_argument("--checkpoint", default=None, help="weights/step_XXXXXX.pt")
    parser.add_argument("--dataset-stats", default=None, help="dataset_stats.json path")
    parser.add_argument("--bind", default="tcp://*:5556")
    parser.add_argument("--default-instruction", default=None)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=14, help="Raw joint width for robot_client")
    parser.add_argument(
        "--model-action-dim",
        type=int,
        default=48,
        help="Unified model action width (UNIFY_DIM)",
    )
    parser.add_argument(
        "--action-fps",
        type=float,
        default=25.0,
        help="Must match robot_client --execute-fps exactly",
    )
    parser.add_argument(
        "--camera-keys",
        default="cam_high,cam_left_wrist,cam_right_wrist",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--num-video-frames",
        type=int,
        default=9,
        help="Sparse video frames implied by num_frames=33, ratio=4",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-zeros",
        action="store_true",
        help="Skip model load; return zero actions for protocol smoke tests",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    serve(parse_args())


if __name__ == "__main__":
    main()
