#!/usr/bin/env python3
"""Robot-side policy client for XiconWAM / robot_control deployments.

Pairs with ``deployment/model_server.py`` through ``deployment/protocol.py``.
Offline open-loop checks without hardware use ``deployment/fake_lerobot_client.py``.

Observations use joint state as proprio; the policy returns ``action`` with
``action_horizon`` rows. The client always owns one background request worker:
every mode starts the next request after K=`execute_steps` actions, continues
the old tail while it waits, then switches at an action boundary. Every request
carries ``chunk_index`` and ``execute_steps`` as generic scheduling context.
K=H is serial open-loop (XiconWAM default). ``chunk_index==0`` marks episode
start. Ctrl-C returns home.

Requires the ``robot_control`` package and ``configs/robots`` (set
``ROBOT_CONTROL_ROOT`` if needed).

Example (serial open-loop against XiconWAM):
  python deployment/robot_client.py \\
    --policy-url tcp://192.168.31.121:5556 \\
    --task "pick up the cube and place it on the platform" \\
    --action-horizon 32 --execute-steps 32 --execute-fps 25 \\
    --chunk-num 0 --dry-run
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np


def _resolve_robot_control_root() -> Path:
    """Locate the robot_control repo used for RobotAdapter + robot YAMLs.

    Bound files in this XiconWAM checkout:
      - deployment/model_server.py
      - deployment/robot_client.py  (this file)
      - deployment/protocol.py
      - deployment/fake_lerobot_client.py
    """
    deploy_dir = Path(__file__).resolve().parent
    env_root = os.environ.get("ROBOT_CONTROL_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append(deploy_dir.parent)
    if len(deploy_dir.parents) >= 3:
        candidates.append(deploy_dir.parents[2])
    for root in candidates:
        if (root / "robot_control").is_dir() and (root / "configs" / "robots").is_dir():
            return root
    raise RuntimeError(
        "Cannot locate robot_control repo root (need robot_control/ package and "
        "configs/robots/). Set ROBOT_CONTROL_ROOT or use fake_lerobot_client.py "
        "for offline open-loop checks without hardware."
    )


PROJECT_ROOT = _resolve_robot_control_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Prefer this checkout's deployment/ over any other path entry.
_DEPLOY_DIR = Path(__file__).resolve().parent
if str(_DEPLOY_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR.parent))

from loguru import logger

from deployment.protocol import (
    PROTOCOL_VERSION,
    validate_zmq_tcp_endpoint,
)
from robot_control.runtime.adapter import RobotAdapter
from robot_control.utils.config import load_config

if TYPE_CHECKING:
    from robot_control.recording.rollout import RolloutRecorder


class PolicyClient(Protocol):
    def call(self, observation: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class _QueuedPolicyCall:
    request: dict[str, Any]
    timeout_ms: int
    future: Future[dict[str, Any]]


class PolicyRequestWorker:
    """Own one ZMQ REQ socket on one thread for a whole policy stage.

    ZMQ sockets are thread-affine.  The worker is the only code that touches
    its socket while the main thread remains responsible for all robot motion.
    Calls are intentionally serialized (one in flight per policy endpoint).
    """

    def __init__(self, url: str, timeout_ms: int) -> None:
        self._url = url
        self._timeout_ms = int(timeout_ms)
        self._queue: queue.Queue[_QueuedPolicyCall | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="policy-zmq", daemon=True)
        self._thread.start()

    def submit(self, request: dict[str, Any], *, timeout_ms: int | None = None) -> Future[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Policy request worker is closed")
        future: Future[dict[str, Any]] = Future()
        call = _QueuedPolicyCall(
            request=request,
            timeout_ms=self._timeout_ms if timeout_ms is None else int(timeout_ms),
            future=future,
        )
        self._queue.put(call)
        return future

    def _run(self) -> None:
        client: PolicyClient | None = None
        try:
            client = build_policy_client(self._url, self._timeout_ms)
            while True:
                call = self._queue.get()
                if call is None:
                    return
                if call.future.set_running_or_notify_cancel():
                    try:
                        call.future.set_result(client.call(call.request, timeout_ms=call.timeout_ms))
                    except BaseException as exc:
                        call.future.set_exception(exc)
        finally:
            if client is not None:
                client.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=0.25)


class PolicyRequestSubmitter(Protocol):
    """Minimal interface used by the action scheduler and test doubles."""

    def submit(self, request: dict[str, Any], *, timeout_ms: int | None = None) -> Future[dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


class PickleZMQPolicyClient:
    """Small REQ/REP client for a pickle-ZMQ policy server."""

    def __init__(self, url: str, timeout_ms: int) -> None:
        import zmq

        url = validate_zmq_tcp_endpoint(url, bind=False)
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.connect(url)

    def call(self, observation: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        import zmq

        if timeout_ms is not None:
            effective_timeout_ms = max(1, int(timeout_ms))
            self._socket.setsockopt(zmq.RCVTIMEO, effective_timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, effective_timeout_ms)
        try:
            self._socket.send_pyobj(observation)
            response = self._socket.recv_pyobj()
        except Exception as exc:
            if isinstance(exc, zmq.Again):
                raise TimeoutError(
                    "Timed out waiting for the policy server. Check --policy-url, "
                    "network/firewall settings, or increase --timeout-ms for model inference."
                ) from exc
            raise
        if not isinstance(response, dict):
            raise TypeError(f"Policy response must be a dict, got {type(response)!r}")
        return response

    def close(self) -> None:
        self._socket.close(linger=0)


def build_policy_client(url: str, timeout_ms: int) -> PolicyClient:
    return PickleZMQPolicyClient(url=url, timeout_ms=timeout_ms)


def validate_response_envelope(request: dict[str, Any], response: dict[str, Any]) -> None:
    """Reject a mismatched correlated response before parsing any action."""
    response_version = response.get("protocol_version")
    if response_version is not None and response_version != PROTOCOL_VERSION:
        raise ValueError(
            f"Policy response protocol_version {response_version!r} does not match {PROTOCOL_VERSION!r}"
        )
    response_id = response.get("request_id")
    if response_id is not None and response_id != request.get("request_id"):
        raise ValueError(
            f"Policy response request_id {response_id!r} does not match request {request.get('request_id')!r}"
        )


def prepare_runtime_config(runtime_cfg: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Apply the one client-side safety override without mutating loaded config."""
    resolved = deepcopy(runtime_cfg)
    execution = resolved.setdefault("execution", {})
    if dry_run and str(execution.get("backend", "")).strip().lower() == "real":
        execution["enable_motion"] = False
    return resolved


DEPLOYMENT_DOMAIN_ID = 21


@dataclass(frozen=True)
class ChunkSchedule:
    """The complete policy-chunk contract used for one deployment session."""

    action_horizon: int
    execute_steps: int
    execute_fps: float

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action-horizon must be positive")
        if not 1 <= self.execute_steps <= self.action_horizon:
            raise ValueError(
                "execute-steps must be positive and no greater than action-horizon "
                f"(got {self.execute_steps} for horizon {self.action_horizon})"
            )
        if self.execute_fps <= 0.0:
            raise ValueError("execute-fps must be positive")


@dataclass(frozen=True)
class ObservationSample:
    """One aligned robot observation captured before an action command."""

    joint: np.ndarray
    eepose: np.ndarray
    snapshot: Any
    camera_alignment: dict[str, Any] | None
    state_sampled_at_s: float
    sampled_at_s: float
    components: dict[str, np.ndarray] | None = None


class ObservationBuilder:
    """Build the fixed joint-policy observation, including configured cameras."""

    def __init__(
        self,
        adapter: Any,
        robot_type: str,
        *,
        camera_fps: float,
        camera_max_skew_ms: float | None = None,
        camera_max_uses_per_frame: int = 3,
        camera_timeout_s: float = 5.0,
        state_getter: Callable[[], np.ndarray] | None = None,
    ) -> None:
        self.adapter = adapter
        self.robot_type = robot_type
        self.camera_timeout_s = max(0.01, float(camera_timeout_s))
        self.camera_fps = float(camera_fps)
        if self.camera_fps <= 0.0:
            raise ValueError("camera_fps must be positive")
        self._state_getter = state_getter

        # Robot YAML camera slots are the schema. Every declared slot is part
        # of the policy observation; an absent slot means no image feature.
        self.camera_alignment_enabled = bool(adapter.camera_keys)
        self.max_camera_skew_s = (
            1.0 / self.camera_fps
            if camera_max_skew_ms is None
            else float(camera_max_skew_ms) / 1000.0
        )
        self.max_uses_per_frame = int(camera_max_uses_per_frame)
        if self.max_camera_skew_s <= 0.0 or self.max_uses_per_frame <= 0:
            raise ValueError("camera timing limits must be positive")
        self.max_frame_age_s = self.max_uses_per_frame / self.camera_fps
        self._last_camera_sequences: dict[str, int] = {}
        self._camera_frame_uses: dict[str, int] = {}

    def build(
        self,
        task: str,
        schedule: ChunkSchedule,
        *,
        chunk_index: int = 0,
        execute_steps: int = 0,
    ) -> dict[str, Any]:
        sample = self.capture()
        joint = sample.joint
        eepose = sample.eepose
        snapshot = sample.snapshot
        camera_alignment = sample.camera_alignment

        observation: dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "protocol_version": PROTOCOL_VERSION,
            "robot_type": self.robot_type,
            "execution_profile": getattr(self.adapter, "execution_profile", ""),
            "observation.state": joint,
            "observation.state_eepose": eepose,
            "observation.state_ee": eepose,
            "proprio": joint,
            "language_instruction": task,
            # ``steps`` is retained for the existing policy-server protocol.
            "steps": schedule.action_horizon,
            "action_horizon": schedule.action_horizon,
            "execution_fps": schedule.execute_fps,
            "domain_id": DEPLOYMENT_DOMAIN_ID,
            "camera_sequences": dict(snapshot.sequences),
            "camera_captured_at_s": dict(snapshot.captured_at_s),
        }
        observation["chunk_index"] = int(chunk_index)
        observation["execute_steps"] = int(execute_steps)
        if camera_alignment is not None:
            observation["camera_alignment"] = camera_alignment

        for index, key in enumerate(self.adapter.camera_keys):
            image_key = f"observation.images.{key}"
            image = snapshot.images.get(image_key)
            if image is None:
                continue
            image = np.ascontiguousarray(image)
            observation[image_key] = image
            observation[f"image{index}"] = image
        return observation

    def capture(self, *, wait_for_aligned_camera: bool = True) -> ObservationSample:
        """Capture state and a camera snapshot.

        Policy observations wait for the configured camera alignment. Rollout
        samples use ``False`` so an optional recorder can never add the
        camera timeout to the action-control loop.
        """
        if wait_for_aligned_camera:
            snapshot, camera_alignment = self._get_aligned_snapshot()
        else:
            snapshot = self.adapter.get_camera_snapshot()
            camera_alignment = None
        state_sampled_at_s = time.perf_counter()
        if self._state_getter is None:
            joint = self.adapter.get_joint_state().astype(np.float32)
        else:
            joint = np.asarray(self._state_getter(), dtype=np.float32).reshape(-1)
        eepose = self.adapter.get_eepose_state().astype(np.float32)
        component_getter = getattr(self.adapter, "get_component_recording_state", None)
        component_values = component_getter() if callable(component_getter) else {}
        if not getattr(getattr(self.adapter, "robot", None), "feedback_usable", True):
            raise RuntimeError(
                "Robot feedback is invalid; cached state cannot be used for policy observations"
            )
        return ObservationSample(
            joint=joint,
            eepose=eepose,
            snapshot=snapshot,
            camera_alignment=camera_alignment,
            state_sampled_at_s=state_sampled_at_s,
            sampled_at_s=time.perf_counter(),
            components=component_values,
        )

    def _get_aligned_snapshot(self) -> tuple[Any, dict[str, Any] | None]:
        if not self.camera_alignment_enabled:
            return self.adapter.get_camera_snapshot(), None

        image_keys = [f"observation.images.{key}" for key in self.adapter.camera_keys]
        deadline = time.perf_counter() + self.camera_timeout_s
        last_reason = "missing"
        while True:
            sampled_at_s = time.perf_counter()
            snapshot = self.adapter.get_camera_snapshot()
            sequences: dict[str, int] = {}
            capture_times: dict[str, float] = {}
            last_reason = "missing"
            for key in image_keys:
                sequence = snapshot.sequences.get(key)
                captured_at_s = snapshot.captured_at_s.get(key)
                if sequence is None or captured_at_s is None:
                    break
                sequences[key] = int(sequence)
                capture_times[key] = float(captured_at_s)
            else:
                oldest = min(capture_times.values())
                newest = max(capture_times.values())
                skew_s = newest - oldest
                max_age_s = sampled_at_s - oldest
                last_reason = "skew"
                if skew_s <= self.max_camera_skew_s:
                    if max_age_s > self.max_frame_age_s:
                        raise RuntimeError(
                            "Camera frame is stale before a new frame arrived; "
                            f"max_uses_per_frame={self.max_uses_per_frame}"
                        )
                    frame_uses = {
                        key: self._camera_frame_uses.get(key, 0) + 1
                        if sequences[key] == self._last_camera_sequences.get(key)
                        else 1
                        for key in image_keys
                    }
                    if any(uses > self.max_uses_per_frame for uses in frame_uses.values()):
                        raise RuntimeError(
                            "Camera frame expired before a new frame arrived; "
                            f"max_uses_per_frame={self.max_uses_per_frame}"
                        )
                    self._last_camera_sequences = sequences
                    self._camera_frame_uses = frame_uses
                    return snapshot, {
                        "sampled_at_s": sampled_at_s,
                        "camera_skew_ms": skew_s * 1000.0,
                        "max_camera_age_ms": max_age_s * 1000.0,
                        "camera_frame_uses": frame_uses,
                    }

            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for aligned camera frames "
                    f"after {self.camera_timeout_s:.2f}s (last rejection={last_reason})"
                )
            time.sleep(0.002)


@dataclass(frozen=True)
class DeploymentSettings:
    robot: str
    runtime: str
    policy_url: str
    task: str
    timeout_ms: int
    schedule: ChunkSchedule
    chunk_num: int
    dry_run: bool
    rollout_data_path: str | None
    require_real_cameras: bool = False


@dataclass(frozen=True)
class PolicyStage:
    """One finite or continuous policy execution stage on an existing robot session."""

    name: str
    policy_url: str
    task: str
    schedule: ChunkSchedule
    chunk_num: int
    run_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("policy stage name must not be empty")
        if not self.task.strip():
            raise ValueError("policy stage task must not be empty")
        if self.chunk_num < 0:
            raise ValueError("policy stage chunk_num must be zero or positive")
        if self.run_seconds is not None and self.run_seconds <= 0.0:
            raise ValueError("policy stage run_seconds must be positive when set")


@dataclass(frozen=True)
class PolicyStageResult:
    """Result of one policy stage; no result implies no robot teardown occurred."""

    name: str
    completed_chunks: int
    requested_chunks: int
    completed: bool


@dataclass(frozen=True)
class ActionChunk:
    source: str
    values: np.ndarray


class JointActionParser:
    """Accept exactly one joint-action response format: ``action: (H, D)``."""

    def __init__(self, adapter: RobotAdapter, *, joint_dim: int | None = None) -> None:
        self.joint_dim = int(
            joint_dim if joint_dim is not None else adapter.get_joint_state().shape[0]
        )

    def parse(self, response: dict[str, Any]) -> ActionChunk:
        payload = self._unwrap(response)
        if "action" not in payload:
            raise KeyError("Policy response must contain a joint-action field named 'action'")
        values = self._to_array(payload["action"])
        if values.shape[1] != self.joint_dim:
            raise ValueError(
                f"action joint dimension mismatch: expected {self.joint_dim}, got {values.shape[1]}"
            )
        return ActionChunk(source="action", values=values)

    @staticmethod
    def _unwrap(response: dict[str, Any]) -> dict[str, Any]:
        if response.get("ok") is False:
            raise RuntimeError(f"Policy server error: {response.get('error', response)}")
        if "action" in response:
            return response
        data = response.get("data")
        if isinstance(data, dict):
            return data
        if "error" in response:
            raise RuntimeError(f"Policy server error: {response['error']}")
        return response

    @staticmethod
    def _to_array(value: Any) -> np.ndarray:
        values = np.asarray(value, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError(
                    f"Batched action must have batch size 1, got shape {values.shape}"
                )
            values = values[0]
        elif values.ndim != 2:
            raise ValueError(
                f"action must have shape (T,D), (1,T,D), or (D,), got {values.shape}"
            )
        if values.shape[0] <= 0 or values.shape[1] <= 0:
            raise ValueError(f"action chunk must not be empty, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Policy action chunk must contain only finite values")
        return np.ascontiguousarray(values, dtype=np.float32)


def validate_chunk_contract(chunk: ActionChunk, schedule: ChunkSchedule) -> None:
    """Refuse any action chunk whose length differs from the requested horizon."""
    actual_steps = int(chunk.values.shape[0])
    if actual_steps != schedule.action_horizon:
        raise ValueError(
            f"Policy chunk length mismatch: expected {schedule.action_horizon} steps, "
            f"got {actual_steps}. Refusing to execute a partial or stale chunk."
        )


def _limit_joint_step(
    adapter: RobotAdapter,
    previous: np.ndarray,
    target: np.ndarray,
    *,
    max_joint_step_rad: float,
    max_gripper_step: float,
) -> np.ndarray:
    previous = np.asarray(previous, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if previous.shape != target.shape:
        raise ValueError(f"Joint safety shape mismatch: previous={previous.shape}, target={target.shape}")

    arm_dim = min(len(adapter.state_names), previous.size)
    arm_previous = previous[:arm_dim]
    arm_target = target[:arm_dim]
    extra = target[arm_dim:]

    limits = np.full(arm_target.shape, max(0.0, float(max_joint_step_rad)), dtype=float)
    for index, name in enumerate(adapter.state_names[:arm_dim]):
        if "gripper" in name:
            limits[index] = max(0.0, float(max_gripper_step))
    enabled = limits > 0.0
    if not np.any(enabled):
        limited_arm = arm_target.copy()
    else:
        delta = arm_target - arm_previous
        limited = delta.copy()
        limited[enabled] = np.clip(delta[enabled], -limits[enabled], limits[enabled])
        if not np.allclose(limited, delta):
            logger.warning("Joint action step limited by deployment safety bounds")
        limited_arm = arm_previous + limited
    if extra.size == 0:
        return limited_arm
    return np.concatenate([limited_arm, extra])


def execute_chunk(
    adapter: RobotAdapter,
    chunk: ActionChunk,
    schedule: ChunkSchedule,
    *,
    dry_run: bool,
    max_joint_step_rad: float = 0.0,
    max_gripper_step: float = 0.0,
    debug_step: bool = False,
    stop_at_s: float | None = None,
) -> int:
    """Send only the trusted leading ``K`` joint targets of a validated chunk."""
    validate_chunk_contract(chunk, schedule)
    steps = schedule.execute_steps
    logger.info(
        "Executing joint chunk | source={} execute_steps={} horizon={} fps={} dim={}",
        chunk.source,
        steps,
        schedule.action_horizon,
        schedule.execute_fps,
        chunk.values.shape[1],
    )

    previous_joint = adapter.get_joint_state().astype(float)
    executed_steps = 0
    interval_s = 1.0 / schedule.execute_fps
    next_deadline = time.perf_counter()
    for index, action in enumerate(chunk.values[:steps]):
        if stop_at_s is not None and time.perf_counter() >= stop_at_s:
            logger.info("Model stage duration elapsed before chunk step {}/{}", index + 1, steps)
            break
        effective_action = _limit_joint_step(
            adapter,
            previous_joint,
            action,
            max_joint_step_rad=max_joint_step_rad,
            max_gripper_step=max_gripper_step,
        )
        previous_joint = effective_action
        if debug_step and not _confirm_debug_step(
            step_index=index,
            total_steps=steps,
            action=effective_action,
            dry_run=dry_run,
        ):
            logger.info("Debug-step execution stopped by operator before step {}", index + 1)
            break

        if dry_run:
            if index == 0 and not debug_step:
                logger.info(
                    "dry-run first effective action: {}",
                    np.array2string(effective_action, precision=4),
                )
        else:
            adapter.robot.move_to_joint(effective_action)
        if debug_step:
            logger.info("Debug step {}/{} command completed", index + 1, steps)

        executed_steps += 1
        next_deadline += interval_s
        remaining_s = next_deadline - time.perf_counter()
        if stop_at_s is not None:
            remaining_s = min(remaining_s, stop_at_s - time.perf_counter())
        if remaining_s > 0.0:
            time.sleep(remaining_s)
        elif index == 0 or index % 10 == 9:
            logger.warning(
                "Action execution missed its {:.1f}Hz deadline by {:.1f}ms at step {}/{}",
                schedule.execute_fps,
                -remaining_s * 1000.0,
                index + 1,
                steps,
            )

    if executed_steps > 0 and not dry_run:
        adapter.reset_pose_reference()
    return executed_steps


def _confirm_debug_step(
    *,
    step_index: int,
    total_steps: int,
    action: np.ndarray,
    dry_run: bool,
) -> bool:
    logger.warning(
        "Debug step {}/{} | action={}",
        step_index + 1,
        total_steps,
        np.array2string(action, precision=4),
    )
    operation = "continue dry-run" if dry_run else "execute this action"
    try:
        answer = _read_debug_key(f"Press Enter/Space to {operation}; press q to stop: ")
    except (EOFError, OSError, RuntimeError) as exc:
        logger.warning("Debug step has no usable terminal input ({}); stopping without execution", exc)
        return False
    if answer == "q":
        return False
    logger.info("Debug step {}/{} accepted; sending action", step_index + 1, total_steps)
    return True


def _read_debug_key(prompt: str) -> str:
    """Read one debug key from the controlling terminal and restore its mode."""
    import termios
    import tty

    stream = None
    close_stream = False
    try:
        try:
            stream = open("/dev/tty", "r", encoding="utf-8", buffering=1)
            close_stream = True
        except OSError:
            stream = sys.stdin
        fd = stream.fileno()
        if not os.isatty(fd):
            raise EOFError("stdin is not a TTY")
        if hasattr(os, "tcgetpgrp") and os.tcgetpgrp(fd) != os.getpgrp():
            raise RuntimeError("process is not in the terminal foreground")

        previous = termios.tcgetattr(fd)
        try:
            print(prompt, end="", flush=True)
            tty.setcbreak(fd)
            while True:
                key = stream.read(1)
                if not key:
                    raise EOFError("terminal input closed")
                if key.lower() == "q":
                    print("q", flush=True)
                    return "q"
                if key in {"\r", "\n", " "}:
                    print(flush=True)
                    return "continue"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    finally:
        if close_stream and stream is not None:
            stream.close()


@dataclass(frozen=True)
class _PendingInference:
    request: dict[str, Any]
    future: Future[dict[str, Any]]
    started_at_s: float
    trigger_step: int


def _validate_model_contract(
    response: dict[str, Any],
    *,
    schedule: ChunkSchedule,
    joint_dim: int,
) -> None:
    """Require the server to declare the same temporal and joint contract."""
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Policy response metadata must be a dict")
    expected = {
        "action_horizon": schedule.action_horizon,
        "action_dim": joint_dim,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Policy response {key}={metadata.get(key)!r} does not match required {value!r}"
            )
    if bool(metadata.get("recorded_action_replay")) and schedule.execute_steps != schedule.action_horizon:
        raise ValueError(
            "recorded-action fake server has no model to produce a K-step observation-conditioned chunk; "
            "use --execute-steps equal to --action-horizon"
        )
    action_fps = metadata.get("action_fps")
    if action_fps is None or not np.isclose(float(action_fps), schedule.execute_fps):
        raise ValueError(
            "Policy response action_fps must match robot execute_fps exactly: "
            f"server={action_fps!r} robot={schedule.execute_fps}"
        )


def _collect_response(
    pending: _PendingInference,
    *,
    action_parser: JointActionParser,
    schedule: ChunkSchedule,
    stage_deadline_s: float | None,
) -> tuple[ActionChunk, float, dict[str, Any]]:
    """Wait for a pending result without issuing a robot command."""
    try:
        if stage_deadline_s is None:
            response = pending.future.result()
        else:
            remaining_s = stage_deadline_s - time.perf_counter()
            if remaining_s <= 0.0:
                raise TimeoutError("Policy stage deadline elapsed while inference was pending")
            response = pending.future.result(timeout=remaining_s)
    except FutureTimeout as exc:
        raise TimeoutError("Policy stage deadline elapsed while inference was pending") from exc
    validate_response_envelope(pending.request, response)
    chunk = action_parser.parse(response)
    validate_chunk_contract(chunk, schedule)
    _validate_model_contract(response, schedule=schedule, joint_dim=action_parser.joint_dim)
    return chunk, time.perf_counter() - pending.started_at_s, response


def _command_joint_action(
    adapter: RobotAdapter,
    action: np.ndarray,
    previous_joint: np.ndarray,
    *,
    max_joint_step_rad: float,
    max_gripper_step: float,
    dry_run: bool,
    debug_step: bool,
    step_index: int,
    total_steps: int,
    command_fn: Callable[[np.ndarray], None] | None = None,
) -> tuple[np.ndarray, bool]:
    """Apply the existing joint safety filter and send exactly one action."""
    effective_action = _limit_joint_step(
        adapter,
        previous_joint,
        action,
        max_joint_step_rad=max_joint_step_rad,
        max_gripper_step=max_gripper_step,
    )
    if debug_step and not _confirm_debug_step(
        step_index=step_index,
        total_steps=total_steps,
        action=effective_action,
        dry_run=dry_run,
    ):
        return previous_joint, False
    if dry_run:
        if step_index == 0 and not debug_step:
            logger.info("dry-run first effective action: {}", np.array2string(effective_action, precision=4))
    elif command_fn is not None:
        command_fn(effective_action)
    else:
        adapter.robot.move_to_joint(effective_action)
        accepted_getter = getattr(adapter, "get_last_action_joint", None)
        if callable(accepted_getter):
            accepted = np.asarray(accepted_getter(), dtype=float).reshape(-1)
            if accepted.shape != effective_action.shape or not np.isfinite(accepted).all():
                raise ValueError("Accepted joint target differs from the active policy layout")
            effective_action = accepted
    from robot_control.monitor import publish_policy_action
    publish_policy_action(adapter, effective_action, dry_run=dry_run)
    return effective_action, True


def run_policy_stage(
    adapter: RobotAdapter,
    *,
    robot_type: str,
    camera_max_skew_ms: float = 80.0,
    camera_max_uses_per_frame: int = 3,
    stage: PolicyStage,
    timeout_ms: int,
    dry_run: bool,
    max_joint_step_rad: float = 0.0,
    max_gripper_step: float = 0.0,
    debug_step: bool = False,
    rollout_recorder: RolloutRecorder | None = None,
    worker_factory: Callable[[str, int], PolicyRequestSubmitter] = PolicyRequestWorker,
    state_getter: Callable[[], np.ndarray] | None = None,
    command_fn: Callable[[np.ndarray], None] | None = None,
    action_dim: int | None = None,
) -> PolicyStageResult:
    """Run one policy stage without creating or closing the robot/camera session.

    The policy client is scoped to this stage so each model can use its own ZMQ
    endpoint.  The caller owns ``adapter`` and can immediately run another
    stage using the same connected robot and camera handles.
    """
    if debug_step and stage.schedule.execute_steps < stage.schedule.action_horizon:
        raise ValueError("--debug-step requires --execute-steps to equal --action-horizon")
    from robot_control.monitor import notify, publish_policy_request, start_monitor
    start_monitor(adapter, "inference")
    notify(adapter, "attach_recorder", rollout_recorder)
    notify(adapter, "inference_started", stage=stage.name, task=stage.task, policy_url=stage.policy_url,
           horizon=stage.schedule.action_horizon, execute_steps=stage.schedule.execute_steps,
           execute_fps=stage.schedule.execute_fps, requested_chunks=stage.chunk_num, dry_run=dry_run)
    worker = worker_factory(stage.policy_url, timeout_ms)
    stage_failed = True
    try:
        observation_builder = ObservationBuilder(
            adapter=adapter,
            robot_type=robot_type,
            camera_fps=stage.schedule.execute_fps,
            camera_max_skew_ms=camera_max_skew_ms,
            camera_max_uses_per_frame=camera_max_uses_per_frame,
            state_getter=state_getter,
        )
        action_parser = JointActionParser(adapter, joint_dim=action_dim)
        from robot_control.monitor import publish_policy_layout
        publish_policy_layout(adapter, action_parser.joint_dim)
        logger.info(
            "Policy stage '{}' ready | profile={} state_dim={} cameras={} url={} "
            "H={} K={} execute_fps={} chunk_num={} run_seconds={} dry_run={}",
            stage.name,
            getattr(adapter, "execution_profile", ""),
            action_parser.joint_dim,
            adapter.camera_keys,
            stage.policy_url,
            stage.schedule.action_horizon,
            stage.schedule.execute_steps,
            stage.schedule.execute_fps,
            stage.chunk_num,
            stage.run_seconds,
            dry_run,
        )

        stage_deadline_s = (
            None
            if stage.run_seconds is None
            else time.perf_counter() + stage.run_seconds
        )
        first_request = observation_builder.build(
            stage.task,
            stage.schedule,
            chunk_index=0,
            execute_steps=0,
        )
        first_pending = _PendingInference(
            request=first_request,
            future=worker.submit(first_request),
            started_at_s=time.perf_counter(),
            trigger_step=0,
        )
        publish_policy_request(adapter, first_request)
        chunk, rtt_s, _response = _collect_response(
            first_pending,
            action_parser=action_parser,
            schedule=stage.schedule,
            stage_deadline_s=stage_deadline_s,
        )
        notify(adapter, "inference_response", request_id=first_request["request_id"], roundtrip_ms=rtt_s * 1000.,
               chunk_index=1, step_index=0)
        logger.info(
            "Policy stage '{}' received initial chunk | rtt={:.3f}s source={} shape={}",
            stage.name, rtt_s, chunk.source, chunk.values.shape,
        )

        active_index = 0
        activated_chunks = 1
        request_index = 0
        pending: _PendingInference | None = None
        previous_joint = np.asarray(
            (state_getter or adapter.get_joint_state)(),
            dtype=float,
        ).reshape(-1)
        interval_s = 1.0 / stage.schedule.execute_fps
        next_deadline_s = time.perf_counter()
        completed = False

        def _start_next_inference(*, trigger_step: int) -> _PendingInference:
            """Sample one observation and enqueue the one allowed next request."""
            nonlocal request_index
            request_index += 1
            observation = observation_builder.build(
                stage.task,
                stage.schedule,
                chunk_index=request_index,
                execute_steps=stage.schedule.execute_steps,
            )
            future = worker.submit(observation)
            publish_policy_request(adapter, observation)
            return _PendingInference(
                request=observation,
                future=future,
                started_at_s=time.perf_counter(),
                trigger_step=trigger_step,
            )

        def _activate_pending(next_pending: _PendingInference) -> None:
            """Switch to a prediction aligned with actions already sent.

            The request was sampled after ``trigger_step`` commands of the old
            chunk. While inference runs, any additional old commands advance
            the robot by the same number of prediction rows, so those leading
            rows of the new chunk are skipped at the switch boundary.
            """
            nonlocal active_index, activated_chunks, chunk, pending
            next_chunk, pending_rtt_s, _response = _collect_response(
                next_pending,
                action_parser=action_parser,
                schedule=stage.schedule,
                stage_deadline_s=stage_deadline_s,
            )
            old_tail_executed = min(
                stage.schedule.action_horizon - next_pending.trigger_step,
                max(0, active_index - next_pending.trigger_step),
            )
            discarded = max(0, stage.schedule.action_horizon - active_index)
            logger.info(
                "Policy stage '{}' switching at action boundary | trigger_step={} old_tail_executed={} "
                "discarded={} new_chunk_start={} rtt={:.3f}s",
                stage.name,
                next_pending.trigger_step,
                old_tail_executed,
                discarded,
                old_tail_executed,
                pending_rtt_s,
            )
            chunk = next_chunk
            active_index = old_tail_executed
            activated_chunks += 1
            pending = None
            notify(adapter, "inference_response", request_id=next_pending.request["request_id"],
                   roundtrip_ms=pending_rtt_s * 1000., chunk_index=activated_chunks,
                   step_index=active_index, skipped_steps=old_tail_executed)

        while True:
            if stage_deadline_s is not None and time.perf_counter() >= stage_deadline_s:
                logger.info("Policy stage '{}' reached run_seconds={}; stopping without another command", stage.name, stage.run_seconds)
                break

            if pending is not None and pending.future.done():
                _activate_pending(pending)

            if active_index >= stage.schedule.action_horizon:
                if pending is not None:
                    # The old chunk is exhausted. Waiting here never emits an
                    # extra stale command. The new chunk starts at H-K because
                    # the full old tail was already sent while waiting.
                    _activate_pending(pending)
                    continue
                if stage.chunk_num != 0 and activated_chunks >= stage.chunk_num:
                    completed = True
                    break
                pending = _start_next_inference(
                    trigger_step=stage.schedule.action_horizon,
                )
                continue

            rollout_sample: ObservationSample | None = None
            if rollout_recorder is not None and rollout_recorder.should_sample():
                try:
                    rollout_sample = observation_builder.capture(wait_for_aligned_camera=False)
                except Exception as exc:
                    rollout_recorder.disable(exc)
                    logger.exception("Rollout observation capture failed; policy continues")

            previous_joint, sent = _command_joint_action(
                adapter,
                chunk.values[active_index],
                previous_joint,
                max_joint_step_rad=max_joint_step_rad,
                max_gripper_step=max_gripper_step,
                dry_run=dry_run,
                debug_step=debug_step,
                step_index=active_index,
                total_steps=stage.schedule.action_horizon,
                command_fn=command_fn,
            )
            if not sent:
                logger.info("Policy stage '{}' stopped by operator before action {}", stage.name, active_index + 1)
                break

            if rollout_sample is not None and rollout_recorder is not None:
                try:
                    action_eepose_getter = getattr(adapter, "get_eepose_for_joint_action", None)
                    if callable(action_eepose_getter):
                        arm_action = previous_joint[: len(adapter.state_names)]
                        action_eepose = action_eepose_getter(arm_action)
                    else:
                        action_eepose = rollout_sample.eepose
                    timing = {
                        "sampled_at_s": rollout_sample.sampled_at_s,
                        "state_sampled_at_s": rollout_sample.state_sampled_at_s,
                        "action_commanded_at_s": time.perf_counter(),
                        "camera_sequences": dict(rollout_sample.snapshot.sequences),
                        "camera_captured_at_s": dict(rollout_sample.snapshot.captured_at_s),
                    }
                    if rollout_sample.camera_alignment is not None:
                        timing.update(rollout_sample.camera_alignment)
                    rollout_recorder.record(
                        observation_state=rollout_sample.joint,
                        observation_state_eepose=rollout_sample.eepose,
                        action_state=previous_joint,
                        action_eepose=action_eepose,
                        images=rollout_sample.snapshot.images,
                        timing=timing,
                        observation_components=_rollout_component_values(adapter, rollout_sample.components or {}, action=False),
                        action_components=_rollout_component_values(
                            adapter, getattr(adapter, "get_last_action_component_recording", lambda: {})(), action=True,
                        ),
                    )
                except Exception as exc:
                    rollout_recorder.disable(exc)
                    logger.exception("Rollout frame processing failed; policy continues")
            active_index += 1
            notify(adapter, "inference_update", step_index=active_index, chunk_index=activated_chunks,
                   remaining_steps=stage.schedule.action_horizon - active_index)

            should_prefetch = (
                pending is None
                and active_index == stage.schedule.execute_steps
                and (stage.chunk_num == 0 or activated_chunks < stage.chunk_num)
            )
            if should_prefetch:
                pending = _start_next_inference(trigger_step=active_index)
                logger.info(
                    "Policy stage '{}' started next inference after step {}/{}",
                    stage.name,
                    active_index,
                    stage.schedule.action_horizon,
                )

            next_deadline_s += interval_s
            remaining_s = next_deadline_s - time.perf_counter()
            if stage_deadline_s is not None:
                remaining_s = min(remaining_s, stage_deadline_s - time.perf_counter())
            if remaining_s > 0.0:
                time.sleep(remaining_s)
            elif active_index == 1 or active_index % 10 == 0:
                logger.warning(
                    "Action execution missed its {:.1f}Hz deadline by {:.1f}ms",
                    stage.schedule.execute_fps, -remaining_s * 1000.0,
                )

        if not dry_run:
            adapter.reset_pose_reference()
        stage_failed = False
        notify(adapter, "inference_update", phase="completed" if completed else "stopped")
        notify(adapter, "event", "policy_stage_finished", stage=stage.name, completed=completed)
        return PolicyStageResult(
            name=stage.name,
            completed_chunks=activated_chunks,
            requested_chunks=stage.chunk_num,
            completed=completed,
        )
    except KeyboardInterrupt:
        # Preserve the caller's existing operator-interrupt/home flow.
        stage_failed = False
        notify(adapter, "inference_update", phase="interrupted")
        raise
    except Exception as exc:
        notify(adapter, "inference_update", phase="error", error=str(exc))
        notify(adapter, "event", "policy_failed", message=str(exc))
        raise
    finally:
        try:
            if not dry_run and stage_failed:
                stop = getattr(adapter.robot, "safe_stop", None)
                if callable(stop):
                    stop()
        finally:
            worker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot Control joint-action policy client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot",
        default="configs/robots/dual_nero_gripper.yaml",
        help="Robot configuration (default: dual Nero gripper).",
    )
    parser.add_argument(
        "--runtime",
        default="configs/runtime/vr_record_real.yaml",
        help="Runtime configuration (default: real hardware profile).",
    )
    parser.add_argument("--policy-url", required=True, help="Policy server target, e.g. tcp://192.168.31.121:5557")
    parser.add_argument("--task", required=True, help="Language instruction sent to the policy.")
    parser.add_argument("--timeout-ms", type=int, default=120000, help="Maximum wait for one ZMQ request or reply.")
    parser.add_argument("--action-horizon", type=int, required=True, help="Required returned chunk length H.")
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=None,
        help="Trigger step K for background inference in every mode. Default: H.",
    )
    parser.add_argument("--execute-fps", type=float, required=True, help="Frequency for sending each joint target.")
    parser.add_argument("--chunk-num", type=int, default=1, help="Chunks to run; 0 runs until Ctrl-C.")
    parser.add_argument("--dry-run", action="store_true", help="Read observations and actions without enabling real motion.")
    parser.add_argument("--require-real-cameras", action="store_true", help="Require all three physical camera slots.")
    parser.add_argument(
        "--rollout-data-path",
        default=None,
        help="Optional LeRobot dataset root for recording inference rollouts; default: disabled.",
    )
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> DeploymentSettings:
    execute_steps = args.action_horizon if args.execute_steps is None else args.execute_steps
    schedule = ChunkSchedule(
        action_horizon=int(args.action_horizon),
        execute_steps=int(execute_steps),
        execute_fps=float(args.execute_fps),
    )
    if int(args.timeout_ms) <= 0:
        raise ValueError("timeout-ms must be positive")
    if int(args.chunk_num) < 0:
        raise ValueError("chunk-num must be zero or positive")
    rollout_data_path = str(args.rollout_data_path).strip() if args.rollout_data_path is not None else None
    if rollout_data_path is not None and rollout_data_path.lower() in {"", "none", "null"}:
        rollout_data_path = None
    task = str(args.task).strip()
    if not task:
        raise ValueError("task must not be empty")
    return DeploymentSettings(
        robot=str(args.robot),
        runtime=str(args.runtime),
        policy_url=str(args.policy_url),
        task=task,
        timeout_ms=int(args.timeout_ms),
        schedule=schedule,
        chunk_num=int(args.chunk_num),
        dry_run=bool(args.dry_run),
        rollout_data_path=rollout_data_path,
        require_real_cameras=bool(getattr(args, "require_real_cameras", False)),
    )


def _rollout_component_values(adapter, supplied, *, action):
    from robot_control.recording.components import recording_values
    return recording_values(adapter, supplied, action=action)


def build_rollout_recorder(
    settings: DeploymentSettings,
    *,
    adapter: RobotAdapter,
    robot_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
) -> RolloutRecorder | None:
    """Build optional rollout storage without making it a policy dependency."""
    if settings.rollout_data_path is None:
        return None

    # Keep the default inference-only path independent of the optional
    # recording stack and its pandas/pyarrow dependencies.
    from robot_control.recording.rollout import RolloutRecorder
    from robot_control.recording.components import recording_fields

    recording_cfg = runtime_cfg.get("recording", {})
    if "fps" not in recording_cfg:
        raise ValueError("runtime recording.fps is required when rollout recording is enabled")
    rollout_fps = int(recording_cfg["fps"])
    if rollout_fps <= 0:
        raise ValueError("runtime recording.fps must be positive when rollout recording is enabled")
    if rollout_fps > settings.schedule.execute_fps:
        raise ValueError(
            "runtime recording.fps must not exceed execute-fps when rollout recording is enabled: "
            f"recording={rollout_fps} execute={settings.schedule.execute_fps}"
        )

    root = Path(settings.rollout_data_path).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    # Rollouts use the current v3.0 dataset contract by default.  The
    # ordinary teleop recorder may still use recording.codebase_version=v2.1;
    # a rollout path is independent and must not silently mix the formats.
    return RolloutRecorder(
        root=root.resolve(),
        task=settings.task,
        robot_type=str(robot_cfg.get("robot_type", "unknown")),
        fps=rollout_fps,
        codebase_version="v3.0",
        camera_keys=list(adapter.camera_keys),
        camera_shapes=dict(adapter.camera_shapes),
        image_height=int(adapter.image_height),
        image_width=int(adapter.image_width),
        state_names=list(adapter.state_names),
        eepose_names=list(adapter.eepose_names),
        component_fields=recording_fields(getattr(adapter, "component_recording_fields", {})),
        camera_available=np.asarray([
            float(str(robot_cfg.get("camera", {}).get("slots", {}).get(key, {}).get("source") or "").lower()
                  not in ("", "null", "none", "dummy"))
            for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ]),
    )


def move_home(
    adapter: RobotAdapter,
    runtime_cfg: dict[str, Any],
    *,
    dry_run: bool,
    reason: str,
) -> bool:
    """Return to configured home only when motion is explicitly enabled."""
    if dry_run:
        logger.info("{} during dry-run; no return-home command will be sent.", reason)
        return False
    teleop_cfg = runtime_cfg.get("teleop", {})
    duration_s = float(teleop_cfg.get("home_duration_s", 2.0))
    steps = int(teleop_cfg.get("home_steps", 60))
    logger.warning(
        "{}; returning to configured home | duration={}s steps={}",
        reason,
        duration_s,
        steps,
    )
    adapter.safe_move_home(
        duration_s=duration_s,
        steps=steps,
        home_body=False,
        home_lift=False,
    )
    return True


def home_before_policy(adapter: RobotAdapter, runtime_cfg: dict[str, Any]) -> None:
    """Home arms before the first policy command. Dry-run callers skip this."""
    from robot_control.runtime.teleop import startup_home

    teleop_cfg = runtime_cfg.get("teleop", {})
    startup_home(
        adapter,
        duration_s=float(teleop_cfg.get("home_duration_s", 2.0)),
        steps=int(teleop_cfg.get("home_steps", 60)),
    )


def return_home_after_interrupt(
    adapter: RobotAdapter,
    runtime_cfg: dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    """Return home after an operator Ctrl-C; dry-run never sends motion."""
    return move_home(adapter, runtime_cfg, dry_run=dry_run, reason="Interrupted")


def stop_policy_motion(adapter: RobotAdapter, *, dry_run: bool) -> None:
    """Stop before recorder teardown, after any caller-owned home operation."""
    if dry_run:
        return
    stop = getattr(adapter.robot, "safe_stop", None)
    if callable(stop):
        stop()
    else:
        # Backends without a separate stop API stop on disconnect.
        adapter.robot.disconnect()


def prepare_policy_robot_config(robot_cfg: dict[str, Any], *, require_real_cameras: bool) -> dict[str, Any]:
    config = deepcopy(robot_cfg)
    if require_real_cameras:
        slots = config.get("camera", {}).get("slots", {})
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            slot = slots.get(key, {})
            if (str(slot.get("source") or "").lower() in ("", "null", "none", "dummy")
                    or slot.get("device") is None):
                raise ValueError(f"Configure physical camera {key} before visual policy validation")
    return config


def main() -> None:
    settings = settings_from_args(parse_args())
    robot_cfg = load_config(settings.robot, base_dir=PROJECT_ROOT)
    robot_cfg = prepare_policy_robot_config(
        robot_cfg, require_real_cameras=settings.require_real_cameras,
    )
    runtime_cfg = prepare_runtime_config(
        load_config(settings.runtime, base_dir=PROJECT_ROOT),
        dry_run=settings.dry_run,
    )

    adapter = RobotAdapter(robot_cfg=robot_cfg, runtime_cfg=runtime_cfg, project_root=PROJECT_ROOT)
    interrupted = False
    rollout_recorder: RolloutRecorder | None = None
    try:
        recording_cfg = runtime_cfg.get("recording", {})
        if adapter.camera_keys:
            adapter.ensure_cameras_ready(timeout_s=8.0)
        rollout_recorder = build_rollout_recorder(
            settings,
            adapter=adapter,
            robot_cfg=robot_cfg,
            runtime_cfg=runtime_cfg,
        )
        if not settings.dry_run:
            home_before_policy(adapter, runtime_cfg)

        run_policy_stage(
            adapter,
            robot_type=str(robot_cfg.get("robot_type", "unknown")),
            camera_max_skew_ms=float(recording_cfg.get("camera_max_skew_ms", 80.0)),
            camera_max_uses_per_frame=int(recording_cfg.get("camera_max_uses_per_frame", 3)),
            stage=PolicyStage(
                name="policy",
                policy_url=settings.policy_url,
                task=settings.task,
                schedule=settings.schedule,
                chunk_num=settings.chunk_num,
            ),
            timeout_ms=settings.timeout_ms,
            dry_run=settings.dry_run,
            rollout_recorder=rollout_recorder,
        )
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Policy interrupted by operator")
    finally:
        try:
            if interrupted:
                return_home_after_interrupt(adapter, runtime_cfg, dry_run=settings.dry_run)
        finally:
            try:
                stop_policy_motion(adapter, dry_run=settings.dry_run)
            finally:
                try:
                    if rollout_recorder is not None:
                        rollout_recorder.close(save=True)
                finally:
                    adapter.close()
                    logger.info("Robot joint-policy client stopped.")


if __name__ == "__main__":
    main()
