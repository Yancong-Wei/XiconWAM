#!/usr/bin/env python3
"""Fake robot client backed by a local LeRobot v3 dataset.

Open-loop offline validation against ``deployment/model_server.py`` (replay
recorded frames as observations). This is NOT a closed-loop robot evaluation.
Default K=H is the XiconWAM serial open-loop baseline. With any K<H, the client
simulates the same background K/H schedule used by ``robot_client.py``.
Measured RPC RTT is projected onto the dataset action clock. It validates
protocol and timing traces, not closed-loop robot quality.

Uses polars/pyarrow + imageio/PyAV only — no `lerobot` or `pandas` package.

Example:

    python deployment/fake_lerobot_client.py \\
      --url tcp://127.0.0.1:5556 \\
      --run-dir runs/robodojo_wan/<run_id> \\
      --lerobot-dataset-path playground/datasets/robodojo \\
      --episode-id 0 --frame-id 0 --chunk-num 3 --execute-steps 32 \\
      --instruction "pick up the cube and place it on the platform" \\
      --image-map \\
        cam_high=observation.images.cam_high \\
        cam_left_wrist=observation.images.cam_left_wrist \\
        cam_right_wrist=observation.images.cam_right_wrist
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    validate_zmq_tcp_endpoint,
)

logger = logging.getLogger(__name__)


def parse_key_values(items: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=DATASET_KEY mapping, got {item!r}")
        name, key = item.split("=", 1)
        name = name.strip()
        key = key.strip()
        if not name or not key:
            raise ValueError(f"Invalid mapping {item!r}")
        mapping[name] = key
    return mapping


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_one_parquet(path: Path) -> list[dict[str, Any]]:
    """Read parquet via polars (preferred) or pyarrow — no pandas / lerobot."""
    try:
        import polars as pl
    except ModuleNotFoundError:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading LeRobot parquet requires `polars` or `pyarrow`. "
                "Run `uv sync` for this project."
            ) from exc
        return pq.read_table(path).to_pylist()
    return pl.read_parquet(path).to_dicts()


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _read_one_parquet(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(path.rglob("*.parquet")):
        rows.extend(_read_one_parquet(parquet_path))
    if not rows:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return rows


def as_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def as_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def get_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    current: Any = row
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Key {key!r} not found in row")
        current = current[part]
    return current


def image_to_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        hi = float(np.nanmax(arr)) if arr.size else 1.0
        lo = float(np.nanmin(arr)) if arr.size else 0.0
        if lo >= -1.0 and hi <= 1.0:
            arr = (arr + 1.0) / 2.0 if lo < 0.0 else arr
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


class LocalLeRobot3:
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.info = read_json(self.root / "meta" / "info.json")
        self.fps = int(self.info.get("fps", 30))
        self.video_keys = tuple(
            key
            for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") == "video"
        )
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self.data = self._load_data()
        self._abs_to_pos = {
            int(as_scalar(row["index"])): pos for pos, row in enumerate(self.data)
        }
        self._video_readers: dict[Path, Any] = {}

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return {}
        tasks: dict[int, str] = {}
        for row in read_parquet_rows(tasks_path):
            task_index = int(as_scalar(row.get("task_index", row.get("index", 0))))
            task = row.get(
                "task", row.get("name", row.get("__index_level_0__", task_index))
            )
            tasks[task_index] = str(as_scalar(task))
        return tasks

    def _load_episodes(self) -> dict[int, dict[str, Any]]:
        episodes_path = self.root / "meta" / "episodes"
        if not episodes_path.exists():
            episodes_path = self.root / "meta" / "episodes.parquet"
        rows = read_parquet_rows(episodes_path)
        rows.sort(key=lambda row: int(as_scalar(row["episode_index"])))
        return {int(as_scalar(row["episode_index"])): row for row in rows}

    def _load_data(self) -> list[dict[str, Any]]:
        rows = read_parquet_rows(self.root / "data")
        rows.sort(key=lambda row: int(as_scalar(row["index"])))
        return rows

    def choose_sample(
        self, episode_id: int | None, frame_id: int | None, rng: random.Random
    ) -> dict[str, Any]:
        random_episode = episode_id is None
        if episode_id is None:
            episode_id = rng.choice(sorted(self.episodes))
        episode_rows = [
            row
            for row in self.data
            if int(as_scalar(row["episode_index"])) == int(episode_id)
        ]
        if not episode_rows:
            raise ValueError(f"No rows found for episode {episode_id}")
        if frame_id is not None:
            for row in episode_rows:
                if int(as_scalar(row.get("frame_index", row["index"]))) == int(
                    frame_id
                ):
                    return row
            raise ValueError(f"No frame_id={frame_id} in episode {episode_id}")
        return rng.choice(episode_rows) if random_episode else episode_rows[0]

    def episode_bounds_for_row(self, row: dict[str, Any]) -> tuple[int, int]:
        episode = self.episodes[int(as_scalar(row["episode_index"]))]
        return int(as_scalar(episode["dataset_from_index"])), int(
            as_scalar(episode["dataset_to_index"])
        )

    def next_row_in_episode(
        self, row: dict[str, Any], stride: int
    ) -> dict[str, Any] | None:
        _, ep_end = self.episode_bounds_for_row(row)
        next_abs_index = int(as_scalar(row["index"])) + max(1, int(stride))
        if next_abs_index >= ep_end:
            return None
        pos = self._abs_to_pos.get(next_abs_index)
        if pos is None:
            return None
        next_row = self.data[pos]
        if int(as_scalar(next_row["episode_index"])) != int(
            as_scalar(row["episode_index"])
        ):
            return None
        return next_row

    def task_for_row(self, row: dict[str, Any]) -> str | None:
        task_index = row.get("task_index")
        if task_index is None:
            return None
        return self.tasks.get(int(as_scalar(task_index)))

    def value_at_abs_index(self, key: str, abs_index: int) -> np.ndarray:
        pos = self._abs_to_pos[int(abs_index)]
        return as_array(get_value(self.data[pos], key), dtype=np.float32)

    def action_chunk(self, row: dict[str, Any], key: str, horizon: int) -> np.ndarray:
        episode = self.episodes[int(as_scalar(row["episode_index"]))]
        abs_idx = int(as_scalar(row["index"]))
        ep_start = int(as_scalar(episode["dataset_from_index"]))
        ep_end = int(as_scalar(episode["dataset_to_index"]))
        indices = [
            max(ep_start, min(ep_end - 1, abs_idx + t)) for t in range(int(horizon))
        ]
        return np.stack(
            [self.value_at_abs_index(key, idx) for idx in indices], axis=0
        ).astype(np.float32)

    def image_for_key(self, row: dict[str, Any], dataset_key: str) -> np.ndarray:
        if dataset_key in self.video_keys:
            return self._video_frame(row, dataset_key)
        return image_to_uint8(get_value(row, dataset_key))

    def _video_frame(self, row: dict[str, Any], video_key: str) -> np.ndarray:
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Video decoding requires imageio[ffmpeg]."
            ) from exc

        episode = self.episodes[int(as_scalar(row["episode_index"]))]
        video_path_template = self.info.get("video_path")
        if not video_path_template:
            raise KeyError("meta/info.json is missing video_path")
        video_path = self.root / video_path_template.format(
            video_key=video_key,
            chunk_index=int(as_scalar(episode[f"videos/{video_key}/chunk_index"])),
            file_index=int(as_scalar(episode[f"videos/{video_key}/file_index"])),
        )
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        timestamp = float(
            as_scalar(episode[f"videos/{video_key}/from_timestamp"])
        ) + float(as_scalar(row["timestamp"]))
        frame_index = max(0, int(round(timestamp * self.fps)))

        reader = self._video_readers.get(video_path)
        if reader is None:
            reader = imageio.get_reader(str(video_path), "ffmpeg")
            self._video_readers[video_path] = reader
        return image_to_uint8(reader.get_data(frame_index))

    def close(self) -> None:
        for reader in self._video_readers.values():
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        self._video_readers.clear()


class PickleZMQClient:
    def __init__(self, url: str, timeout_ms: int):
        import zmq

        url = validate_zmq_tcp_endpoint(url, bind=False)
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.connect(url)

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            self._socket.send_pyobj(request)
            response = self._socket.recv_pyobj()
        except Exception as exc:
            import zmq

            if isinstance(exc, zmq.Again):
                raise TimeoutError(
                    "Timed out waiting for XiconWAM model_server. Confirm it logged "
                    "'XiconWAM policy server listening', URL/port/firewall match, "
                    "or increase --timeout-ms."
                ) from exc
            raise
        if not isinstance(response, dict):
            raise TypeError(f"Response must be a dict, got {type(response)!r}")
        return response

    def close(self) -> None:
        self._socket.close(linger=0)


def build_request(
    dataset: LocalLeRobot3,
    row: dict[str, Any],
    image_map: dict[str, str],
    *,
    state_key: str,
    instruction: str,
    horizon: int | None,
    chunk_index: int = 0,
    execute_steps: int = 0,
) -> dict[str, Any]:
    """Build a v4 request with universal chunk scheduling context."""
    images = {
        request_name: dataset.image_for_key(row, dataset_key)
        for request_name, dataset_key in image_map.items()
    }
    state = as_array(get_value(row, state_key), dtype=np.float32)
    request = {
        "request_id": str(uuid.uuid4()),
        "protocol_version": PROTOCOL_VERSION,
        "language_instruction": instruction,
        "observation.state": state,
    }
    request["chunk_index"] = int(chunk_index)
    request["execute_steps"] = int(execute_steps)
    if horizon is not None:
        request["steps"] = int(horizon)
        request["action_horizon"] = int(horizon)
    for name, image in images.items():
        request[f"observation.images.{name}"] = image
    request["fake_lerobot"] = {
        "dataset_root": str(dataset.root),
        "episode_index": int(as_scalar(row["episode_index"])),
        "frame_index": int(as_scalar(row.get("frame_index", row["index"]))),
        "chunk_index": int(chunk_index),
    }
    return request


def request_images(request: dict[str, Any]) -> dict[str, np.ndarray]:
    prefix = "observation.images."
    return {
        key[len(prefix) :]: np.asarray(value)
        for key, value in request.items()
        if key.startswith(prefix)
    }


def response_action(response: dict[str, Any]) -> np.ndarray:
    if response.get("ok") is False:
        raise RuntimeError(f"Server returned error: {response.get('error', response)}")
    if "action" in response:
        return np.asarray(response["action"], dtype=np.float32)
    chunk = response.get("action_chunk")
    if isinstance(chunk, dict) and "values" in chunk:
        return np.asarray(chunk["values"], dtype=np.float32)
    raise KeyError(
        f"Response does not contain action/action_chunk.values: {list(response)}"
    )


def plot_cameras(images: dict[str, np.ndarray], output_path: Path) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        _plot_cameras_cv2(images, output_path)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = max(1, len(images))
    cols = min(3, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (name, image) in zip(axes_arr, images.items()):
        ax.imshow(image_to_uint8(image))
        ax.set_title(name)
        ax.axis("off")
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_actions(
    pred: np.ndarray, gt: np.ndarray | None, output_path: Path, *, error: bool = False
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        _plot_actions_cv2(pred, gt, output_path, error=error)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    dim = int(pred.shape[1])
    cols = min(4, max(1, dim))
    rows = int(math.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows), squeeze=False)
    axes_arr = axes.reshape(-1)
    x = np.arange(pred.shape[0])
    gt_arr = (
        None
        if gt is None or np.asarray(gt).size == 0
        else np.asarray(gt, dtype=np.float32)
    )
    if gt_arr is not None and gt_arr.ndim == 1:
        gt_arr = gt_arr[None, :]

    for d in range(dim):
        ax = axes_arr[d]
        if error and gt_arr is not None and d < gt_arr.shape[1]:
            n = min(pred.shape[0], gt_arr.shape[0])
            ax.plot(np.arange(n), pred[:n, d] - gt_arr[:n, d], label="pred-gt")
        else:
            ax.plot(x, pred[:, d], label="pred")
            if gt_arr is not None and d < gt_arr.shape[1]:
                ax.plot(np.arange(gt_arr.shape[0]), gt_arr[:, d], label="gt", alpha=0.8)
        ax.set_title(f"dim {d}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes_arr[dim:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_timeline(
    summaries: list[dict[str, Any]], output_path: Path, *, rtc: bool
) -> None:
    """Show exactly how an offline replay advanced its virtual control clock."""
    try:
        import matplotlib
    except ModuleNotFoundError:
        _plot_timeline_cv2(summaries, output_path, rtc=rtc)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = np.arange(len(summaries))
    rtt_ms = [float(item["rtt_ms"]) for item in summaries]
    virtual_steps = [int(item["virtual_old_steps"]) for item in summaries]
    discarded = [int(item["discarded_steps"]) for item in summaries]
    fig, (rtt_ax, steps_ax) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    rtt_ax.bar(indices, rtt_ms, color="#4C78A8")
    rtt_ax.set_ylabel("RTT (ms)")
    rtt_ax.grid(True, axis="y", alpha=0.25)
    steps_ax.bar(
        indices, virtual_steps, label="old steps during inference", color="#59A14F"
    )
    steps_ax.bar(
        indices,
        discarded,
        bottom=virtual_steps,
        label="discarded old tail",
        color="#E15759",
    )
    steps_ax.set_ylabel("old chunk steps")
    steps_ax.set_xlabel("model chunk request")
    steps_ax.grid(True, axis="y", alpha=0.25)
    steps_ax.legend(fontsize=8)
    fig.suptitle(
        "RTC-guided teacher-forced replay" if rtc else "no-RTC teacher-forced replay"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_rgb_png(image: np.ndarray, output_path: Path) -> None:
    """Write the small deployment diagnostics without requiring Matplotlib."""
    import cv2

    cv2.imwrite(str(output_path), cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))


def _plot_cameras_cv2(images: dict[str, np.ndarray], output_path: Path) -> None:
    import cv2

    names = list(images)
    cols = min(3, max(1, len(names)))
    rows = int(math.ceil(len(names) / cols))
    tile_h, tile_w = 240, 320
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, name in enumerate(names):
        image = image_to_uint8(images[name])
        resized = cv2.resize(image, (tile_w, tile_h - 24), interpolation=cv2.INTER_AREA)
        y, x = divmod(index, cols)
        top, left = y * tile_h, x * tile_w
        canvas[top + 24 : top + tile_h, left : left + tile_w] = resized
        cv2.putText(
            canvas,
            name,
            (left + 6, top + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    _write_rgb_png(canvas, output_path)


def _plot_line_cv2(
    canvas: np.ndarray,
    values: np.ndarray,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    minimum: float,
    maximum: float,
) -> None:
    import cv2

    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return
    span = max(1e-6, maximum - minimum)
    points = []
    for index, value in enumerate(data):
        x = left + (0 if data.size == 1 else int(index * (width - 1) / (data.size - 1)))
        y = top + height - 1 - int((float(value) - minimum) * (height - 1) / span)
        points.append((x, min(top + height - 1, max(top, y))))
    if len(points) == 1:
        cv2.circle(canvas, points[0], 2, color, -1)
    else:
        cv2.polylines(
            canvas, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA
        )


def _plot_actions_cv2(
    pred: np.ndarray, gt: np.ndarray | None, output_path: Path, *, error: bool
) -> None:
    import cv2

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    gt_arr = (
        None
        if gt is None or np.asarray(gt).size == 0
        else np.asarray(gt, dtype=np.float32)
    )
    if gt_arr is not None and gt_arr.ndim == 1:
        gt_arr = gt_arr[None, :]
    dim = int(pred.shape[1])
    cols = min(4, max(1, dim))
    rows = int(math.ceil(dim / cols))
    cell_w, cell_h = 280, 180
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for dim_index in range(dim):
        row, col = divmod(dim_index, cols)
        left, top = col * cell_w + 35, row * cell_h + 30
        width, height = cell_w - 50, cell_h - 55
        series = [pred[:, dim_index]]
        if error and gt_arr is not None and dim_index < gt_arr.shape[1]:
            count = min(pred.shape[0], gt_arr.shape[0])
            series = [pred[:count, dim_index] - gt_arr[:count, dim_index]]
        elif gt_arr is not None and dim_index < gt_arr.shape[1]:
            series.append(gt_arr[:, dim_index])
        low = min(float(np.min(values)) for values in series)
        high = max(float(np.max(values)) for values in series)
        if np.isclose(low, high):
            low, high = low - 1.0, high + 1.0
        cv2.rectangle(
            canvas, (left, top), (left + width, top + height), (190, 190, 190), 1
        )
        _plot_line_cv2(
            canvas,
            series[0],
            left=left,
            top=top,
            width=width,
            height=height,
            color=(45, 94, 175),
            minimum=low,
            maximum=high,
        )
        if len(series) == 2:
            _plot_line_cv2(
                canvas,
                series[1],
                left=left,
                top=top,
                width=width,
                height=height,
                color=(225, 87, 89),
                minimum=low,
                maximum=high,
            )
        label = f"dim {dim_index}" + (" err" if error else "")
        cv2.putText(
            canvas,
            label,
            (left, top - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    _write_rgb_png(canvas, output_path)


def _plot_timeline_cv2(
    summaries: list[dict[str, Any]], output_path: Path, *, rtc: bool
) -> None:
    import cv2

    width, height = 1000, 540
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin = 70
    count = max(1, len(summaries))
    bar_width = max(8, int((width - 2 * margin) / count) - 8)
    rtt = [float(item["rtt_ms"]) for item in summaries]
    old = [int(item["virtual_old_steps"]) for item in summaries]
    discarded = [int(item["discarded_steps"]) for item in summaries]

    def bars(
        values: list[float],
        top: int,
        graph_height: int,
        color: tuple[int, int, int],
        *,
        stacked: list[float] | None = None,
    ) -> None:
        maximum = max(
            1.0,
            max(
                (
                    value + (stacked[index] if stacked else 0.0)
                    for index, value in enumerate(values)
                ),
                default=0.0,
            ),
        )
        cv2.rectangle(
            canvas,
            (margin, top),
            (width - margin, top + graph_height),
            (190, 190, 190),
            1,
        )
        for index, value in enumerate(values):
            x = margin + int(index * (width - 2 * margin) / count) + 4
            base = 0.0 if stacked is None else stacked[index]
            y0 = top + graph_height - int(base * graph_height / maximum)
            y1 = top + graph_height - int((base + value) * graph_height / maximum)
            cv2.rectangle(canvas, (x, y1), (x + bar_width, y0), color, -1)

    cv2.putText(
        canvas,
        "RTC-guided teacher-forced replay" if rtc else "no-RTC teacher-forced replay",
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "RTT (ms)",
        (margin, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    bars(rtt, 70, 170, (76, 120, 168))
    cv2.putText(
        canvas,
        "old chunk steps (green=executed, red=discarded)",
        (margin, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    bars(old, 292, 170, (89, 161, 79))
    bars(discarded, 292, 170, (225, 87, 89), stacked=[float(value) for value in old])
    _write_rgb_png(canvas, output_path)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resolve_output_dir(
    *,
    run_dir: str | None,
    output_dir: str | None,
    episode_id: int | None,
    frame_id: int | None,
) -> Path:
    """Default open-loop artifacts under ``<run-dir>/openloop/``."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    if not run_dir:
        raise SystemExit(
            "Provide --run-dir (writes to <run-dir>/openloop/) or an explicit --output-dir"
        )
    ep = f"ep{episode_id}" if episode_id is not None else "ep_random"
    fr = f"f{frame_id}" if frame_id is not None else "f_random"
    return (Path(run_dir).expanduser().resolve() / "openloop" / f"{ep}_{fr}").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fake robot client for local LeRobot v3 policy validation"
    )
    parser.add_argument("--url", default="tcp://localhost:5556")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Training run directory; default --output-dir is <run-dir>/openloop/ep*_f*",
    )
    parser.add_argument("--lerobot-dataset-path", required=True)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument(
        "--chunk-num",
        type=int,
        default=0,
        help="Number of chunk queries. Default 0 replays until episode end.",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=None,
        help="Background inference trigger K; default is the returned horizon (serial open-loop).",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override artifact dir; default is <run-dir>/openloop/ep*_f*",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Requested action_horizon; default uses the server response length.",
    )
    parser.add_argument(
        "--image-map",
        nargs="*",
        default=[
            "cam_high=observation.images.cam_high",
            "cam_left_wrist=observation.images.cam_left_wrist",
            "cam_right_wrist=observation.images.cam_right_wrist",
        ],
        help="Mappings like cam_high=observation.images.cam_high.",
    )
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = resolve_output_dir(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        episode_id=args.episode_id,
        frame_id=args.frame_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LocalLeRobot3(args.lerobot_dataset_path)
    client = PickleZMQClient(args.url, timeout_ms=args.timeout_ms)
    try:
        image_map = parse_key_values(args.image_map)
        if not image_map:
            image_map = {key: key for key in dataset.video_keys}
        row = dataset.choose_sample(args.episode_id, args.frame_id, rng)
        dataset_instruction = dataset.task_for_row(row)
        instruction = args.instruction or dataset_instruction or ""

        all_pred: list[np.ndarray] = []
        all_gt: list[np.ndarray] = []
        all_requests: list[dict[str, Any]] = []
        all_responses: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        current_row = row
        chunk_index = 0
        virtual_time_s = 0.0
        full_episode = int(args.chunk_num) <= 0
        is_initial_request = True
        rtc_trigger_step: int | None = None
        while True:
            request = build_request(
                dataset,
                current_row,
                image_map,
                state_key=args.state_key,
                instruction=instruction,
                horizon=args.chunk_size,
                chunk_index=chunk_index,
                execute_steps=0 if is_initial_request else int(rtc_trigger_step or 0),
            )
            request_virtual_time_s = virtual_time_s
            events.append(
                {
                    "event": "request",
                    "chunk_index": chunk_index,
                    "request_id": request["request_id"],
                    "virtual_time_s": request_virtual_time_s,
                    "frame_index": request["fake_lerobot"]["frame_index"],
                    "schedule_context": True,
                }
            )
            t0 = time.perf_counter()
            response = client.call(request)
            rtt_ms = (time.perf_counter() - t0) * 1000
            if response.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError(
                    f"protocol_version mismatch: {response.get('protocol_version')!r} != {PROTOCOL_VERSION!r}"
                )
            pred = response_action(response)
            horizon = int(args.chunk_size or pred.shape[0])
            action_fps = float(
                ((response.get("metadata") or {}).get("action_fps")) or dataset.fps
            )
            if pred.shape[0] != horizon:
                raise ValueError(
                    f"Expected horizon {horizon}, server returned {pred.shape[0]}"
                )
            trigger_step = int(args.execute_steps or horizon)
            if not 1 <= trigger_step <= horizon:
                raise ValueError(
                    f"--execute-steps must be in [1,{horizon}], got {trigger_step}"
                )
            rtc_trigger_step = trigger_step
            virtual_old_steps = 0
            discarded_steps = 0
            if not is_initial_request:
                virtual_old_steps = min(
                    horizon - trigger_step,
                    max(0, int(math.floor((rtt_ms / 1000.0) * action_fps))),
                )
                discarded_steps = horizon - trigger_step - virtual_old_steps
            response_action_time_s = request_virtual_time_s + (
                virtual_old_steps / action_fps
            )
            events.append(
                {
                    "event": "response",
                    "chunk_index": chunk_index,
                    "request_id": request["request_id"],
                    "virtual_time_s": response_action_time_s,
                    "rtt_ms": rtt_ms,
                    "virtual_old_steps": virtual_old_steps,
                    "server_pi05_rtc": (response.get("metadata") or {}).get("pi05_rtc"),
                }
            )
            if not is_initial_request:
                events.append(
                    {
                        "event": "switch",
                        "chunk_index": chunk_index,
                        "virtual_time_s": response_action_time_s,
                        "old_tail_executed_steps": virtual_old_steps,
                    }
                )
                if discarded_steps:
                    events.append(
                        {
                            "event": "discard",
                            "chunk_index": chunk_index,
                            "virtual_time_s": response_action_time_s,
                            "discarded_old_tail_steps": discarded_steps,
                        }
                    )
            gt = dataset.action_chunk(current_row, args.action_key, horizon)
            all_requests.append(request)
            all_responses.append(response)
            all_pred.append(pred)
            all_gt.append(gt)
            summaries.append(
                {
                    "request_id": request["request_id"],
                    "episode_index": request["fake_lerobot"]["episode_index"],
                    "frame_index": request["fake_lerobot"]["frame_index"],
                    "pred_shape": list(pred.shape),
                    "gt_shape": list(gt.shape),
                    "rtt_ms": rtt_ms,
                    "mode": (
                        "server_rtc"
                        if (response.get("metadata") or {}).get("pi05_rtc")
                        else "generic"
                    ),
                    "trigger_step": trigger_step,
                    "virtual_old_steps": virtual_old_steps,
                    "discarded_steps": discarded_steps,
                    "request_virtual_time_s": request_virtual_time_s,
                    "response_action_time_s": response_action_time_s,
                    "server_timing": response.get("timing", {}),
                    "pi05_rtc": (response.get("metadata") or {}).get("pi05_rtc"),
                }
            )
            logger.info(
                "chunk=%s frame=%s pred_shape=%s gt_shape=%s rtt_ms=%.1f mode=%s virtual_old=%s discarded=%s pi05_rtc=%s",
                chunk_index,
                request["fake_lerobot"]["frame_index"],
                pred.shape,
                gt.shape,
                rtt_ms,
                (
                    "server_rtc"
                    if (response.get("metadata") or {}).get("pi05_rtc")
                    else "generic"
                ),
                virtual_old_steps,
                discarded_steps,
                (response.get("metadata") or {}).get("pi05_rtc"),
            )

            # The next observation is sampled at this chunk's K boundary, not
            # at the later switch boundary. If L old-tail commands ran while
            # waiting, the next chunk starts at row L and needs only K-L more
            # commands to reach its own K boundary.
            stride = trigger_step
            virtual_time_s = request_virtual_time_s + stride / action_fps
            next_row = dataset.next_row_in_episode(current_row, stride)
            chunk_index += 1
            is_initial_request = False
            if next_row is None:
                break
            if not full_episode and chunk_index >= max(1, int(args.chunk_num)):
                break
            current_row = next_row

        pred_all = np.concatenate(all_pred, axis=0)
        gt_all = (
            np.concatenate(all_gt, axis=0)
            if all_gt
            else np.empty((0, 0), dtype=np.float32)
        )
        first_request = all_requests[0]
        plot_cameras(request_images(first_request), output_dir / "cameras.png")
        plot_actions(pred_all, gt_all, output_dir / "actions_pred_vs_gt.png")
        plot_actions(pred_all, gt_all, output_dir / "actions_error.png", error=True)
        server_rtc = any(
            bool((response.get("metadata") or {}).get("pi05_rtc"))
            for response in all_responses
        )
        plot_timeline(
            summaries,
            output_dir
            / ("rtc_timeline.png" if server_rtc else "open_loop_timeline.png"),
            rtc=server_rtc,
        )

        np.savez_compressed(
            output_dir / "request_response.npz",
            request=np.asarray(first_request, dtype=object),
            requests=np.asarray(all_requests, dtype=object),
            responses=np.asarray(all_responses, dtype=object),
            pred_actions=pred_all,
            gt_actions=gt_all,
            state=np.asarray(first_request["observation.state"], dtype=np.float32),
            summaries=np.asarray(summaries, dtype=object),
        )
        summary = {
            "dataset": str(dataset.root),
            "episode_index": int(as_scalar(row["episode_index"])),
            "start_frame_index": int(as_scalar(row.get("frame_index", row["index"]))),
            "chunk_num_requested": int(args.chunk_num),
            "full_episode": full_episode,
            "server_rtc": server_rtc,
            "execute_steps": (
                None if args.execute_steps is None else int(args.execute_steps)
            ),
            "instruction": instruction,
            "image_keys": list(request_images(first_request).keys()),
            "state_shape": list(np.asarray(first_request["observation.state"]).shape),
            "pred_shape": list(pred_all.shape),
            "gt_shape": list(gt_all.shape),
            "chunks": summaries,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2)
        )
        (output_dir / "timeline.jsonl").write_text(
            "".join(
                json.dumps(json_safe(item), sort_keys=True) + "\n" for item in summaries
            ),
            encoding="utf-8",
        )
        (output_dir / "event_trace.jsonl").write_text(
            "".join(
                json.dumps(json_safe(item), sort_keys=True) + "\n" for item in events
            ),
            encoding="utf-8",
        )
        logger.info("Saved fake-client outputs to %s", output_dir)
    finally:
        client.close()
        dataset.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
