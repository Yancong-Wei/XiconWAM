"""Synthetic RoboDojo-shaped pipeline demo; no trained model, no robot imports."""

import argparse
import asyncio
import io
import json
from pathlib import Path

import numpy as np
from xicon_engine import PolicyClient, PolicyInfo, PolicyServer
from xicon_engine.validation import LeRobotEpisode, evaluate

from .policy import HoldPolicy


async def run(output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("demo output must be empty; choose a new directory")
    root = output / "synthetic_dataset"
    (root / "meta/episodes").mkdir(parents=True)
    (root / "data").mkdir()
    cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    info = PolicyInfo(
        "SYNTHETIC-HOLD-NOT-XICONWAM-INFERENCE",
        32,
        14,
        14,
        20.0,
        "fixture/robodojo-shaped-hold-v1",
        camera_keys=cameras,
        action_names=tuple(f"synthetic_axis_{i}" for i in range(14)),
        action_units=("rad",) * 14,
    )
    images = {}
    for i, key in enumerate(cameras):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, i] = 180
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        images[f"observation.images.{key}"] = {"bytes": buffer.getvalue()}
    (root / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": info.fps})
    )
    pq.write_table(
        pa.Table.from_pylist([{"episode_index": 0}]),
        root / "meta/episodes/file.parquet",
    )
    rows = []
    for frame in range(80):
        phase = frame / info.fps + np.arange(14) * 0.15
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame,
                "timestamp": frame / info.fps,
                "observation.state": (0.1 * np.sin(phase)).tolist(),
                "action": (0.1 * np.sin(phase + 1 / info.fps)).tolist(),
                **images,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/file.parquet")
    server = await PolicyServer(HoldPolicy(info), port=0).start()

    def validate():
        dataset = LeRobotEpisode(root)
        try:
            with PolicyClient(f"ws://127.0.0.1:{server.port}") as client:
                return evaluate(
                    dataset,
                    client,
                    output=output / "report",
                    image_map={k: f"observation.images.{k}" for k in cameras},
                    instruction="SYNTHETIC pipeline demo only",
                    execute_steps=10,
                )
        finally:
            dataset.close()

    try:
        summary = await asyncio.to_thread(validate)
        (output / "DEMO_ONLY.json").write_text(
            json.dumps(
                {
                    "data": "synthetic sine waves and colored camera tiles",
                    "policy": "hold current state across the horizon; no checkpoint loaded",
                    "purpose": "real WebSocket + Engine LeRobot v3 reader + pred/GT plots",
                    "report": str(output / "report"),
                },
                indent=2,
            )
            + "\n"
        )
        return summary
    finally:
        await server.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(run(args.output)), indent=2))


if __name__ == "__main__":
    main()
