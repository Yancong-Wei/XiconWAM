#!/usr/bin/env python3
"""Offline wiring check against the XiconWAM RoboDojo websocket server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CORE_ROOT = Path(__file__).resolve().parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from client_server.ws.model_client import WsModelClient


EXPECTED_KEYS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


def fake_observation(*, env_idx: int = 0) -> dict:
    return {
        "vision": {
            "cam_head": {
                "color": np.zeros((480, 640, 3), dtype=np.uint8),
            },
            "cam_left_wrist": {
                "color": np.zeros((480, 640, 3), dtype=np.uint8),
            },
            "cam_right_wrist": {
                "color": np.zeros((480, 640, 3), dtype=np.uint8),
            },
        },
        "instruction": "pick up the cube and place it on the platform",
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.zeros(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        },
        "env_idx": env_idx,
        "data_format_version": "v1.0",
    }


def validate_action_chunk(actions, *, replan_steps: int) -> None:
    if not isinstance(actions, list) or not actions:
        raise TypeError(f"get_action must return a non-empty list, got {type(actions)!r}")
    if len(actions) > replan_steps:
        raise ValueError(
            f"get_action returned {len(actions)} steps; expected <= {replan_steps}"
        )
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise TypeError(f"action[{index}] must be a dict, got {type(action)!r}")
        for key, width in EXPECTED_KEYS:
            if key not in action:
                raise KeyError(f"action[{index}] missing key {key!r}")
            vector = np.asarray(action[key], dtype=np.float32).reshape(-1)
            if vector.shape != (width,):
                raise ValueError(
                    f"action[{index}][{key}] shape {vector.shape}, expected ({width},)"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"action[{index}][{key}] contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--evaluation-id", default="xiconwam-debug-eval")
    parser.add_argument("--trial-id", default="xiconwam-debug-trial")
    args = parser.parse_args()

    url = f"ws://{args.host}:{args.port}"
    client = WsModelClient(
        url=url,
        evaluation_id=args.evaluation_id,
        trial_id=args.trial_id,
    )
    try:
        client.call(func_name="reset")
        client.call(func_name="update_obs", obs=fake_observation())
        actions = client.call(func_name="get_action")
        validate_action_chunk(actions, replan_steps=int(args.replan_steps))
        print(
            f"[debug_client] ok | steps={len(actions)} "
            f"keys={[key for key, _ in EXPECTED_KEYS]}"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
