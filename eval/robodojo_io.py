"""RoboDojo observation packing for the current XPolicyLab websocket server."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == (size_wh[1], size_wh[0]):
        return image
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize(size_wh, resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def stitch_robodojo_cameras(
    head: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    if (
        left.shape[0] != right.shape[0]
        or left.shape[1] + right.shape[1] != head.shape[1]
    ):
        raise ValueError(
            "Wrist images must form one row matching the head image width."
        )
    canvas = np.empty((head.shape[0] + left.shape[0], head.shape[1], 3), dtype=np.uint8)
    canvas[: head.shape[0]] = head
    canvas[head.shape[0] :, : left.shape[1]] = left
    canvas[head.shape[0] :, left.shape[1] :] = right
    return canvas


def observation_to_model_image(observation: dict[str, Any]) -> np.ndarray:
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        raise KeyError("XPolicyLab observation is missing the 'vision' mapping.")

    def camera_rgb(name: str) -> np.ndarray:
        camera = vision.get(name)
        if not isinstance(camera, dict) or "color" not in camera:
            raise KeyError(f"XPolicyLab observation is missing vision.{name}.color.")
        return np.asarray(camera["color"], dtype=np.uint8)

    head = resize_rgb(camera_rgb("cam_head"), (320, 256))
    left = resize_rgb(camera_rgb("cam_left_wrist"), (160, 128))
    right = resize_rgb(camera_rgb("cam_right_wrist"), (160, 128))
    return stitch_robodojo_cameras(head, left, right)


def pack_joint_state(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("XPolicyLab observation is missing the 'state' mapping.")
    keys = (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )
    missing = [key for key in keys if key not in state]
    if missing:
        raise KeyError(f"XPolicyLab observation is missing state keys: {missing}")
    vector = np.concatenate(
        [np.asarray(state[key], dtype=np.float32).reshape(-1) for key in keys]
    )
    if vector.shape != (14,):
        raise ValueError(f"Expected a 14-D joint state, got shape {vector.shape}.")
    return vector


def action_vector_to_xpolicylab(action: np.ndarray) -> dict[str, np.ndarray]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (14,):
        raise ValueError(f"Expected a 14-D RoboDojo action, got shape {vector.shape}.")
    return {
        "left_arm_joint_state": vector[:6],
        "left_ee_joint_state": vector[6:7],
        "right_arm_joint_state": vector[7:13],
        "right_ee_joint_state": vector[13:14],
    }
