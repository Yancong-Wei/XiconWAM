"""Shared pickle-ZMQ deployment protocol for XiconWAM real-robot serving.

Bound files (update together):
  - deployment/model_server.py         (XiconWAM serve; serial open-loop default)
  - deployment/robot_client.py         (robot_control client)
  - deployment/fake_lerobot_client.py  (offline open-loop LeRobot client)
  - deployment/protocol.py             (this file)

Multi-robot / multi-policy switching uses different ``tcp://ip:port`` binds.
Episode start is ``chunk_index==0`` with ``execute_steps==0``. Serial open-loop
uses ``execute_steps == action_horizon``.
"""
from __future__ import annotations

from typing import Any
import socket
from urllib.parse import urlsplit

import numpy as np


PROTOCOL_VERSION = "robot-control-deployment/v4"
ACTION_KEYS = ("action", "action_joint", "action_eepose", "action_ee")


def should_clear_rtc(request: dict[str, Any]) -> bool:
    """True when universal client scheduling starts a new control episode."""
    return request.get("chunk_index") is not None and int(request["chunk_index"]) == 0


def validate_zmq_tcp_endpoint(endpoint: str, *, bind: bool) -> str:
    """Validate the project's supported ``tcp://host:port`` endpoint form."""
    value = str(endpoint).strip()
    role = "server --bind" if bind else "robot --policy-url"
    example = "tcp://*:5556" if bind else "tcp://192.168.1.20:5556"
    if value.startswith("tcp/://") or ";" in value:
        raise ValueError(
            f"Invalid {role} endpoint {value!r}; use {example!r} (two slashes, colon before port)"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid {role} endpoint {value!r}; use {example!r}") from exc
    if parsed.scheme != "tcp" or not parsed.hostname or port is None or parsed.path:
        raise ValueError(f"Invalid {role} endpoint {value!r}; use {example!r}")
    if bind and parsed.hostname != "*":
        raise ValueError(
            f"Invalid server --bind {value!r}: policy servers always bind all interfaces; "
            f"use 'tcp://*:{port}'"
        )
    if not bind and parsed.hostname in {"*", "0.0.0.0", "::"}:
        raise ValueError(
            f"Invalid robot --policy-url {value!r}: wildcard addresses are server-only; "
            "use the policy machine IP, for example 'tcp://192.168.1.20:5556'"
        )
    return value


def policy_connect_targets(port: int) -> tuple[str, str | None]:
    """Return same-machine and best-effort LAN targets for server logs."""
    local = f"tcp://localhost:{int(port)}"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            lan_ip = probe.getsockname()[0]
        socket.inet_aton(lan_ip)
        return local, f"tcp://{lan_ip}:{int(port)}"
    except OSError:
        return local, None


def make_action_response(
    request: dict[str, Any],
    action: Any,
    *,
    action_key: str = "action",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if action_key not in ACTION_KEYS:
        raise ValueError(f"Unsupported action key: {action_key}")
    values = np.asarray(action, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Action must have non-empty shape (T,D), got {values.shape}")
    return {
        "ok": True,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.get("request_id"),
        action_key: np.ascontiguousarray(values),
        "metadata": dict(metadata or {}),
    }


def make_error_response(request: Any, error: str, **metadata: Any) -> dict[str, Any]:
    request_id = request.get("request_id") if isinstance(request, dict) else None
    return {
        "ok": False,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "error": str(error),
        "metadata": metadata,
    }


def get_action(response: dict[str, Any]) -> tuple[str, np.ndarray]:
    if response.get("ok") is False:
        raise RuntimeError(f"Policy server error: {response.get('error', response)}")
    for key in ACTION_KEYS:
        if key in response:
            return key, np.asarray(response[key], dtype=np.float32)
    raise KeyError(f"Policy response must contain one of {ACTION_KEYS}")
