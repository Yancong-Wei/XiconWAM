#!/usr/bin/env python3
"""Websocket policy server for XiconWAM RoboDojo evaluation."""

from __future__ import annotations

import argparse
import ast
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

CORE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CORE_ROOT.parents[1]
for path in (str(CORE_ROOT), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from policy import Policy

logger = logging.getLogger(__name__)


def _parse_val(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _apply_overrides(cfg: dict[str, Any], tokens: list[str] | None) -> dict[str, Any]:
    if not tokens:
        return cfg
    if all(("=" in token and not token.startswith("-")) for token in tokens):
        for token in tokens:
            key, value = token.split("=", 1)
            cfg[key] = _parse_val(value)
        return cfg
    if len(tokens) % 2 != 0:
        raise ValueError(f"--overrides expects key value pairs, got: {tokens}")
    iterator = iter(tokens)
    for key in iterator:
        cfg[key.lstrip("-")] = _parse_val(next(iterator))
    return cfg


def _resolve_run_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir = os.environ.get("XICON_RUN_DIR") or cfg.get("run_dir")
    if run_dir:
        cfg["run_dir"] = run_dir
    checkpoint = os.environ.get("XICON_CHECKPOINT") or cfg.get("checkpoint")
    if checkpoint:
        cfg["checkpoint"] = checkpoint
    dataset_stats = os.environ.get("XICON_DATASET_STATS") or cfg.get("dataset_stats")
    if dataset_stats:
        cfg["dataset_stats"] = dataset_stats
    return cfg


def load_config(config_path: Path, overrides: list[str] | None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    cfg = _apply_overrides(cfg, overrides)
    cfg = _resolve_run_paths(cfg)
    if cfg.get("host") in (None, ""):
        cfg["host"] = "0.0.0.0"
    if cfg.get("port") in (None, ""):
        raise ValueError("port must be set via deploy.yml or --overrides")
    cfg["port"] = int(cfg["port"])
    return cfg


def serve(cfg: dict[str, Any]) -> None:
    model = Policy(cfg)
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=str(cfg["host"]),
            port=int(cfg["port"]),
            ws_ping_interval_s=float(cfg.get("ws_ping_interval_s", 20.0)),
            ws_ping_timeout_s=float(cfg.get("ws_ping_timeout_s", 20.0)),
        ),
    )
    logger.info(
        "XiconWAM RoboDojo policy server | host=%s port=%s run_dir=%s checkpoint=%s",
        cfg["host"],
        cfg["port"],
        cfg.get("run_dir"),
        cfg.get("checkpoint"),
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        logger.info("Policy server stopped")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        type=Path,
        default=CORE_ROOT / "deploy.yml",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config(Path(args.config_path).expanduser().resolve(), args.overrides)
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = int(args.port)
    # Keep cwd at the repo root so Hydra-relative assets resolve when callers
    # launch from elsewhere.
    os.chdir(REPO_ROOT)
    serve(cfg)


if __name__ == "__main__":
    main()
