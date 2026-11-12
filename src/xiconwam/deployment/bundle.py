"""Portable, versioned deployment artifacts. Export never instantiates a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from omegaconf import OmegaConf

from xiconwam.model_config import action_dit

FORMAT = "xiconwam.deployment"
VERSION = 1
logger = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or any(p in {"..", "."} for p in path.parts):
        raise ValueError(f"Invalid bundle member: {relative!r}")
    candidate = root.joinpath(*path.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle member escapes root: {relative!r}")
    if any(
        root.joinpath(*path.parts[:i]).is_symlink()
        for i in range(1, len(path.parts) + 1)
    ):
        raise ValueError(f"Bundle members must not be symlinks: {relative!r}")
    return candidate


def _environment():
    packages = {}
    for name in (
        "xiconwam",
        "xicon-engine",
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "omegaconf",
        "hydra-core",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _provenance():
    # Package-relative names stay stable between editable installs and wheels.
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {
        "git_commit": commit,
        "working_tree_dirty": bool(status) if commit else None,
        "python_sources_sha256": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*.py"))
        },
    }


def export_bundle(
    config,
    checkpoint,
    contract,
    output,
    *,
    base_dir=None,
    data_source=None,
    instructions=None,
):
    """Copy every active runtime asset and publish atomically to a NEW directory.

    Relative input resource paths follow the training working directory, supplied
    via base_dir (default: cwd). No datasets or optimizer states are copied.
    """
    from xicon_engine import PolicyInfo
    from hydra.utils import instantiate

    from xiconwam.data.lerobot.robot_video_dataset import (
        configured_window_steps,
        trained_window,
    )
    from xiconwam.data.lerobot.utils.normalizer import load_dataset_stats_from_json
    from xiconwam.deployment.policy import XiconWAMPolicy
    from xiconwam.kit.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config, checkpoint, contract = [
        Path(p).expanduser().resolve(strict=True)
        for p in (config, checkpoint, contract)
    ]
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Bundle output must be a new directory: {output}")
    base = Path(base_dir or Path.cwd()).expanduser().resolve(strict=True)
    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)
    original = OmegaConf.to_container(cfg, resolve=True)
    data = cfg.data.train
    if str(data.get("_target_", "")).endswith("build_mixture_dataset"):
        if not data_source or data_source not in data.sources:
            raise ValueError(
                "Mixture export requires --data-source selecting one training source"
            )
        entry = data.sources[data_source]
        if not entry.get("enabled", True):
            raise ValueError("Cannot export a disabled data source")
        data = entry.config
    elif data_source is not None:
        raise ValueError("--data-source is only valid for mixture training")
    cfg.data = {"train": OmegaConf.to_container(data, resolve=True)}
    data = cfg.data.train
    info = PolicyInfo.from_payload(json.loads(contract.read_text()))
    if (
        info.horizon != configured_window_steps(data) - 1
        or info.fps <= 0
        or list(data.camera_layout) != list(info.camera_keys)
    ):
        raise ValueError(
            "Contract horizon/cameras must match the selected training source"
        )
    for directory in data.dataset_dirs:
        metadata = Path(str(directory)).expanduser()
        metadata = (
            metadata if metadata.is_absolute() else base / metadata
        ) / "meta/info.json"
        if metadata.is_file():
            fps = float(json.loads(metadata.read_text())["fps"]) / int(
                data.global_sample_stride
            )
            if abs(fps - info.fps) > 1e-6:
                raise ValueError(
                    f"Contract fps={info.fps} differs from sampled dataset fps={fps}"
                )
    if data.concat_multi_camera != "robotwin" or list(data.video_size) != [384, 320]:
        raise ValueError("Current serving requires robotwin video_size=[384,320]")
    video_frames, _ = trained_window(
        configured_window_steps(data), data.action_video_freq_ratio
    )
    processor = instantiate(data.processor)
    if processor.drop_high_level_prob != 1.0:
        raise ValueError(
            "Bundle serving requires deterministic low-level instruction selection"
        )

    def source(value):
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else base / path).resolve(strict=True)

    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(source(data.pretrained_norm_stats))
    )
    XiconWAMPolicy(None, processor, info, num_video_frames=video_frames)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    bindings, copied = {}, {}
    try:

        def asset(key, value):
            src = source(value)
            if src == output or output.is_relative_to(src):
                raise ValueError("Bundle output cannot be inside an input asset")
            for previous, relative in copied.items():
                if src == previous or (
                    previous.is_dir() and src.is_relative_to(previous)
                ):
                    relative = (Path(relative) / src.relative_to(previous)).as_posix()
                    break
            else:
                relative = f"assets/{len(copied):02d}-{src.name}"
                dst = stage / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    if any(p.is_symlink() and p.is_dir() for p in src.rglob("*")):
                        raise ValueError(
                            f"Materialize nested directory symlinks before export: {src}"
                        )
                    shutil.copytree(
                        src,
                        dst,
                        symlinks=False,
                        ignore=shutil.ignore_patterns(
                            ".git", ".cache", "__pycache__", ".DS_Store", ".env", ".aws"
                        ),
                    )
                else:
                    shutil.copyfile(src, dst)
                copied[src] = relative
            bindings[key] = relative
            OmegaConf.update(cfg, key, relative)

        # These are the supported backend loader's active filesystem inputs.
        backbone_key = next(
            key
            for key in ("video_dit", "video_backbone", "backbone")
            if cfg.model.get(key) is not None
        )
        backbone = cfg.model[backbone_key]
        active = ["model_id", "model_path", "ae_model_path"]
        if backbone.get("load_text_encoder", False):
            active += ["reason_model_id", "qwen3_model_spec"]
        for key in active:
            value = backbone.get(key)
            if value:
                asset(f"model.{backbone_key}.{key}", value)
        action = action_dit(cfg.model)
        if action is not None:
            action_key = (
                "action_dit"
                if cfg.model.get("action_dit") is not None
                else (
                    "action_backbone"
                    if cfg.model.get("action_backbone") is not None
                    else f"{backbone_key}.action_dit"
                )
            )
            key = "checkpoint" if action.get("checkpoint") else "pretrained_path"
            if action.get(key) and not (
                key == "pretrained_path" and action.get("from_scratch", False)
            ):
                asset(f"model.{action_key}.{key}", action[key])
        elif backbone.get("action_dit_pretrained_path"):
            asset(
                f"model.{backbone_key}.action_dit_pretrained_path",
                backbone.action_dit_pretrained_path,
            )
        if cfg.model.get("action_dit_pretrained_path"):
            asset(
                "model.action_dit_pretrained_path", cfg.model.action_dit_pretrained_path
            )
        asset("data.train.pretrained_norm_stats", data.pretrained_norm_stats)
        cache = data.get("text_embedding_cache_dir")
        if not backbone.get("load_text_encoder", False) and not cache:
            raise ValueError(
                "A text cache is required when the text encoder is disabled"
            )
        if cache:
            asset("data.train.text_embedding_cache_dir", cache)
        # No training dataset access or online cache fallback during deployment.
        data.dataset_dirs = []
        data.load_text_encoder_on_miss = False
        data.backbone = OmegaConf.to_container(backbone, resolve=True)
        for key, value in list(bindings.items()):
            if key.startswith(f"model.{backbone_key}."):
                bindings[
                    key.replace(f"model.{backbone_key}.", "data.train.backbone.")
                ] = value
        shutil.copyfile(checkpoint, stage / "checkpoint.pt")
        (stage / "contract.json").write_text(
            json.dumps(info.to_payload(), indent=2) + "\n"
        )
        OmegaConf.save(cfg, stage / "config.yaml")
        OmegaConf.save(OmegaConf.create(original), stage / "training_config.yaml")
        # Validate cached payloads and record their prompt identities.
        prompts = []
        if cache:
            import torch
            from xiconwam.data.lerobot.text_embedding_cache import (
                load_text_embedding_cache,
                text_embedding_cache_filename,
            )
            from xiconwam.kit.pytorch_utils import mixed_precision_to_dtype

            dtype = mixed_precision_to_dtype(cfg.mixed_precision)
            pattern = text_embedding_cache_filename(
                "*",
                int(data.context_len),
                str(data.text_encoder_id),
                is_hash=True,
                dtype=dtype,
            )
            for path in sorted(
                (stage / bindings["data.train.text_embedding_cache_dir"]).glob(pattern)
            ):
                payload = load_text_embedding_cache(
                    path,
                    int(data.context_len),
                    str(data.text_encoder_id),
                    expected_dtype=mixed_precision_to_dtype(cfg.mixed_precision),
                )
                digest = payload.get("prompt_hash")
                if not digest:
                    raise ValueError(f"Text cache lacks prompt identity: {path.name}")
                expected = text_embedding_cache_filename(
                    digest,
                    int(data.context_len),
                    str(data.text_encoder_id),
                    is_hash=True,
                    dtype=mixed_precision_to_dtype(cfg.mixed_precision),
                )
                if (
                    path.name != expected
                    or not torch.isfinite(payload["context"]).all()
                    or payload["context"].shape[-1] != int(backbone.text_dim)
                ):
                    raise ValueError(f"Invalid text cache identity/values: {path.name}")
                prompts.append(
                    {"prompt_hash": digest, "path": path.relative_to(stage).as_posix()}
                )
            if not backbone.get("load_text_encoder", False) and not prompts:
                raise ValueError("Bundle text cache contains no embeddings")
        if instructions and not backbone.get("load_text_encoder", False):
            from xiconwam.data.lerobot.prompts import DEFAULT_PROMPT
            from xiconwam.data.lerobot.text_embedding_cache import prompt_hash

            cached = {entry["prompt_hash"] for entry in prompts}
            for instruction in instructions:
                prompt = DEFAULT_PROMPT.format(
                    task=(
                        data.override_instruction
                        if data.get("override_instruction") is not None
                        else processor.augment_instruction({"task": instruction})
                    )
                )
                if prompt_hash(prompt) not in cached:
                    raise ValueError(
                        f"Missing cached embedding for required instruction: {instruction!r}"
                    )
        files = {
            p.relative_to(stage).as_posix(): {
                "sha256": sha256(p),
                "size": p.stat().st_size,
            }
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "format": FORMAT,
            "format_version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "entrypoints": {
                "config": "config.yaml",
                "checkpoint": "checkpoint.pt",
                "contract": "contract.json",
            },
            "resource_bindings": bindings,
            "files": files,
            "policy": info.to_payload(),
            "data_source": data_source,
            "text_prompts": prompts,
            "instructions": list(instructions or []),
            "environment": _environment(),
            "code": _provenance(),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_bundle(stage)
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_bundle(directory):
    """Verify inventory and hashes before model allocation. This is not a signature."""
    root = Path(directory).expanduser().resolve(strict=True)
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Bundle tree must not contain symlinks")
    manifest_path = _member(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("format_version") != VERSION:
        raise ValueError("Unsupported deployment bundle format/version")
    files = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Bundle file inventory mismatch (missing or extra files)")
    for relative, record in files.items():
        path = _member(root, relative)
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {relative}")
    entries = manifest["entrypoints"]
    for name in ("config", "checkpoint", "contract"):
        if entries[name] not in files:
            raise ValueError(f"Untracked bundle entrypoint: {name}")
    cfg = OmegaConf.load(_member(root, entries["config"]))
    for key, relative in manifest["resource_bindings"].items():
        path = _member(root, relative)
        if not path.exists() or OmegaConf.select(cfg, key) != relative:
            raise ValueError(f"Bundle resource binding mismatch: {key}")
        if not (relative in files or any(f.startswith(relative + "/") for f in files)):
            raise ValueError(f"Untracked bundle resource: {relative}")
    contract = json.loads(_member(root, entries["contract"]).read_text())
    if contract != manifest["policy"]:
        raise ValueError("Manifest policy differs from contract")
    return manifest


def load_bundle_config(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = verify_bundle(root)
    current = _environment()
    expected_sources = manifest["code"]["python_sources_sha256"]
    if expected_sources != _provenance()["python_sources_sha256"]:
        logger.warning(
            "Installed XiconWAM sources differ from the exported bundle; verify code/checkpoint compatibility"
        )
    for package, expected in manifest["environment"]["packages"].items():
        actual = current["packages"].get(package)
        if expected != actual:
            logger.warning(
                "Bundle environment differs: %s exported=%s installed=%s",
                package,
                expected,
                actual,
            )
    cfg = OmegaConf.load(root / manifest["entrypoints"]["config"])
    for key, relative in manifest["resource_bindings"].items():
        OmegaConf.update(cfg, key, str(_member(root, relative)))
    return (
        cfg,
        root / manifest["entrypoints"]["checkpoint"],
        root / manifest["entrypoints"]["contract"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export or verify a portable XiconWAM deployment bundle"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("config", "checkpoint", "contract", "output"):
        export.add_argument(f"--{name}", required=True)
    export.add_argument(
        "--base-dir",
        help="Training working directory for relative asset paths (default: cwd)",
    )
    export.add_argument(
        "--data-source", help="Required for mixture runs; selects the deployed source"
    )
    export.add_argument(
        "--instruction",
        dest="instructions",
        action="append",
        help="Require a cached instruction; repeat for every deployment task",
    )
    verify = commands.add_parser("verify")
    verify.add_argument("directory")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    manifest = (
        export_bundle(**args)
        if command == "export"
        else verify_bundle(args["directory"])
    )
    print(
        json.dumps(
            {
                "format": manifest["format"],
                "version": manifest["format_version"],
                "files": len(manifest["files"]),
                "policy": manifest["policy"]["name"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
