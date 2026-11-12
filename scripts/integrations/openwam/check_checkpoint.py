"""Validate a complete Wan2.2 model directory or a standalone Video DiT."""

import argparse
import json
from pathlib import Path

import torch
from _wan import check_schema, checkpoint_files, open_tensors, wan_schema
from convert_checkpoint import component_schemas, tokenizer_source

from xiconwam.actor.dit_checkpoint import video_state


def _tensor_report(tensors, expected, finite):
    report = check_schema({key: tuple(value.shape) for key, value in tensors.items()}, expected)
    report["invalid_dtype"] = [key for key, value in tensors.items() if value.dtype not in (torch.float32, torch.float16, torch.bfloat16)]
    report["nonfinite"] = [key for key, value in tensors.items() if finite and not torch.isfinite(value).all()]
    report["checked_finite"] = finite
    report["valid"] = report["valid"] and not (report["invalid_dtype"] or report["nonfinite"])
    return report


def _dit_report(source, expected, finite):
    if source.is_file() and source.suffix != ".safetensors":
        tensors, _ = video_state(source)
        return _tensor_report(tensors, expected, finite)
    with open_tensors(checkpoint_files(source, official_wan=True)) as readers:
        slices = {key: reader.get_slice(key) for key, reader in readers.items()}
        report = check_schema({key: tuple(value.get_shape()) for key, value in slices.items()}, expected)
        report["invalid_dtype"] = [key for key, value in slices.items() if value.get_dtype() not in {"BF16", "F16", "F32"}]
        report["nonfinite"] = [key for key, reader in readers.items() if finite and not torch.isfinite(reader.get_tensor(key)).all()]
        report["checked_finite"] = finite
        report["valid"] = report["valid"] and not (report["invalid_dtype"] or report["nonfinite"])
        return report


def _load_pth(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and len(state) == 1:
        value = next(iter(state.values()))
        if isinstance(value, dict):
            state = value
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Invalid state dictionary: {path}")
    return state


def check_checkpoint(source, *, finite=False, expected_schemas=None):
    source = Path(source)
    if source.is_file():
        expected = wan_schema() if expected_schemas is None else expected_schemas["dit"]
        return _dit_report(source, expected, finite)
    schemas = component_schemas() if expected_schemas is None else expected_schemas
    report = {"dit": _dit_report(source, schemas["dit"], finite)}
    for name, filename in (("vae", "Wan2.2_VAE.pth"), ("text_encoder", "models_t5_umt5-xxl-enc-bf16.pth")):
        path = source / filename
        if not path.is_file():
            report[name] = {"valid": False, "error": f"Missing {path}"}
            continue
        state = _load_pth(path)
        if name == "vae":
            state = {key if key.startswith("model.") else "model." + key: value for key, value in state.items()}
        report[name] = _tensor_report(state, schemas[name], finite)
    try:
        path = tokenizer_source(source)
        report["tokenizer"] = {"valid": True, "path": str(path)}
    except FileNotFoundError as exc:
        report["tokenizer"] = {"valid": False, "error": str(exc)}
    report["valid"] = all(item["valid"] for item in report.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Complete Wan directory or standalone Video DiT")
    parser.add_argument("--finite", action="store_true", help="Check every weight value is finite")
    args = parser.parse_args()
    report = check_checkpoint(args.checkpoint, finite=args.finite)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
