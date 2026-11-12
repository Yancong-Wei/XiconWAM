"""Shared setup for the standalone Wan2.2 checkpoint tools."""

import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch
from omegaconf import OmegaConf
from safetensors import safe_open

from xiconwam.vision.wan.wan_video_dit import WanVideoDiT

OPENWAM_PREFIX = "video_backbone.dit."


def backbone_config():
    config = OmegaConf.load(ROOT / "configs/model/video_dit/wan22.yaml")
    return OmegaConf.to_container(config, resolve=True)


def checkpoint_files(source, *, official_wan=False):
    source = Path(source)
    if source.is_file():
        return [source]
    pattern = (
        "diffusion_pytorch_model*.safetensors" if official_wan else "*.safetensors"
    )
    files = sorted(source.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files under {source}")
    return files


@contextmanager
def open_tensors(files):
    with ExitStack() as stack:
        tensors = {}
        for path in files:
            reader = stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for key in reader.keys():  # noqa: SIM118 - safetensors readers are not mappings.
                if key in tensors:
                    raise ValueError(
                        f"Duplicate tensor {key!r} across checkpoint shards"
                    )
                tensors[key] = reader
        yield tensors


def wan_schema():
    config = backbone_config()["dit_config"]
    with torch.device("meta"):
        model = WanVideoDiT(**config)
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def check_schema(shapes, expected):
    missing = sorted(expected.keys() - shapes.keys())
    unexpected = sorted(shapes.keys() - expected.keys())
    mismatches = {
        key: {"found": shapes[key], "expected": expected[key]}
        for key in sorted(shapes.keys() & expected.keys())
        if shapes[key] != expected[key]
    }
    return {
        "valid": not (missing or unexpected or mismatches),
        "matched_tensors": len(expected) - len(missing) - len(mismatches),
        "expected_tensors": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": mismatches,
    }
