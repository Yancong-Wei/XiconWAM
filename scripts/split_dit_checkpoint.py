"""Extract XiconWAM components and package a video model for independent loading.

The original training checkpoint contains the DiTs and policy state, not the
frozen video encoders. Pass the exact base model used in training for those.
"""

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from xiconwam.actor.dit_checkpoint import (
    ACTION_FORMAT,
    VIDEO_FORMAT,
    read_checkpoint,
    video_state,
)
from xiconwam.vision.registry import load_dit

COSMOS_DIT = Path("base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt")


def extract_joint(path):
    """Read a complete MoT/IDM/HIDE weights file, preserving action metadata."""
    payload = read_checkpoint(path)
    streams = payload.get("streams")
    if isinstance(streams, Mapping):
        prefixes = {"video": "mixtures.video.", "action": "mixtures.action."}
        state_encoder = payload.get("state_encoder")
        projector = None
    elif (
        isinstance(payload.get("dit"), Mapping)
        and payload.get("format") != VIDEO_FORMAT
    ):
        streams = payload["dit"]
        prefixes = {"video": "video_dit.", "action": "action_dit."}
        state_encoder = {
            k.removeprefix("state_encoder."): v
            for k, v in streams.items()
            if k.startswith("state_encoder.")
        } or None
        projector = {
            k.removeprefix("video_context_projector."): v
            for k, v in streams.items()
            if k.startswith("video_context_projector.")
        } or None
    else:
        raise ValueError(
            "Expected complete MoT/IDM/HIDE weights. LoRA-only weights and full "
            "DeepSpeed state directories cannot be split by this tool."
        )
    extracted = {}
    for name, prefix in prefixes.items():
        extracted[name] = {
            k[len(prefix) :]: v for k, v in streams.items() if k.startswith(prefix)
        }
        if not extracted[name]:
            raise ValueError(f"Joint checkpoint contains no {name} stream tensors")
    action = {
        "format": ACTION_FORMAT,
        "action_dit": extracted["action"],
        "state_encoder": state_encoder,
        "video_context_projector": projector,
        "state_position": payload["state_position"],
        "action_config": payload["action_config"],
        "step": payload.get("step"),
    }
    return extracted["video"], action, payload


def split_joint_weights(path, *, video_path=None, action_path=None):
    """Extract raw components for one-off use; CLI packages Video with its encoders."""
    selected = [Path(p).resolve() for p in (video_path, action_path) if p is not None]
    if Path(path).resolve() in selected or len(selected) != len(set(selected)):
        raise ValueError(
            "Joint checkpoint and component outputs must use distinct paths"
        )
    video, action, payload = extract_joint(path)
    if video_path is not None:
        output = Path(video_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".safetensors":
            metadata = {"format": "pt"}
            if payload.get("backbone_name"):
                metadata["backbone_name"] = payload["backbone_name"]
            save_file(
                {k: v.contiguous() for k, v in video.items()}, output, metadata=metadata
            )
        else:
            torch.save(
                {
                    "format": VIDEO_FORMAT,
                    "backbone_name": payload.get("backbone_name"),
                    "dit": video,
                    "step": payload.get("step"),
                },
                output,
            )
    if action_path is not None:
        output = Path(action_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(action, output)


def _config(path):
    cfg = OmegaConf.load(path)
    # Accept either a standalone backbone YAML or an entire saved run config.
    model = cfg.get("model")
    if model is not None:
        from xiconwam.model_config import video_backbone

        cfg = video_backbone(model)
    if cfg is None or not cfg.get("name"):
        raise ValueError(
            "Expected a backbone YAML or a saved run config with model.video_dit"
        )
    return OmegaConf.to_container(cfg, resolve=True)


def export_video(
    config, state, metadata, output, *, base_model=None, expected_schema=None
):
    name = config["name"]
    if name not in {"wan22", "cosmos25", "flux2"}:
        raise ValueError(f"Unsupported backbone: {name}")
    if metadata.get("backbone_name") not in (None, name):
        raise ValueError(f"Checkpoint backbone does not match {name}")
    if expected_schema is None:
        # Initialize Dynamo before entering the meta-device context; lazy import
        # inside a tensor operation can otherwise trigger a PyTorch circular import.
        from torch import _dynamo  # noqa: F401

        with torch.device("meta"):
            model = load_dit(
                {**config, "from_scratch": True},
                device="meta",
                torch_dtype=torch.float32,
            )
        expected_schema = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    missing = sorted(expected_schema.keys() - shapes.keys())
    unexpected = sorted(shapes.keys() - expected_schema.keys())
    mismatch = sorted(
        k
        for k in shapes.keys() & expected_schema.keys()
        if shapes[k] != expected_schema[k]
    )
    if missing or unexpected or mismatch:
        raise ValueError(
            f"Incompatible Video DiT: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={mismatch}"
        )
    if any(
        v.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        for v in state.values()
    ):
        raise ValueError("Video weights must have floating-point dtype")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if name in {"wan22", "cosmos25"}:
        if base_model is None:
            raise ValueError(
                "--base-model must point to the exact video model used during training"
            )
        base = Path(base_model).expanduser().resolve()
        if output == base or base in output.parents:
            raise ValueError("Video output must be outside the base model directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".video-export-", dir=output.parent
    ) as scratch:
        stage = Path(scratch) / "output"
        if name == "wan22":
            auxiliary = [
                base / "Wan2.2_VAE.pth",
                base / "models_t5_umt5-xxl-enc-bf16.pth",
            ]
            tokenizer = base / "google/umt5-xxl"
            for path in auxiliary:
                if not path.is_file():
                    raise FileNotFoundError(path)
            if not (tokenizer / "tokenizer_config.json").is_file() or not any(
                (tokenizer / f).is_file() for f in ("tokenizer.json", "spiece.model")
            ):
                raise FileNotFoundError(f"Wan tokenizer files missing in {tokenizer}")
            stage.mkdir()
            save_file(
                {k: v.contiguous() for k, v in state.items()},
                stage / "diffusion_pytorch_model.safetensors",
                metadata={"format": "pt"},
            )
            for path in auxiliary:
                shutil.copy2(path, stage / path.name)
            (stage / "google").mkdir()
            shutil.copytree(tokenizer, stage / "google/umt5-xxl")
            (stage / "xiconwam_source.json").write_text(
                json.dumps({"encoder_source": str(base)}, indent=2) + "\n"
            )
        elif name == "cosmos25":
            tokenizer = base / "tokenizer.pth"
            if not tokenizer.is_file():
                raise FileNotFoundError(tokenizer)
            (stage / COSMOS_DIT.parent).mkdir(parents=True)
            torch.save(state, stage / COSMOS_DIT)
            shutil.copy2(tokenizer, stage / "tokenizer.pth")
        else:
            if any(not k.startswith("transformer.") for k in state):
                raise ValueError("Flux Video DiT weights must have transformer. prefix")
            save_file(
                {
                    k.removeprefix("transformer."): v.contiguous()
                    for k, v in state.items()
                },
                stage,
                metadata={"format": "pt"},
            )
        stage.rename(output)
    return {"backbone": name, "output": str(output), "tensors": len(state)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="Joint weights or standalone Video DiT weights"
    )
    parser.add_argument(
        "--video-model", type=Path, help="Wan/Cosmos model directory or Flux DiT file"
    )
    parser.add_argument(
        "--action-output", type=Path, help="Action DiT and policy state output"
    )
    parser.add_argument(
        "--backbone-config", type=Path, help="Backbone YAML or saved run config"
    )
    parser.add_argument(
        "--base-model", type=Path, help="Wan/Cosmos model used in training"
    )
    args = parser.parse_args()
    if not args.video_model and not args.action_output:
        parser.error("Specify --video-model and/or --action-output")
    if args.video_model and not args.backbone_config:
        parser.error("--video-model requires --backbone-config")
    targets = [
        p.resolve() for p in (args.video_model, args.action_output) if p is not None
    ]
    if len(targets) != len(set(targets)) or args.checkpoint.resolve() in targets:
        parser.error("Input and output paths must be distinct")
    if any(p.exists() for p in targets):
        raise FileExistsError("An output already exists; select new output paths")
    action = None
    if args.action_output:
        video, action, metadata = extract_joint(args.checkpoint)
    elif args.checkpoint.suffix == ".safetensors":
        video, metadata = video_state(args.checkpoint)
    else:
        payload = read_checkpoint(args.checkpoint)
        if isinstance(payload.get("streams"), Mapping) or (
            isinstance(payload.get("dit"), Mapping)
            and payload.get("format") != VIDEO_FORMAT
        ):
            video, _, metadata = extract_joint(args.checkpoint)
        else:
            video, metadata = video_state(args.checkpoint)
    if args.video_model:
        report = export_video(
            _config(args.backbone_config),
            video,
            metadata,
            args.video_model,
            base_model=args.base_model,
        )
        print(json.dumps(report, indent=2))
    if action is not None:
        args.action_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(action, args.action_output)
        print(f"Action DiT and policy state: {args.action_output}")


if __name__ == "__main__":
    main()
