"""Convert an OpenWAM checkpoint into a self-contained Wan2.2 model directory."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from _wan import check_schema, checkpoint_files, open_tensors, wan_schema


PREFIXES = {
    "dit": "video_backbone.dit.",
    "vae": "video_backbone.vae.",
    "text_encoder": "video_backbone.text_encoder.",
}


def component_schemas():
    from xiconwam.vision.wan.wan_video_text_encoder import WanTextEncoder
    from xiconwam.vision.wan.wan_video_vae import WanVideoVAE38

    with torch.device("meta"):
        vae = WanVideoVAE38()
        text = WanTextEncoder()
    return {
        "dit": wan_schema(),
        "vae": {key: tuple(value.shape) for key, value in vae.state_dict().items()},
        "text_encoder": {key: tuple(value.shape) for key, value in text.state_dict().items()},
    }


def tokenizer_source(source, tokenizer_dir=None):
    root = Path(source)
    if root.is_file():
        root = root.parent
    candidates = [Path(tokenizer_dir)] if tokenizer_dir else [
        root / "tokenizer/google/umt5-xxl", root / "google/umt5-xxl"
    ]
    for path in candidates:
        if (path / "tokenizer_config.json").is_file() and (
            (path / "tokenizer.json").is_file() or (path / "spiece.model").is_file()
        ):
            return path.resolve()
    raise FileNotFoundError("OpenWAM tokenizer files missing; pass --tokenizer-dir")


def convert(source, output, *, tokenizer_dir=None, expected_schemas=None):
    source_files = checkpoint_files(source)
    tokenizer = tokenizer_source(source, tokenizer_dir)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    schemas = component_schemas() if expected_schemas is None else expected_schemas
    output.parent.mkdir(parents=True, exist_ok=True)

    with open_tensors(source_files) as tensors:
        components, report = {}, {}
        for component, prefix in PREFIXES.items():
            keys = {key[len(prefix):]: key for key in tensors if key.startswith(prefix)}
            slices = {key: tensors[original].get_slice(original) for key, original in keys.items()}
            result = check_schema({key: tuple(value.get_shape()) for key, value in slices.items()}, schemas[component])
            result["invalid_dtype"] = [
                key for key, value in slices.items() if value.get_dtype() not in {"BF16", "F16", "F32"}
            ]
            result["valid"] = result["valid"] and not result["invalid_dtype"]
            report[component] = result
            components[component] = keys
        if not all(item["valid"] for item in report.values()):
            raise ValueError(json.dumps(report, indent=2))

        with tempfile.TemporaryDirectory(prefix=".openwam-convert-", dir=output.parent) as scratch:
            stage = Path(scratch) / "model"
            stage.mkdir()
            dit = components["dit"]
            save_file(
                {key: tensors[source_key].get_tensor(source_key) for key, source_key in dit.items()},
                stage / "diffusion_pytorch_model.safetensors",
                metadata={"format": "pt", "source": "OpenWAM-Alpha"},
            )
            vae = components["vae"]
            if any(not key.startswith("model.") for key in vae):
                raise ValueError("OpenWAM VAE weights must have a model. prefix")
            torch.save(
                {key.removeprefix("model."): tensors[source_key].get_tensor(source_key) for key, source_key in vae.items()},
                stage / "Wan2.2_VAE.pth",
            )
            text = components["text_encoder"]
            torch.save(
                {key: tensors[source_key].get_tensor(source_key) for key, source_key in text.items()},
                stage / "models_t5_umt5-xxl-enc-bf16.pth",
            )
            (stage / "google").mkdir()
            shutil.copytree(tokenizer, stage / "google/umt5-xxl")
            provenance = {
                "source": "OpenWAM", "source_files": [p.name for p in source_files],
                "tokenizer_source": str(tokenizer), "components": report,
            }
            (stage / "xiconwam_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
            stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="OpenWAM checkpoint file or directory")
    parser.add_argument("--output", type=Path, required=True, help="New model directory")
    parser.add_argument("--tokenizer-dir", type=Path, help="OpenWAM tokenizer/google/umt5-xxl directory")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, tokenizer_dir=args.tokenizer_dir), indent=2))


if __name__ == "__main__":
    main()
