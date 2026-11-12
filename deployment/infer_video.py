"""Run image-and-text visual prediction with any configured video backbone."""

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from xiconwam.actor.helpers.inference import configure_model_execution
from xiconwam.actor.video import VideoModel
from xiconwam.data.video_manifest import image_tensor, resize_image
from xiconwam.kit.config_resolvers import register_default_resolvers
from xiconwam.kit.video_io import save_mp4
from xiconwam.model_config import as_dict, video_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-name", default="task/wan_video")
    source.add_argument("--run-config", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument(
        "--frames", type=int, help="Defaults to 17 video frames or 2 Flux endpoints"
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args, overrides = parser.parse_known_args()
    register_default_resolvers()
    if args.run_config:
        cfg = OmegaConf.merge(
            OmegaConf.load(args.run_config), OmegaConf.from_dotlist(overrides)
        )
    else:
        root = Path(__file__).resolve().parents[2]
        with initialize_config_dir(
            config_dir=str(root / "configs"), version_base="1.3"
        ):
            cfg = compose(config_name=args.config_name, overrides=overrides)
    if args.output.exists():
        raise FileExistsError(args.output)
    backbone = as_dict(video_backbone(cfg.model))
    backbone["load_text_encoder"] = True
    if backbone.get("from_scratch"):
        raise ValueError(
            "This run uses random initialization for resume. Export its Video DiT "
            "as a complete model directory, set model.video_dit.model_id to it, "
            "and set model.video_dit.from_scratch=false."
        )
    model = VideoModel.from_backbone(
        backbone,
        device=args.device,
        torch_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype],
    )
    configure_model_execution(
        model, vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1)
    )
    model.eval()
    with Image.open(args.image) as image:
        rgb = image_tensor(resize_image(image, args.width, args.height))
    frames = (
        args.frames
        if args.frames is not None
        else (2 if backbone["name"] == "flux2" else 17)
    )
    result = model.infer_video(
        rgb,
        args.prompt,
        frames=frames,
        steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
    )
    save_mp4(result, str(args.output), fps=args.fps)


if __name__ == "__main__":
    main()
