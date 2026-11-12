"""Generate video from an image and a prompt with native or transferred Wan2.2 weights."""

import argparse
from pathlib import Path

import torch
from _wan import backbone_config
from PIL import Image

from xiconwam.data.video_manifest import image_tensor, resize_image
from xiconwam.kit.video_io import save_mp4
from xiconwam.actor.video import VideoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--attention-mask",
        choices=("first_frame_causal", "bidirectional"),
        default="first_frame_causal",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = backbone_config()
    cfg.update(
        model_id=str(args.wan_dir),
        load_text_encoder=True,
        tokenizer_max_len=512,
    )
    cfg["dit_config"]["video_attention_mask_mode"] = args.attention_mask
    model = VideoModel.from_backbone(
        cfg,
        device=args.device,
        torch_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype],
    ).eval()
    with Image.open(args.image) as image:
        rgb = image_tensor(resize_image(image, args.width, args.height))
    frames = model.infer_video(
        rgb,
        args.prompt,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
    )
    save_mp4(frames, str(args.output), fps=args.fps)


if __name__ == "__main__":
    main()
