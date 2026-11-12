"""Download VPP2 Stage-2 and its Wan2.1 inference assets from Hugging Face."""

import argparse
from pathlib import Path

REPO_ID = "Haodong082399/VPP2"
STAGE2 = "checkpoints_video/vpp2-video-stage2-17f.pth"
WAN_DIR = "checkpoints/Wan2.1-I2V-14B-480P"
WAN_ASSETS = (
    "config.json",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "google/umt5-xxl/*",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="032f161")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    directory = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=args.revision,
            allow_patterns=[
                "config.json",
                STAGE2,
                *(f"{WAN_DIR}/{name}" for name in WAN_ASSETS),
            ],
            local_dir=args.output,
            max_workers=4,
        )
    )
    for name in (
        "config.json",
        STAGE2,
        *(f"{WAN_DIR}/{name}" for name in WAN_ASSETS[:-1]),
    ):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    if not any((directory / WAN_DIR / "google/umt5-xxl").glob("*")):
        raise FileNotFoundError("Teacher tokenizer files")
    print(directory / STAGE2)
    print(directory / WAN_DIR)


if __name__ == "__main__":
    main()
