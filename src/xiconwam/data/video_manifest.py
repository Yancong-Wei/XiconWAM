"""RGB windows for standalone video pretraining and video distillation."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("video"), str):
                raise TypeError(f"{path}:{line_number} requires a video path")
            if not isinstance(row.get("prompt"), str):
                raise TypeError(f"{path}:{line_number} requires a prompt string")
            for key in ("video", "condition_image"):
                if row.get(key) is not None:
                    row[key] = str(
                        (path.parent / Path(row[key]).expanduser()).resolve()
                    )
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty video manifest: {path}")
    return rows


def read_frames(path, *, start, stride, count):
    import av

    if start < 0 or min(stride, count) < 1:
        raise ValueError(
            "Video frame start must be nonnegative and stride/count positive"
        )
    frames = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start and (index - start) % stride == 0:
                frames.append(frame.to_image())
                if len(frames) == count:
                    return frames
    raise ValueError(
        f"{path}: wanted {count} frames from {start} with stride {stride}, decoded {len(frames)}"
    )


def resize_image(image, width, height):
    return ImageOps.fit(
        image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS
    )


def image_tensor(image):
    return (
        torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(127.5).sub_(1)
    )


class VideoManifestDataset(Dataset):
    is_video_manifest = True

    def __init__(self, manifest, frames=17, height=256, width=416):
        self.rows = load_manifest(manifest)
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)
        if (
            self.frames < 5
            or (self.frames - 1) % 4
            or min(self.height, self.width) < 32
            or self.height % 32
            or self.width % 32
        ):
            raise ValueError(
                "Wan2.2 video windows require T=4k+1 (k>=1) and H/W multiples of 32"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = read_frames(
            row["video"],
            start=int(row.get("start_frame", 0)),
            stride=int(row.get("frame_stride", 1)),
            count=self.frames,
        )
        if row.get("condition_image"):
            with Image.open(row["condition_image"]) as condition:
                frames[0] = condition.convert("RGB")
        frames = [
            image_tensor(resize_image(frame, self.width, self.height))
            for frame in frames
        ]
        return {"video": torch.stack(frames, dim=1), "prompt": row["prompt"]}
