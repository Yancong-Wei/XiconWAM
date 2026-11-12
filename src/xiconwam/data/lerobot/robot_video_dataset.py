from pathlib import Path
from typing import Optional
import numpy as np
import errno
import time
from collections import Counter
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from .text_embedding_cache import (
    load_text_embedding_cache,
    prompt_hash,
    text_embedding_cache_filename,
)
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from .prompts import DEFAULT_PROMPT
from .resources import reuse_during_construction, shared_resource
from xiconwam.kit.logging_config import get_logger
from xiconwam.kit import misc, pytorch_utils
from accelerate import PartialState

logger = get_logger(__name__)


def configured_window_steps(config) -> int:
    """Raw control ticks, with support for saved configs using num_frames."""
    return int(config.get("window_steps", config.get("num_frames", 33)))


def trained_window(num_frames: int, action_video_freq_ratio: int) -> tuple[int, int]:
    """Video frames and action steps one training window teaches the policy.

    Frames are kept every ``action_video_freq_ratio`` ticks, so ``33`` and ``4``
    become the 9 frames ``0, 4, ..., 32``. Actions cover each tick in between,
    which is ``num_frames - 1``.
    """
    frames = int(num_frames)
    ratio = int(action_video_freq_ratio)
    if frames <= 1:
        raise ValueError(f"`num_frames` must be greater than 1, got {num_frames}.")
    if ratio <= 0:
        raise ValueError(
            "`action_video_freq_ratio` must be greater than 0, "
            f"got {action_video_freq_ratio}."
        )
    if (frames - 1) % ratio != 0:
        raise ValueError(
            "num_frames-1 must be divisible by action_video_freq_ratio, "
            f"got {frames - 1} and {ratio}"
        )
    if ((frames - 1) // ratio) % 4 != 0:
        raise ValueError(
            "video transitions must be divisible by 4 for tokenization, "
            f"got {(frames - 1) // ratio}"
        )
    return len(range(0, frames, ratio)), frames - 1


class EpisodeWindowDataset(torch.utils.data.Dataset):
    @reuse_during_construction
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        camera_layout=None,
        processor=None,
        text_embedding_cache_dir=None,
        text_encoder_id=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        mixed_precision: Optional[str] = None,
        load_text_encoder_on_miss: bool = False,
        preflight_text_cache: bool = True,
        max_read_retries: int = 2,
        backbone=None,
        unify_action_map=None,
        unify_state_map=None,
        norm_stats_output=None,
        window_steps=None,
        video_dit=None,
    ):
        if window_steps is not None:
            num_frames = int(window_steps)
        if video_dit is not None:
            if backbone is not None:
                raise ValueError("Specify dataset video_dit once (legacy: backbone)")
            backbone = video_dit
        del unify_action_map, unify_state_map
        _, action_steps = trained_window(num_frames, action_video_freq_ratio)
        self.video_sample_indices = tuple(
            range(0, int(num_frames), int(action_video_freq_ratio))
        )

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=action_steps,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_obs_indices=self.video_sample_indices,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        if camera_layout is None:
            self.camera_layout = [
                meta["key"]
                for meta in OmegaConf.to_container(shape_meta, resolve=True)["images"]
            ]
        else:
            self.camera_layout = [str(k) for k in camera_layout]
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser()
        )
        if text_encoder_id is None or not str(text_encoder_id).strip():
            raise ValueError(
                "`text_encoder_id` must be provided by the model configuration."
            )
        self.text_encoder_id = str(text_encoder_id)
        self.text_embedding_dtype = (
            None
            if mixed_precision is None
            else pytorch_utils.mixed_precision_to_dtype(mixed_precision)
        )
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.load_text_encoder_on_miss = bool(load_text_encoder_on_miss)
        self.is_training_set = bool(is_training_set)
        self.max_read_retries = int(max_read_retries)
        if self.max_read_retries < 0:
            raise ValueError("max_read_retries must be non-negative")
        self.read_error_counts = Counter()
        if isinstance(backbone, DictConfig):
            backbone = OmegaConf.to_container(backbone, resolve=True)
        self.backbone = backbone
        self._online_text_encoder = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            self.norm_stats_path = str(
                Path(
                    pretrained_norm_stats
                    if not is_training_set and pretrained_norm_stats
                    else norm_stats_output
                    or Path(misc.get_work_dir()) / "dataset_stats.json"
                )
                .expanduser()
                .resolve()
            )
            if is_training_set and PartialState().is_main_process:
                Path(self.norm_stats_path).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    stats_output = self.norm_stats_path
                    shared_resource(
                        ("saved_stats", stats_output, str(pretrained_norm_stats)),
                        lambda: save_dataset_stats_to_json(dataset_stats, stats_output),
                    )
                else:
                    dataset_stats = None
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if is_training_set and PartialState().is_main_process:
                    stats_output = self.norm_stats_path
                    shared_resource(
                        ("saved_stats", stats_output, str(pretrained_norm_stats)),
                        lambda: save_dataset_stats_to_json(dataset_stats, stats_output),
                    )

            processor.include_gt_action = False
            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
            if preflight_text_cache:
                self._preflight_text_cache()

    def _preflight_text_cache(self):
        """Validate the finite instruction set before the DataLoader starts."""
        processor = self.lerobot_dataset.processor
        if self.override_instruction is None and processor.drop_high_level_prob != 1.0:
            raise ValueError(
                "Text-cache preflight requires drop_high_level_prob=1; precompute deterministic instructions"
            )
        if self.override_instruction is not None:
            # _get uses an explicit override verbatim, including language markers.
            prompts = {DEFAULT_PROMPT.format(task=self.override_instruction)}
        else:
            tasks = {
                str(task)
                for dataset in self.lerobot_dataset.multi_dataset._datasets
                for task in dataset.meta.tasks.values()
            }
            prompts = {
                DEFAULT_PROMPT.format(
                    task=processor.augment_instruction({"task": task})
                )
                for task in tasks
            }

        def check():
            for prompt in sorted(prompts):
                filename = text_embedding_cache_filename(
                    prompt,
                    self.context_len,
                    self.text_encoder_id,
                    dtype=self.text_embedding_dtype,
                )
                path = (
                    None
                    if self.text_embedding_cache_dir is None
                    else self.text_embedding_cache_dir / filename
                )
                if self.load_text_encoder_on_miss and (
                    path is None or not path.is_file()
                ):
                    logger.warning(
                        "Text preflight: cache missing for prompt %s; explicit online fallback is enabled",
                        prompt_hash(prompt),
                    )
                    continue
                self._get_cached_text_context(prompt)
            return True

        shared_resource(
            (
                "text_preflight",
                str(self.text_embedding_cache_dir),
                self.context_len,
                self.text_encoder_id,
                str(self.text_embedding_dtype),
                self.load_text_encoder_on_miss,
                tuple(sorted(prompts)),
            ),
            check,
        )

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset.__getitem__(sample_idx)

            if not self.skip_padding_as_possible or not self.is_training_set:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = any(
                bool(pad.any().item())
                for pad in (action_is_pad, image_is_pad, proprio_is_pad)
            )

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        image_is_pad = sample["image_is_pad"]

        images = sample["pixel_values"]
        cameras = self._cameras_with_black_tiles(
            self.lerobot_dataset.processor.camera_views(images)
        )
        expected_video_frames = len(self.video_sample_indices)
        if any(camera.shape[0] != expected_video_frames for camera in cameras):
            raise ValueError(
                f"Sparse video frame count mismatch: expected {expected_video_frames}."
            )
        if image_is_pad.shape[0] != expected_video_frames:
            raise ValueError(
                f"Sparse image padding length mismatch: expected {expected_video_frames}, got {image_is_pad.shape[0]}."
            )
        if self.concat_multi_camera == "robotwin":
            if len(cameras) != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {len(cameras)}"
                )
            sizes = [(256, 320), (128, 160), (128, 160)]
            cameras = [
                (
                    camera
                    if tuple(camera.shape[-2:]) == size
                    else transforms_F.resize(
                        camera,
                        list(size),
                        interpolation=transforms_F.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                )
                for camera, size in zip(cameras, sizes)
            ]
            video = cameras[0].new_empty((expected_video_frames, 3, 384, 320))
            video.zero_()
            video[:, :, :256] = cameras[0]
            video[:, :, 256:, :160] = cameras[1]
            video[:, :, 256:, 160:] = cameras[2]
        elif len(cameras) > 1:
            if self.concat_multi_camera not in {"horizontal", "vertical"}:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}"
                )
            video = torch.cat(
                cameras, dim=-1 if self.concat_multi_camera == "horizontal" else -2
            )
        else:
            video = cameras[0]

        if tuple(video.shape[-2:]) != tuple(self.video_size):
            video = self.resize_transform(video)
            video = self.crop_transform(video)
        video = self.normalize_transform(video)

        video = video.permute(1, 0, 2, 3)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}"
            )
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]

        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        return data

    def _cameras_with_black_tiles(self, cameras):
        """Pad / replace missing layout slots with black tensors."""
        if isinstance(cameras, torch.Tensor):
            camera_list = list(cameras.unbind(0))
        else:
            camera_list = list(cameras)
        layout_n = len(self.camera_layout)
        shape_meta = self.lerobot_dataset.shape_meta
        image_meta = {meta["key"]: meta for meta in shape_meta["images"]}
        expected_frames = len(self.video_sample_indices)
        filled = []
        for index, key in enumerate(self.camera_layout):
            if index < len(camera_list) and camera_list[index] is not None:
                filled.append(camera_list[index])
                continue
            meta = image_meta.get(key)
            if meta is None:
                height, width = 128, 160
            else:
                height, width = int(meta["shape"][-2]), int(meta["shape"][-1])
            ref = next((c for c in camera_list if c is not None), None)
            dtype = ref.dtype if ref is not None else torch.float32
            device = ref.device if ref is not None else "cpu"
            filled.append(
                torch.zeros(
                    (expected_frames, 3, height, width),
                    dtype=dtype,
                    device=device,
                )
            )
        if len(filled) < layout_n:
            raise ValueError(
                f"camera_layout expects {layout_n} views, built {len(filled)}."
            )
        return filled

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            if not self.load_text_encoder_on_miss:
                raise ValueError("`text_embedding_cache_dir` is not set.")
            return self._encode_text_online(prompt)
        hashed = prompt_hash(prompt)
        dtype = self.text_embedding_dtype
        cache_path = self.text_embedding_cache_dir / text_embedding_cache_filename(
            hashed,
            self.context_len,
            self.text_encoder_id,
            is_hash=True,
            dtype=dtype,
        )
        if not cache_path.is_file():
            if not self.load_text_encoder_on_miss:
                raise FileNotFoundError(
                    f"Text embedding cache miss for prompt hash {hashed} at {cache_path}. "
                    "Set load_text_encoder_on_miss=true to encode online, or run "
                    "precompute_text_embeds.sh."
                )
            return self._encode_text_online(prompt)
        payload = load_text_embedding_cache(
            cache_path,
            self.context_len,
            self.text_encoder_id,
            hashed,
            expected_dtype=dtype,
        )
        context = payload["context"]
        context_mask = payload["mask"].bool()
        context[~context_mask] = 0
        return context, context_mask

    def _ensure_online_text_encoder(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError(
                "Online text encoding inside DataLoader workers is unsupported; precompute the text cache or use num_workers=0"
            )
        if self._online_text_encoder is not None:
            return self._online_text_encoder
        if self.backbone is None:
            raise ValueError(
                "Online text encoding requires `backbone` on the dataset "
                "(pass model.video_dit) when load_text_encoder_on_miss=true."
            )
        from xiconwam.vision.registry import load_text_encoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self.text_embedding_dtype or torch.bfloat16
        self._online_text_encoder = load_text_encoder(
            self.backbone,
            device=device,
            torch_dtype=dtype,
            context_len=self.context_len,
        )
        return self._online_text_encoder

    def _encode_text_online(self, prompt: str):
        encoder = self._ensure_online_text_encoder()
        dtype = self.text_embedding_dtype or torch.bfloat16
        with torch.no_grad():
            encoded = encoder.encode_batch([prompt])
        context = encoded.context[0].detach().to(device="cpu", dtype=dtype)
        context_mask = encoded.mask[0].detach().to(device="cpu", dtype=torch.bool)
        context[~context_mask] = 0
        return context, context_mask

    def __getitem__(self, idx):
        retries = self.max_read_retries if self.is_training_set else 0
        for attempt in range(retries + 1):
            try:
                return self._get(idx)
            except Exception as error:
                name = type(error).__name__
                self.read_error_counts[name] += 1
                transient = isinstance(error, OSError) and error.errno in {
                    errno.EIO,
                    errno.ETIMEDOUT,
                    errno.EAGAIN,
                    errno.ESTALE,
                }
                if transient and attempt < retries:
                    logger.warning(
                        "Transient sample read failure: index=%s attempt=%d error=%r counts=%s",
                        idx,
                        attempt + 1,
                        error,
                        dict(self.read_error_counts),
                    )
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Failed to read sample index={idx} training={self.is_training_set} "
                    f"datasets={self.lerobot_dataset.dataset_dirs} attempts={attempt + 1} "
                    f"error={name}: {error}"
                ) from error
