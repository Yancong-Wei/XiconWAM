from typing import Dict, Any, Optional, List
from collections.abc import Mapping

import torch
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf

from ..transforms.action_state_merger import DEFAULT_ACTION_DIM, ConcatLeftAlign
from ..utils.normalizer import LinearNormalizer, NormMode
from xiconwam.kit.pytorch_utils import dict_apply
from xiconwam.kit.logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)


def _mapping(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a mapping, got {type(value).__name__}.")


def _camera_ops(size, jitter: dict | None) -> list:
    """Resize each camera to its shape_meta size. Jitter is train-only."""
    from torchvision.transforms import ColorJitter, Resize

    from ..transforms.image import ToTensor

    height, width = int(size[0]), int(size[1])
    ops = [ToTensor(), Resize(size=[height, width])]
    if jitter:
        ops.append(
            ColorJitter(
                brightness=float(jitter.get("brightness", 0.0)),
                contrast=float(jitter.get("contrast", 0.0)),
                saturation=float(jitter.get("saturation", 0.0)),
                hue=float(jitter.get("hue", 0.0)),
            )
        )
    return ops


class WindowPacker(BaseProcessor):
    def camera_views(
        self, pixel_values: torch.Tensor | list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...] | list[torch.Tensor]:
        return pixel_values.unbind(0) if self.stack_images else pixel_values

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        action_output_dim: int = DEFAULT_ACTION_DIM,
        proprio_output_dim: int = DEFAULT_ACTION_DIM,
        action_state_transforms: Optional[List[Any]] = None,
        use_stepwise_action_norm: bool = False,
        norm_default_mode: NormMode = "z-score",
        norm_exception_mode: Dict[str, Dict[str, NormMode]] | None = None,
        action_state_merger=None,
        num_output_cameras: int | None = None,
        train_transforms: Dict[str, List[Any]] | None = None,
        val_transforms: Dict[str, List[Any]] | None = None,
        color_jitter: Optional[Mapping[str, float]] = None,
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,
        tokenizer: Optional[Any] = None,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
        stack_images: bool = True,
        include_gt_action: bool = True,
    ):
        self.stack_images = bool(stack_images)
        self.include_gt_action = bool(include_gt_action)
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        image_keys = [meta["key"] for meta in shape_meta["images"]]
        if num_output_cameras is None:
            num_output_cameras = len(image_keys)
        self.num_output_cameras = int(num_output_cameras)
        if train_transforms is None and val_transforms is None:
            # One resize per camera, taken from shape_meta, plus optional jitter.
            jitter = _mapping(color_jitter)
            self.train_transforms = {}
            self.val_transforms = {}
            for meta in shape_meta["images"]:
                key = meta["key"]
                size = meta["shape"][-2:]
                self.val_transforms[key] = _camera_ops(size, None)
                self.train_transforms[key] = _camera_ops(size, jitter)
        elif train_transforms is None or val_transforms is None:
            raise ValueError(
                "Pass both train_transforms and val_transforms, or neither."
            )
        else:
            self.train_transforms = (
                dict(train_transforms)
                if isinstance(train_transforms, Mapping)
                else dict.fromkeys(image_keys, train_transforms)
            )
            self.val_transforms = (
                dict(val_transforms)
                if isinstance(val_transforms, Mapping)
                else dict.fromkeys(image_keys, val_transforms)
            )

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        if action_state_merger is None:
            action_state_merger = ConcatLeftAlign()
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        self.tokenizer = tokenizer
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(
                    mask, dtype=torch.bool
                )

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError(
                "is_train has not been set. Please call train() and eval() first."
            )
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError(
                "normalizer has not been set. Please call set_normalizer_from_stats() first."
            )
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )

    def augment_instruction(self, data: Dict[str, str]) -> str:
        """Choose the instruction language and optional high-level task."""
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = low_level_instruction
        else:
            instruction = (
                f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"
            )

        return instruction

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert (
                    actual_shape == meta_shape
                ), f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert (
                actual_shape == meta_shape
            ), f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        if self.action_state_transforms is not None:
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)

        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert (
                    actual_shape == meta_shape
                ), f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert (
                actual_shape == meta_shape
            ), f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = {}
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]
        if sample["image_is_pad"].ndim != 1:
            raise ValueError(
                "`image_is_pad` must be one-dimensional, "
                f"got shape {tuple(sample['image_is_pad'].shape)}."
            )
        num_image_steps = int(sample["image_is_pad"].shape[0])

        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]
            assert (
                image.ndim == 4
            ), f"Expected 4 dimensions (num_image_steps, C, H, W), got shape {image.shape}"

            transforms = self.train_transforms if self.is_train else self.val_transforms
            for trans in transforms[key]:
                image = trans(image)

            meta_shape = [num_image_steps] + shape
            assert (
                image.shape == meta_shape
            ), f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"

            processed_images.append(image)
        if self.num_output_cameras != len(processed_images):
            if self.num_output_cameras < len(processed_images):
                processed_images = processed_images[: self.num_output_cameras]
            else:
                processed_images.extend(
                    torch.zeros_like(processed_images[0])
                    for _ in range(self.num_output_cameras - len(processed_images))
                )
        sample["pixel_values"] = (
            torch.stack(processed_images, dim=0)
            if self.stack_images
            else processed_images
        )

        if self.include_gt_action and not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            for key, dim_mask in self.delta_action_dim_mask.items():
                cur_action = data["action"][key]
                cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                cur_dim_mask = dim_mask.to(device=cur_action.device)
                pad_delta_mask = cur_action_is_pad.unsqueeze(
                    1
                ) & cur_dim_mask.unsqueeze(0)
                cur_action[pad_delta_mask] = 0.0
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"]
            sample["action_is_pad"] = data["action_is_pad"]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]
            assert sample["action"].shape[-1] == self.action_output_dim
        sample["proprio"] = data["state"]
        sample["proprio_is_pad"] = data["state_is_pad"]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data
