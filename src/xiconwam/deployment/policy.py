"""Batchless physical observations -> XiconWAM -> physical action chunks."""

from __future__ import annotations

import numpy as np
from xicon_engine import ActionChunk, Policy, PolicyInfo


class HoldPolicy(Policy):
    """Transport fixture only. Never represents checkpoint inference."""

    def __init__(self, info: PolicyInfo):
        if info.action_dim != info.state_dim:
            raise ValueError("hold fixture requires equal state/action widths")
        self._info = info

    @property
    def info(self):
        return self._info

    def reset(self):
        pass

    def infer(self, observation):
        observation.validate(self.info)
        return ActionChunk(
            np.repeat(observation.state[None], self.info.horizon, axis=0)
        )


class XiconWAMPolicy(Policy):
    def __init__(
        self,
        model,
        processor,
        info,
        *,
        num_video_frames,
        inference_steps=20,
        seed=42,
        text_context=None,
        override_instruction=None,
    ):
        from xiconwam.data.lerobot.transforms.relative_action import (
            RelativeJointTransform,
        )

        self._info = info
        self.model = model
        self.processor = processor.eval()
        self.num_video_frames = num_video_frames
        self.inference_steps = inference_steps
        self.seed = seed
        self.text_context = text_context
        self.override_instruction = override_instruction
        if inference_steps <= 0 or num_video_frames <= 1:
            raise ValueError("inference steps and video window must be positive")
        meta = processor.shape_meta
        if [tuple(m["shape"][-2:]) for m in meta["images"]] != [
            (256, 320),
            (128, 160),
            (128, 160),
        ]:
            raise ValueError(
                "initial deployment requires the RoboDojo three-camera shapes"
            )
        for group, width in (("state", info.state_dim), ("action", info.action_dim)):
            if sum(int(m["raw_shape"]) for m in meta[group]) != width:
                raise ValueError(
                    f"{group} contract width differs from training shape_meta"
                )
            if any(m["raw_shape"] != m["shape"] for m in meta[group]):
                raise ValueError(
                    "initial deployment supports joint vectors with unchanged raw shapes only"
                )
        if tuple(m["key"] for m in meta["images"]) != info.camera_keys:
            raise ValueError("camera order must match training shape_meta")
        for transform in processor.action_state_transforms or ():
            if not isinstance(transform, RelativeJointTransform):
                raise TypeError(
                    "initial deployment supports RelativeJointTransform only"
                )
        if info.representation != "absolute_joint_position":
            raise ValueError("deployment decodes absolute joint position targets only")
        if (
            len(info.action_names) != info.action_dim
            or len(info.action_units) != info.action_dim
        ):
            raise ValueError("deployment requires explicit per-axis names and units")

    @property
    def info(self):
        return self._info

    def reset(self):
        # Current actors infer each request independently; no retained episode history.
        pass

    def infer(self, observation):
        import torch

        observation.validate(self.info)
        processor = self.processor
        raw_state = {}
        offset = 0
        for meta in processor.shape_meta["state"]:
            width = int(meta["raw_shape"])
            raw_state[meta["key"]] = torch.from_numpy(
                observation.state[offset : offset + width].copy()
            )[None]
            offset += width
        state = {k: v.clone() for k, v in raw_state.items()}
        batch = processor.action_state_transform({"state": state})
        batch = processor.normalizer.forward(batch)
        proprio = processor.action_state_merger.forward(batch)["state"]
        cameras = []
        for key in self.info.camera_keys:
            image = torch.from_numpy(observation.images[key].copy()).permute(2, 0, 1)[
                None
            ]
            for transform in processor.val_transforms[key]:
                image = transform(image)
            cameras.append(image)
        # Match EpisodeWindowDataset's robotwin layout and [-1, 1] range.
        expected = [(256, 320), (128, 160), (128, 160)]
        if len(cameras) != 3 or [tuple(c.shape[-2:]) for c in cameras] != expected:
            raise ValueError(
                "initial deployment requires the RoboDojo robotwin three-camera layout"
            )
        image = (
            torch.cat((cameras[0], torch.cat(cameras[1:], dim=-1)), dim=-2)[0] * 2 - 1
        )
        from xiconwam.data.lerobot.prompts import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT.format(
            task=(
                self.override_instruction
                if self.override_instruction is not None
                else processor.augment_instruction({"task": observation.instruction})
            )
        )
        text = {"prompt": prompt}
        if self.text_context is not None:
            context = self.text_context(prompt)
            text = {
                "prompt": None,
                "context": context["context"],
                "context_mask": context["mask"],
            }
        with torch.inference_mode():
            output = self.model.infer_action(
                **text,
                input_image=image,
                action_horizon=self.info.horizon,
                num_video_frames=self.num_video_frames,
                proprio=proprio[0],
                num_inference_steps=self.inference_steps,
                seed=self.seed,
            )
        actions = output["action"].detach().cpu().float()
        if actions.shape != (self.info.horizon, processor.action_output_dim):
            raise ValueError(f"unexpected model action shape: {tuple(actions.shape)}")
        # Do not call WindowPacker.postprocess: it slices off num_obs_steps-1 rows.
        decoded = processor.action_state_merger.backward(
            {"action": actions[None], "state": proprio[None]}
        )
        decoded = processor.normalizer.backward(decoded)
        # Use the exact physical reference, even if normalization clipped the input.
        decoded["state"] = {k: v[None] for k, v in raw_state.items()}
        for transform in reversed(processor.action_state_transforms or ()):
            decoded = transform.backward(decoded)
        values = torch.cat(
            [decoded["action"][m["key"]] for m in processor.shape_meta["action"]],
            dim=-1,
        )[0].numpy()
        chunk = ActionChunk(values)
        chunk.validate(self.info)
        return chunk

    def close(self):
        self.model = None
