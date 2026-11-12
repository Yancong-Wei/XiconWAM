from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from xiconwam.actor.action_expert import (
    ActionExpert,
    StateEncoder,
    normalize_state_position,
    policy_action_config,
    validate_checkpoint_state_position,
    validate_policy_action_config,
)
from xiconwam.actor.attention import (
    AttentionSegment,
    StructuredAttentionMask,
    build_structured_attention_mask,
    elide_fully_valid_attention_mask,
    prepare_key_padding_mask,
)
from xiconwam.actor.flow.scheduler_continuous import ContinuousFlowMatchScheduler
from xiconwam.actor.helpers import endpoint
from xiconwam.actor.helpers.action_mask import masked_action_mse_per_sample
from xiconwam.actor.helpers.batching import (
    ActionBatchResult,
    ActionInferenceResult,
    JointBatchResult,
    JointInferenceResult,
    batch_single_state,
    batch_single_tensor,
    decode_video_batch,
    encode_image_batch,
    randn_per_sample,
    validate_batch_inputs,
)
from xiconwam.actor.helpers.static_cache import cache_static_method
from xiconwam.actor.mixture import StreamMixer
from xiconwam.kit.logging_config import get_logger
from xiconwam.vision.protocol import BLOCK_PROTOCOL_STAGED

logger = get_logger(__name__)


class ObjectiveMixin:
    def build_inputs(self, sample):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "Xicon training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
            )

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for Xicon training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if (
                action_is_pad.shape[0] != batch_size
                or action_is_pad.shape[1] != action_horizon
            ):
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim == 1:
                action_dim_is_pad = action_dim_is_pad.unsqueeze(0).expand(
                    batch_size, -1
                )
            if action_dim_is_pad.ndim != 2:
                raise ValueError(
                    "`sample['action_dim_is_pad']` must be 1D [D] or 2D [B, D], "
                    f"got shape {tuple(action_dim_is_pad.shape)}"
                )
            if (
                action_dim_is_pad.shape[0] != batch_size
                or action_dim_is_pad.shape[1] != action.shape[2]
            ):
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected "
                    f"({batch_size}, {action.shape[2]})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if (
                image_is_pad.shape[0] != batch_size
                or image_is_pad.shape[1] != num_frames
            ):
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        input_latents = self._encode_video_latents(input_video)

        first_frame_latents = None
        fuse_flag = False
        if self.video_expert.fuse_vae_embedding_in_latents:
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context_mask = elide_fully_valid_attention_mask(context_mask)
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if context_mask is not None:
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        state = None
        if self.state_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `state_dim` is enabled."
                )
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.state_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.state_dim}, got {proprio.shape[2]}"
                )
            state = proprio[:, 0, :]
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=state.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        return {
            "context": context,
            "context_mask": context_mask,
            "state": state,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    @cache_static_method("video_expert.video_attention_mask_mode")
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> StructuredAttentionMask:
        total_seq_len = video_seq_len + action_seq_len
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mode = self.video_expert.video_attention_mask_mode
        segments = []

        if mode == "bidirectional":
            segments.append(AttentionSegment(0, video_seq_len, ((0, video_seq_len),)))
        elif mode == "first_frame_causal":
            segments.append(
                AttentionSegment(0, first_frame_tokens, ((0, first_frame_tokens),))
            )
            if first_frame_tokens < video_seq_len:
                segments.append(
                    AttentionSegment(
                        first_frame_tokens, video_seq_len, ((0, video_seq_len),)
                    )
                )
        elif mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in per_frame_causal mode."
                )
            for frame_start in range(0, video_seq_len, video_tokens_per_frame):
                frame_end = frame_start + video_tokens_per_frame
                segments.append(
                    AttentionSegment(frame_start, frame_end, ((0, frame_end),))
                )
        else:
            raise ValueError(f"Unsupported video attention mask mode: {mode}")

        if action_seq_len > 0:
            if first_frame_tokens == video_seq_len:
                action_key_ranges = ((0, total_seq_len),)
            else:
                action_key_ranges = (
                    (0, first_frame_tokens),
                    (video_seq_len, total_seq_len),
                )
            segments.append(
                AttentionSegment(video_seq_len, total_seq_len, action_key_ranges)
            )
        return build_structured_attention_mask(
            query_len=total_seq_len,
            key_len=total_seq_len,
            segments=segments,
            device=device,
        )

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample):
        if endpoint.enabled(self):
            return endpoint.training_loss(self, sample)
        inputs = self.build_inputs(sample)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_dim_is_pad = inputs.get("action_dim_is_pad")
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._append_state_to_action_sequence(
            action_pre, inputs.get("state")
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.streams(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        if self.loss_lambda_video == 0.0:
            loss_video = pred_action.new_zeros((), dtype=torch.float32)
        else:
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]

            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
            )
            loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = masked_action_mse_per_sample(
            pred_action,
            target_action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=action_dim_is_pad,
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_action * loss_action + self.loss_lambda_video * loss_video
        )
        loss_dict = {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "weighted_loss_action": loss_action.detach() * self.loss_lambda_action,
            "weighted_loss_video": loss_video.detach() * self.loss_lambda_video,
        }
        return loss_total, loss_dict
