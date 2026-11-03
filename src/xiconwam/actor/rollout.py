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


class RolloutMixin:
    @torch.no_grad()
    def _prepare_inference_cross_attention(
        self,
        context: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
        Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    ]:
        video_context = self.video_expert.project_context(context)
        action_context = (
            self.action_expert.project_context(context)
            if self.action_expert.cross_attention
            else None
        )
        if not self.inference_cross_kv_reuse:
            return video_context, action_context, None, None
        return (
            video_context,
            action_context,
            self.video_expert.build_cross_attention_kv_cache(video_context),
            (
                self.action_expert.build_cross_attention_kv_cache(action_context)
                if self.action_expert.cross_attention
                else None
            ),
        )

    @torch.no_grad()
    def _predict_video_noise(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        projected_context: Optional[torch.Tensor] = None,
        cross_kv_cache: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> torch.Tensor:
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=latents_video,
            timestep=timestep_video,
            context=context if projected_context is None else projected_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            context_is_projected=projected_context is not None,
            cross_kv_cache=cross_kv_cache,
        )
        video_tokens = video_pre["tokens"]
        attention_mask = None
        if self.video_expert.video_attention_mask_mode != "bidirectional":
            attention_mask = self.video_expert.build_structured_video_attention_mask(
                video_seq_len=video_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_tokens.device,
            )
        for layer_index in range(len(self.video_expert.blocks)):
            video_tokens = self.video_expert.forward_block(
                layer_index,
                video_tokens,
                video_pre,
                attention_mask,
            )
        return self.video_expert.post_dit(video_tokens, video_pre)

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        video_cross_kv_cache: Optional[
            tuple[tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        action_cross_kv_cache: Optional[
            tuple[tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=latents_video,
            timestep=timestep_video,
            context=context if video_context is None else video_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            context_is_projected=video_context is not None,
            cross_kv_cache=video_cross_kv_cache,
        )
        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context if action_context is None else action_context,
            context_mask=context_mask,
            context_is_projected=action_context is not None,
            cross_kv_cache=action_cross_kv_cache,
        )
        action_pre = self._append_state_to_action_sequence(action_pre, state)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.streams(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
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
                    "kv_cache": video_pre["cross_kv_cache"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                    "kv_cache": action_pre["cross_kv_cache"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(
            timestep_action, dtype=first_frame_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._append_state_to_action_sequence(action_pre, state)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.streams(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
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
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor | StructuredAttentionMask,
        video_seq_len: int,
        action_context: Optional[torch.Tensor] = None,
        action_cross_kv_cache: Optional[
            tuple[tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context if action_context is None else action_context,
            context_mask=context_mask,
            context_is_projected=action_context is not None,
            cross_kv_cache=action_cross_kv_cache,
        )
        action_pre = self._append_state_to_action_sequence(action_pre, state)
        action_tokens = self.streams.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
                "kv_cache": action_pre["cross_kv_cache"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _action_only_inference_step_count(
        self,
        num_inference_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sigma_shift: Optional[float],
    ) -> int:
        video_timesteps, _ = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        return len(video_timesteps)

    @torch.inference_mode()
    def infer_joint_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[Union[int, Sequence[Optional[int]]]] = None,
        rand_device: str = "cpu",
        test_action_with_infer_action: bool = False,
        decode_video: bool = True,
    ) -> JointBatchResult:
        if endpoint.enabled(self):
            return endpoint.infer(
                self,
                action_only=False,
                **{k: v for k, v in locals().items() if k != "self"},
            )
        validate_batch_inputs(input_image, prompt, context, context_mask)
        self.eval()
        if test_action_with_infer_action:
            raise ValueError(
                "Batch inference does not support `test_action_with_infer_action`."
            )

        batch_size, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if (
                action.ndim != 3
                or action.shape[0] != batch_size
                or action.shape[1] != action_horizon
            ):
                raise ValueError(
                    f"`action` must have shape [B,T,D], got {tuple(action.shape)} with B={batch_size} and T={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.state_dim is None:
                raise ValueError(
                    "`proprio` was provided but `state_dim=None` so `state_encoder` is disabled."
                )
            if proprio.ndim != 2 or proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` must be [B,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.state_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        latents_video = randn_per_sample(
            (self.vae.model.z_dim, latent_t, latent_h, latent_w),
            seeds=seed,
            batch_size=batch_size,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = randn_per_sample(
            (action_horizon, self.action_expert.action_dim),
            seeds=seed,
            batch_size=batch_size,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(self.video_expert.fuse_vae_embedding_in_latents)

        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )
        context_mask = prepare_key_padding_mask(context_mask)
        video_context, action_context, video_cross_kv_cache, action_cross_kv_cache = (
            self._prepare_inference_cross_attention(context)
        )

        infer_timesteps_video, infer_deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                # Align action steps with the effective visual inference schedule.
                num_inference_steps=len(infer_timesteps_video),
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.expand(batch_size).to(
                dtype=latents_video.dtype, device=self.device
            )
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                video_context=video_context,
                action_context=action_context,
                video_cross_kv_cache=video_cross_kv_cache,
                action_cross_kv_cache=action_cross_kv_cache,
                state=proprio,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        result = {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32)
        }
        if decode_video:
            result["video"] = self._decode_latents(latents_video)
        return result

    @torch.inference_mode()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        test_action_with_infer_action: bool = False,
        decode_video: bool = True,
    ) -> JointInferenceResult:
        input_image = batch_single_tensor(input_image, 3, "input_image")
        proprio = batch_single_state(proprio, history=False)
        context = batch_single_tensor(context, 2, "context")
        context_mask = batch_single_tensor(context_mask, 1, "context_mask")
        action = batch_single_tensor(action, 2, "action")
        result = self.infer_joint_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            test_action_with_infer_action=False,
            decode_video=decode_video,
        )
        result["action"] = result["action"][0]
        if "video" in result:
            result["video"] = result["video"][0]
        return result

    @torch.inference_mode()
    def infer_action_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[Union[int, Sequence[Optional[int]]]] = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> ActionBatchResult:
        if endpoint.enabled(self):
            return endpoint.infer(
                self,
                action_only=True,
                **{k: v for k, v in locals().items() if k != "self"},
            )
        validate_batch_inputs(input_image, prompt, context, context_mask)
        self.eval()
        attention_mode = self.video_expert.video_attention_mask_mode

        if attention_mode == "bidirectional":
            joint_out = self.infer_joint_batch(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                test_action_with_infer_action=False,
                decode_video=False,
            )
            return {"action": joint_out["action"]}
        if attention_mode != "first_frame_causal":
            raise ValueError(
                "`infer_action` supports `video_attention_mask_mode` values "
                "'first_frame_causal' and 'bidirectional', "
                f"got {attention_mode!r}."
            )

        batch_size, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.state_dim is None:
                raise ValueError(
                    "`proprio` was provided but `state_dim=None` so `state_encoder` is disabled."
                )
            if proprio.ndim != 2 or proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` must be [B,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.state_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latents_action = randn_per_sample(
            (action_horizon, self.action_expert.action_dim),
            seeds=seed,
            batch_size=batch_size,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image
        )
        fuse_flag = bool(self.video_expert.fuse_vae_embedding_in_latents)

        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )
        context_mask = prepare_key_padding_mask(context_mask)
        video_context, action_context, _, action_cross_kv_cache = (
            self._prepare_inference_cross_attention(context)
        )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=first_frame_latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            context_is_projected=True,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=(
                latents_action.shape[1] + self._state_sequence_length(proprio)
            ),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.streams.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask.slice(
                0, video_seq_len, 0, video_seq_len
            ),
        )

        action_inference_steps = self._action_only_inference_step_count(
            num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            sigma_shift=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=action_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(
            infer_timesteps_action, infer_deltas_action
        ):
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_context=action_context,
                action_cross_kv_cache=action_cross_kv_cache,
                state=proprio,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        return {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.inference_mode()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> ActionInferenceResult:
        input_image = batch_single_tensor(input_image, 3, "input_image")
        proprio = batch_single_state(proprio, history=False)
        context = batch_single_tensor(context, 2, "context")
        context_mask = batch_single_tensor(context_mask, 1, "context_mask")
        result = self.infer_action_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            num_video_frames=num_video_frames,
        )
        return {"action": result["action"][0]}

    @torch.inference_mode()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
