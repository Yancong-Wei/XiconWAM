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

from xiconwam.actor.objective import ObjectiveMixin
from xiconwam.actor.rollout import RolloutMixin


class XiconActor(ObjectiveMixin, RolloutMixin, torch.nn.Module):
    _forward_endpoint = endpoint.predict

    _forward_endpoint_cached_action = endpoint._cached_action

    model_variant = "xicon"

    inference_compile_targets = (
        "_predict_video_noise",
        "_predict_action_noise_with_cache",
        "_predict_joint_noise",
    )

    def __init__(
        self,
        video_expert,
        action_expert: ActionExpert,
        streams: StreamMixer,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        state_position: str = "context",
        projector_hidden_dim: int = 64,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.inference_cross_kv_reuse = False
        self.video_expert = video_expert
        self.backbone_name = video_expert.backbone_name
        if getattr(video_expert, "visual_target", None) == "endpoint":
            self.inference_compile_targets = (
                "_forward_endpoint",
                "_forward_endpoint_cached_action",
            )
        self.action_expert = action_expert
        self.streams = streams
        self.dit = self.streams

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.state_position = normalize_state_position(state_position)
        self.projector_hidden_dim = int(projector_hidden_dim)
        if self.state_dim is not None:
            state_output_dim = (
                self.text_dim
                if self.state_position == "context"
                else int(self.action_expert.hidden_dim)
            )
            self.state_encoder = StateEncoder(
                state_dim=self.state_dim,
                hidden_dim=state_output_dim,
                projector_hidden_dim=self.projector_hidden_dim,
            ).to(torch_dtype)
        else:
            self.state_encoder = None

        self.train_video_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(device=self.device, dtype=self.torch_dtype)

    @property
    def video_dit(self):
        return self.video_expert

    def optimizer_param_groups(self, learning_rate: float, weight_decay: float):
        from xiconwam.engine.optimization import policy_optimizer_param_groups

        return policy_optimizer_param_groups(self, learning_rate, weight_decay)

    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone: dict[str, Any],
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        state_dim: Optional[int] = None,
        state_position: str = "context",
        projector_hidden_dim: int = 64,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        **model_init_kwargs: Any,
    ):
        from xiconwam.vision.registry import load_components, normalize_spec

        cfg = normalize_spec(backbone)
        components = load_components(
            cfg,
            device=device,
            torch_dtype=torch_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        video_expert = components.dit
        action_expert_cls = ActionExpert
        if components.block_protocol == BLOCK_PROTOCOL_STAGED:
            if action_dit_config.get("preset") != "staged":
                raise ValueError(
                    "Staged mixed attention requires ActionExpert preset=staged"
                )
            action_dit_config = dict(action_dit_config)
            for key, expected in dict(
                num_heads=video_expert.num_heads,
                attn_head_dim=video_expert.attn_head_dim,
                num_layers_double=video_expert.double_layers,
                num_layers_single=video_expert.single_layers,
                attention_backend=video_expert.attention_backend,
            ).items():
                if key in action_dit_config and action_dit_config[key] != expected:
                    raise ValueError(
                        f"StreamMixer ActionExpert {key} must match backbone: {expected}"
                    )
                action_dit_config.setdefault(key, expected)
        action_expert = action_expert_cls.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionExpert num_heads must match the video expert.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionExpert attn_head_dim must match the video expert.")
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("ActionExpert num_layers must match the video expert.")
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            streams=StreamMixer({"video": video_expert, "action": action_expert}),
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(components.text_dim),
            state_dim=state_dim,
            state_position=state_position,
            projector_hidden_dim=projector_hidden_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            **model_init_kwargs,
        )
        model.backbone_name = cfg["name"]
        from xiconwam.actor.flow.factory import build_video_schedulers

        model.train_video_scheduler, model.infer_video_scheduler = (
            build_video_schedulers(
                family=components.dit.scheduler_family,
                config=cfg.get("video_scheduler"),
                num_train_timesteps=video_num_train_timesteps,
                train_shift=video_train_shift,
                infer_shift=video_infer_shift,
            )
        )
        model.train_scheduler = model.train_video_scheduler
        model.infer_scheduler = model.infer_video_scheduler
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": action_dit_pretrained_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.streams.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        if self.tokenizer is None:
            context, mask = self.text_encoder(prompt)
            return context.to(device=self.device), mask.to(
                device=self.device, dtype=torch.bool
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        return prompt_emb.to(device=self.device), mask

    def _append_state_to_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self.state_position != "context"
            or self.state_encoder is None
            or state is None
        ):
            return context, context_mask
        if state.ndim != 2:
            raise ValueError(
                f"`state` must be 2D [B, D], got shape {tuple(state.shape)}"
            )
        if self.state_dim is None or state.shape[1] != self.state_dim:
            raise ValueError(
                f"`state` last dim must be {self.state_dim}, got {state.shape[1]}"
            )
        state_token = self.state_encoder(
            state.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        if context_mask is None:
            return torch.cat([context, state_token], dim=1), None
        state_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, state_token], dim=1),
            torch.cat([context_mask, state_mask], dim=1),
        )

    def _append_state_to_action_sequence(
        self,
        action_pre: dict[str, Any],
        state: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if self.state_position != "sequence" or self.state_encoder is None:
            return action_pre
        if state is None:
            return action_pre
        if state.ndim != 2:
            raise ValueError(
                f"`state` must be 2D [B,D], got shape {tuple(state.shape)}"
            )
        if self.state_dim is None or state.shape[1] != self.state_dim:
            raise ValueError(
                f"`state` last dim must be {self.state_dim}, got {state.shape[1]}"
            )
        state_tokens = self.state_encoder(
            state.to(device=self.device, dtype=action_pre["tokens"].dtype).unsqueeze(1)
        ).to(dtype=action_pre["tokens"].dtype)
        return self.action_expert.append_state_tokens(action_pre, state_tokens)

    def _state_sequence_length(self, state: Optional[torch.Tensor]) -> int:
        return int(
            self.state_position == "sequence"
            and self.state_encoder is not None
            and state is not None
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor):
        if endpoint.enabled(self):
            return endpoint.encode_video(self, video_tensor)
        z = self.vae.encode(video_tensor, device=self.device)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor):
        return encode_image_batch(self.vae, input_image, self.device)

    def _decode_latents(self, latents):
        if endpoint.enabled(self):
            return endpoint.decode_latents(self, latents)
        return decode_video_batch(self.vae, latents, self.device)

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        return timestep / float(scheduler.num_train_timesteps)

    def forward(self, sample):
        return self.training_loss(sample)

    def save_checkpoint(self, path, optimizer=None, step=None):
        from xiconwam.actor.lora import build_lora_checkpoint_payload, has_lora

        if has_lora(self.video_expert):
            payload = build_lora_checkpoint_payload(self, step=step)
            payload["state_position"] = self.state_position
            payload["action_config"] = policy_action_config(self)
            if optimizer is not None:
                payload["optimizer"] = optimizer.state_dict()
            torch.save(payload, path)
            return
        payload = {
            "streams": self.streams.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "backbone_name": self.backbone_name,
            "model_variant": self.model_variant,
        }
        if self.state_encoder is not None:
            payload["state_encoder"] = self.state_encoder.state_dict()
        payload["state_position"] = self.state_position
        payload["action_config"] = policy_action_config(self)
        if endpoint.enabled(self):
            payload["structure"] = endpoint.structure(self)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, merge_lora: bool = False):
        from xiconwam.actor.lora import (
            is_lora_checkpoint,
            load_lora_model_checkpoint_state,
        )

        payload = torch.load(path, map_location="cpu", mmap=True)
        if is_lora_checkpoint(payload):
            load_lora_model_checkpoint_state(
                self, payload, merge_after_load=bool(merge_lora)
            )
            if optimizer is not None and "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            return payload
        validate_checkpoint_state_position(payload, self.state_position)
        validate_policy_action_config(self, payload)
        if endpoint.enabled(self):
            endpoint.validate_structure(self, payload)
        checkpoint_backbone = payload.get("backbone_name")
        if (
            checkpoint_backbone is not None
            and checkpoint_backbone != self.backbone_name
        ):
            raise ValueError(
                f"Checkpoint backbone {checkpoint_backbone!r} does not match model backbone "
                f"{self.backbone_name!r}."
            )
        checkpoint_variant = payload.get("model_variant")
        if checkpoint_variant not in (None, self.model_variant):
            raise ValueError(
                f"Checkpoint variant {checkpoint_variant!r} does not match "
                f"{self.model_variant!r}."
            )
        weights = payload.get("streams")
        if weights is None and "dit" in payload:
            logger.warning("Loading a video-only checkpoint into the video expert.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        elif weights is None:
            raise ValueError(f"Checkpoint missing stream weights: {path}")
        else:
            self.streams.load_state_dict(weights, strict=False)
        if self.state_encoder is not None:
            if "state_encoder" in payload:
                self.state_encoder.load_state_dict(
                    payload["state_encoder"], strict=True
                )
            else:
                logger.warning(
                    "Checkpoint has no state encoder weights; keeping the current parameters."
                )
        elif "state_encoder" in payload:
            logger.warning(
                "Checkpoint contains state encoder weights, but this actor has no state encoder."
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
