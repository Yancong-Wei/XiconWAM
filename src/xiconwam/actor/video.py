"""Standalone visual training using the selected video backbone."""

import torch

from xiconwam.actor.flow.factory import build_video_schedulers
from xiconwam.actor.helpers import endpoint
from xiconwam.actor.helpers.batching import decode_video_batch
from xiconwam.actor.helpers.gradient import gradient_checkpoint_forward
from xiconwam.actor.objective import ObjectiveMixin
from xiconwam.actor.world import XiconActor
from xiconwam.vision.registry import load_components


class VideoModel(torch.nn.Module):
    video_only = True
    weights_suffix = ".safetensors"

    encode_prompt = XiconActor.encode_prompt
    _compute_video_loss_per_sample = ObjectiveMixin._compute_video_loss_per_sample

    def __init__(self, components, scheduler_config, *, device, torch_dtype):
        super().__init__()
        self.dit = components.dit
        self.vae = components.vae.requires_grad_(False).eval()
        self.text_encoder = components.text_encoder
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False).eval()
        self.tokenizer = components.tokenizer
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        scheduler_config = dict(scheduler_config or {})
        self.train_scheduler, self.infer_scheduler = build_video_schedulers(
            family=self.dit.scheduler_family,
            config=scheduler_config,
            num_train_timesteps=int(scheduler_config.get("num_train_timesteps", 1000)),
            train_shift=float(scheduler_config.get("train_shift", 5.0)),
            infer_shift=float(scheduler_config.get("infer_shift", 5.0)),
        )

    @classmethod
    def from_backbone(
        cls, backbone, *, device, torch_dtype, skip_dit_load_from_pretrain=False
    ):
        components = load_components(
            backbone,
            device=device,
            torch_dtype=torch_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        return cls(
            components,
            backbone.get("video_scheduler"),
            device=device,
            torch_dtype=torch_dtype,
        )

    @property
    def video_dit(self):
        return self.dit

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def save_checkpoint(self, path, optimizer=None, step=None):
        from xiconwam.actor.dit_checkpoint import save_video_weights

        save_video_weights(self.dit, path, step=step)

    def load_checkpoint(self, path, optimizer=None):
        from xiconwam.actor.dit_checkpoint import load_video_weights

        load_video_weights(self.dit, path)

    def _validate_shape(self, frames, height, width):
        if endpoint.enabled(self):
            if frames < 2 or height % 16 or width % 16:
                raise ValueError(
                    "Flux endpoint training needs two frames and H/W multiples of 16"
                )
            return
        temporal = self.vae.temporal_downsample_factor
        spatial = self.vae.upsampling_factor
        _, patch_h, patch_w = self.dit.patch_size
        if frames <= 1 or (frames - 1) % temporal:
            raise ValueError(f"Video needs T > 1 and (T - 1) divisible by {temporal}")
        if (
            min(height, width) <= 0
            or height % (spatial * patch_h)
            or width % (spatial * patch_w)
        ):
            raise ValueError(
                f"Video height/width must be multiples of {spatial * patch_h}/{spatial * patch_w}"
            )

    def _video_prediction(self, latents, timestep, context, mask):
        pre = self.dit.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=mask,
            fuse_vae_embedding_in_latents=True,
        )
        tokens = pre["tokens"]
        attention_mask = None
        if self.dit.video_attention_mask_mode != "bidirectional":
            attention_mask = self.dit.build_structured_video_attention_mask(
                tokens.shape[1], pre["meta"]["tokens_per_frame"], tokens.device
            )
        for index in range(len(self.dit.blocks)):
            tokens = gradient_checkpoint_forward(
                lambda value, i=index: self.dit.forward_block(
                    i, value, pre, attention_mask
                ),
                self.dit.use_gradient_checkpointing and torch.is_grad_enabled(),
                tokens,
            )
        return self.dit.post_dit(tokens, pre)

    def _endpoint_prediction(self, target, reference, timestep, context, mask, grid):
        pre = endpoint._scene(self, target, reference, context, mask, timestep, grid)
        return endpoint._visual(self, pre)[0]

    def forward(self, sample):
        return self.training_loss(sample)

    def training_loss(self, sample):
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("Expected video [B,3,T,H,W]")
        self._validate_shape(*video.shape[2:])
        with torch.no_grad():
            context, mask = sample.get("context"), sample.get("context_mask")
            if (context is None) != (mask is None):
                raise ValueError("Provide both context and context_mask")
            if context is None:
                context, mask = self.encode_prompt(sample["prompt"])
            context = context.to(device=self.device, dtype=self.torch_dtype)
            mask = mask.to(device=self.device, dtype=torch.bool)
            clean = (
                endpoint.encode_video(self, video)
                if endpoint.enabled(self)
                else self.vae.encode(video, device=self.device)
            ).to(dtype=self.torch_dtype)
        if endpoint.enabled(self):
            return self._endpoint_loss(clean, context, mask, sample.get("image_is_pad"))

        scheduler = self.train_scheduler
        noise = torch.randn_like(clean)
        timestep = scheduler.sample_training_t(len(clean), self.device, clean.dtype)
        latents = scheduler.add_noise(clean, noise, timestep)
        latents[:, :, :1] = clean[:, :, :1]
        target = scheduler.training_target(clean, noise, timestep)
        predicted = self._video_prediction(latents, timestep, context, mask)
        per_sample = self._compute_video_loss_per_sample(
            predicted[:, :, 1:],
            target[:, :, 1:],
            sample.get("image_is_pad"),
            include_initial_video_step=False,
        )
        loss = (per_sample * scheduler.training_weight(timestep)).mean()
        return loss, {"loss_video": loss.detach(), "weighted_loss_video": loss.detach()}

    def _endpoint_loss(self, clean, context, mask, image_is_pad):
        reference = clean[:, :, 0].flatten(2).transpose(1, 2)
        target = clean[:, :, -1].flatten(2).transpose(1, 2)
        noise = torch.randn_like(target)
        timestep = self.train_scheduler.sample_training_t(
            len(clean), self.device, target.dtype
        )
        noisy = self.train_scheduler.add_noise(target, noise, timestep)
        prediction = self._endpoint_prediction(
            noisy, reference, timestep, context, mask, clean.shape[-2:]
        )
        error = (prediction.float() - (noise - target).float()).square()
        per_sample = error.mean((1, 2))
        if image_is_pad is not None:
            valid = (~image_is_pad[:, -1]).to(self.device, per_sample.dtype)
            loss = (
                per_sample * self.train_scheduler.training_weight(timestep) * valid
            ).sum() / valid.sum().clamp_min(1)
        else:
            loss = (per_sample * self.train_scheduler.training_weight(timestep)).mean()
        return loss, {"loss_video": loss.detach(), "weighted_loss_video": loss.detach()}

    @torch.inference_mode()
    def infer_video(
        self, image, prompt, *, frames, steps, seed, guidance=1.0, negative_prompt=""
    ):
        if endpoint.enabled(self):
            if frames != 2:
                raise ValueError("Flux predicts an endpoint pair. Set frames=2.")
            return self._infer_endpoint(
                image, prompt, steps, seed, guidance, negative_prompt
            )
        height, width = image.shape[-2:]
        self._validate_shape(frames, height, width)
        context, mask = self.encode_prompt([prompt])
        if guidance != 1:
            negative_context, negative_mask = self.encode_prompt([negative_prompt])
        image = image[None, :, None].to(device=self.device, dtype=self.torch_dtype)
        anchor = self.vae.encode(image, device=self.device).to(dtype=self.torch_dtype)
        latent_frames = (frames - 1) // self.vae.temporal_downsample_factor + 1
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (*anchor.shape[:2], latent_frames, *anchor.shape[-2:]),
            device=self.device,
            dtype=self.torch_dtype,
            generator=generator,
        )
        latents[:, :, :1] = anchor
        scheduler = self.infer_scheduler
        timesteps, deltas = scheduler.build_inference_schedule(
            steps, self.device, self.torch_dtype
        )
        for timestep, delta in zip(timesteps, deltas):
            prediction = self._video_prediction(latents, timestep[None], context, mask)
            if guidance != 1:
                unconditional = self._video_prediction(
                    latents, timestep[None], negative_context, negative_mask
                )
                prediction = unconditional + guidance * (prediction - unconditional)
            latents = scheduler.step(prediction, delta, latents, stream_id="video")
            latents[:, :, :1] = anchor
        if not torch.isfinite(latents).all():
            raise RuntimeError("Video denoising produced non-finite latents")
        return decode_video_batch(self.vae, latents, self.device)[0]

    @torch.inference_mode()
    def _infer_endpoint(self, image, prompt, steps, seed, guidance, negative_prompt):
        self._validate_shape(2, *image.shape[-2:])
        context, mask = self.encode_prompt([prompt])
        if guidance != 1:
            negative_context, negative_mask = self.encode_prompt([negative_prompt])
        clean = endpoint.encode_video(self, image[None, :, None])
        reference = clean[:, :, 0].flatten(2).transpose(1, 2)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        target = torch.randn(
            reference.shape,
            device=self.device,
            dtype=self.torch_dtype,
            generator=generator,
        )
        timesteps, deltas = self.infer_scheduler.build_inference_schedule(
            steps, self.device, self.torch_dtype
        )
        for timestep, delta in zip(timesteps, deltas):
            prediction = self._endpoint_prediction(
                target, reference, timestep[None], context, mask, clean.shape[-2:]
            )
            if guidance != 1:
                negative = self._endpoint_prediction(
                    target,
                    reference,
                    timestep[None],
                    negative_context,
                    negative_mask,
                    clean.shape[-2:],
                )
                prediction = negative + guidance * (prediction - negative)
            target = self.infer_scheduler.step(
                prediction, delta, target, stream_id="video"
            )
        future = target.transpose(1, 2).reshape_as(clean[:, :, 0])
        return endpoint.decode_latents(
            self, torch.stack([clean[:, :, 0], future], dim=2)
        )[0]
