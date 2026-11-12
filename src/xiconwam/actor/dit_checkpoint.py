"""Portable weights for the video and action components of a policy."""

from collections.abc import Mapping
from pathlib import Path

import torch

from xiconwam.actor.action_expert import (
    policy_action_config,
    validate_checkpoint_state_position,
    validate_policy_action_config,
)

VIDEO_FORMAT = "xiconwam.video_dit.v1"
ACTION_FORMAT = "xiconwam.action_dit.v1"


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def video_state(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as reader:
            return {
                key: reader.get_tensor(key)
                for key in reader.keys()  # noqa: SIM118 - safetensors readers are not mappings.
            }, reader.metadata() or {}
    payload = read_checkpoint(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a Video DiT state dictionary in {path}")
    if "dit" in payload:
        return payload["dit"], payload
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload, {}
    raise ValueError(f"Expected standalone Video DiT weights in {path}")


def load_video_weights(module, path, *, assign=False):
    weights, metadata = video_state(path)
    if metadata.get("backbone_name") not in (None, module.backbone_name):
        raise ValueError(
            f"Video checkpoint backbone does not match {module.backbone_name}"
        )
    from xiconwam.actor.lora import has_lora, load_standard_state_dict

    if has_lora(module):
        load_standard_state_dict(module, weights, strict=True)
    else:
        module.load_state_dict(weights, strict=True, assign=assign)


def save_video_weights(module, path, *, step=None):
    from xiconwam.actor.lora import has_lora

    if has_lora(module):
        raise ValueError(
            "Merge the Video DiT LoRA adapters before exporting standalone weights."
        )
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import save_file

        metadata = {"format": "pt", "backbone_name": module.backbone_name}
        if step is not None:
            metadata["step"] = str(step)
        weights = {
            key: value.detach().cpu().contiguous()
            for key, value in module.state_dict().items()
        }
        save_file(weights, path, metadata=metadata)
        return
    torch.save(
        {
            "format": VIDEO_FORMAT,
            "backbone_name": module.backbone_name,
            "step": step,
            "dit": module.state_dict(),
        },
        path,
    )


def save_action_weights(model, path, *, step=None):
    torch.save(
        {
            "format": ACTION_FORMAT,
            "step": step,
            "action_dit": model.action_expert.state_dict(),
            "state_encoder": (
                model.state_encoder.state_dict()
                if model.state_encoder is not None
                else None
            ),
            "state_position": model.state_position,
            "action_config": policy_action_config(model),
            "video_context_projector": (
                model.video_context_projector.state_dict()
                if getattr(model, "video_context_projector", None) is not None
                else None
            ),
        },
        path,
    )


def load_action_weights(model, path):
    payload = read_checkpoint(path)
    if payload.get("format") != ACTION_FORMAT:
        raise ValueError(
            f"Expected a standalone XiconWAM Action DiT checkpoint in {path}"
        )
    validate_checkpoint_state_position(payload, model.state_position)
    validate_policy_action_config(model, payload)
    encoder_state = payload["state_encoder"]
    if (encoder_state is None) != (model.state_encoder is None):
        raise ValueError(
            "Action checkpoint and policy disagree about the state encoder"
        )
    projector = getattr(model, "video_context_projector", None)
    projector_state = payload.get("video_context_projector")
    if (projector is None) != (projector_state is None):
        raise ValueError(
            "Action checkpoint and model disagree about the HIDE video projector"
        )
    model.action_expert.load_state_dict(payload["action_dit"], strict=True)
    if projector is not None:
        projector.load_state_dict(projector_state, strict=True)
    if model.state_encoder is not None:
        model.state_encoder.load_state_dict(encoder_state, strict=True)
    return payload
