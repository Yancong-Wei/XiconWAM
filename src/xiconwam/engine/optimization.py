"""Optimizer grouping and effective-batch checks shared by launch and training."""

import math


def validate_global_batch_size(cfg, world_size):
    values = (
        int(cfg.batch_size),
        int(world_size),
        int(cfg.gradient_accumulation_steps),
    )
    if any(value <= 0 for value in values):
        raise ValueError(
            "batch_size, world_size and gradient_accumulation_steps must be positive"
        )
    actual = math.prod(values)
    expected = cfg.get("expected_global_batch_size")
    if expected is not None and actual != int(expected):
        raise ValueError(
            f"Global batch is {actual}, expected {expected}: per_gpu={values[0]} "
            f"world_size={values[1]} accumulation={values[2]}. "
            "Adjust batch_size or gradient_accumulation_steps to keep the SFT budget."
        )
    return actual


def policy_optimizer_param_groups(model, learning_rate, weight_decay):
    rates = dict(getattr(model, "group_learning_rate", None) or {})
    unknown = set(rates) - {"video", "action"}
    if unknown:
        raise ValueError(f"Unknown learning-rate groups: {sorted(unknown)}")
    for name, rate in {"default": learning_rate, **rates}.items():
        if not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError(
                f"Learning rate for {name} must be finite and positive: {rate}"
            )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Model has no trainable parameters")
    # Keep legacy MoT optimizer topology when no group rates were requested.
    if not rates:
        return [
            {"params": trainable, "lr": learning_rate, "weight_decay": weight_decay}
        ]
    video_ids = {id(parameter) for parameter in model.video_dit.parameters()}
    groups = []
    for name, is_video in (("video", True), ("action", False)):
        params = [p for p in trainable if (id(p) in video_ids) == is_video]
        if params:
            groups.append(
                {
                    "name": name,
                    "params": params,
                    "lr": float(rates.get(name, learning_rate)),
                    "weight_decay": weight_decay,
                }
            )
    return groups
