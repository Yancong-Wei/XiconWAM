"""Shared action-loss masking for time pad and feature-dim pad."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_action_mse_per_sample(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: Optional[torch.Tensor] = None,
    action_dim_is_pad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample MSE over valid (time, dim) cells.

    Args:
        pred_action / target_action: ``[B, T, D]``
        action_is_pad: ``[B, T]`` True where the timestep is padding
        action_dim_is_pad: ``[B, D]`` or ``[D]`` True where the feature is padding

    Returns:
        ``[B]`` mean squared error over non-pad cells (1.0 when a sample has none).
    """
    err = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
    valid = torch.ones_like(err, dtype=torch.bool)
    if action_is_pad is not None:
        if action_is_pad.ndim != 2:
            raise ValueError(
                f"`action_is_pad` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
            )
        valid &= ~action_is_pad[:, :, None]
    if action_dim_is_pad is not None:
        dim_pad = action_dim_is_pad
        if dim_pad.ndim == 1:
            dim_pad = dim_pad[None, :]
        if dim_pad.ndim != 2:
            raise ValueError(
                "`action_dim_is_pad` must be 1D [D] or 2D [B, D], "
                f"got shape {tuple(action_dim_is_pad.shape)}"
            )
        valid &= ~dim_pad[:, None, :]
    valid_f = valid.to(dtype=err.dtype)
    denom = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
    return (err * valid_f).sum(dim=(1, 2)) / denom
