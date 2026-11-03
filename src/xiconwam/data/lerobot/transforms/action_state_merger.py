from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.nn.functional import pad

from xiconwam.data.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

# Alias kept for callers / tests; width is the model unify constant.
DEFAULT_ACTION_DIM = UNIFY_DIM


class ConcatLeftAlign:
    """Concat raw keys, then scatter into the fixed unify head via a slot map.

    Without ``unify_action_map``, falls back to left-align pad into
    ``action_target_dim`` (same as legacy). With a map, ``action_target_dim``
    must equal ``UNIFY_DIM`` and unused slots stay zero with dim-pad True.
    """

    def __init__(
        self,
        action_target_dim: int | None = DEFAULT_ACTION_DIM,
        state_target_dim: int | None = DEFAULT_ACTION_DIM,
        unify_action_map: Optional[Sequence] = None,
        unify_state_map: Optional[Sequence] = None,
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim
        self.unify_action_map = unify_action_map
        self.unify_state_map = unify_state_map
        self._padding_masks = {}
        self._action_dst_index = None
        self._state_dst_index = None
        if unify_action_map is not None:
            if action_target_dim is None:
                raise ValueError(
                    "`action_target_dim` is required when `unify_action_map` is set."
                )
            self._action_dst_index = parse_unify_spec(
                unify_action_map, unify_dim=int(action_target_dim)
            )
        state_map = unify_state_map if unify_state_map is not None else unify_action_map
        if state_map is not None:
            if state_target_dim is None:
                raise ValueError(
                    "`state_target_dim` is required when a unify state map is set."
                )
            self._state_dst_index = parse_unify_spec(
                state_map, unify_dim=int(state_target_dim)
            )

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if "action" in batch:
            batch["action"] = self._concat(batch["action"], self.action_meta)
            batch["action"], batch["action_dim_is_pad"] = self._to_unify(
                batch["action"],
                self.action_target_dim,
                self._action_dst_index,
            )

        batch["state"] = self._concat(batch["state"], self.state_meta)
        batch["state"], batch["state_dim_is_pad"] = self._to_unify(
            batch["state"],
            self.state_target_dim,
            self._state_dst_index,
        )

        return batch

    def backward(self, batch):
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        if self._state_dst_index is not None:
            batch["state"] = self._from_unify(batch["state"], self._state_dst_index)
        else:
            batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)

        if "action" in batch:
            if self.action_target_dim is not None:
                assert batch["action"].shape[-1] == self.action_target_dim
            if self._action_dst_index is not None:
                batch["action"] = self._from_unify(
                    batch["action"], self._action_dst_index
                )
            else:
                batch["action"] = self._crop(batch["action"], self.action_meta)
            batch["action"] = self._split(batch["action"], self.action_meta)

        return batch

    def _to_unify(self, x: torch.Tensor, dim: int | None, dst_index):
        if dst_index is not None:
            assert x.ndim == 2
            if x.shape[-1] != len(dst_index):
                raise ValueError(
                    f"raw width {x.shape[-1]} != unify map size {len(dst_index)}"
                )
            unified_np, dim_valid = map_to_unify(
                x.detach().cpu().numpy(), dst_index, unify_dim=int(dim)
            )
            unified = torch.as_tensor(unified_np, device=x.device, dtype=x.dtype)
            dim_is_pad = torch.as_tensor(~dim_valid, device=x.device, dtype=torch.bool)
            return unified, dim_is_pad
        return self._pad(x, dim)

    @staticmethod
    def _from_unify(x: torch.Tensor, dst_index: np.ndarray) -> torch.Tensor:
        assert x.ndim == 3
        raw_np = unmap_from_unify(x.detach().cpu().numpy(), dst_index)
        return torch.as_tensor(raw_np, device=x.device, dtype=x.dtype)

    def _pad(self, x: torch.Tensor, dim: int):
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim)) if pad_dim else x
        cache_key = (dim, x.shape[-1], x.device)
        if cache_key not in self._padding_masks:
            self._padding_masks[cache_key] = (
                torch.arange(dim, device=x.device) >= x.shape[-1]
            )
        mask = self._padding_masks[cache_key]
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        x = (
            x[meta[0]["key"]]
            if len(meta) == 1
            else torch.cat([x[m["key"]] for m in meta], dim=-1)
        )
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = x[:, :, idx : idx + dim]
            idx += dim

        return y
