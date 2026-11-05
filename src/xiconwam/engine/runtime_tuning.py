"""Scheduling-only training tweaks. They do not change the model or the loss.

ZeRO-2 ``overlap_comm`` allreduces the same gradient buckets on a side stream.
``gc.freeze`` only keeps the long-lived objects from the first steps out of
later full collections. The CUDA allocator setting is read before ``torch``
import. Set ``XICONWAM_FAST_RUNTIME=0`` to keep the previous runtime.
"""

from __future__ import annotations

import gc
import logging
import os

logger = logging.getLogger(__name__)

_GC_FREEZE_STEPS = (1, 2, 4)
_logged_gc_freeze = False


def fast_runtime_enabled() -> bool:
    return os.environ.get("XICONWAM_FAST_RUNTIME", "1") == "1"


def enable_expandable_segments() -> None:
    """Request expandable CUDA segments before ``torch`` is imported."""
    if not fast_runtime_enabled():
        return
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in current:
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"
        if not current
        else f"{current},expandable_segments:True"
    )


def apply_deepspeed_runtime_settings(ds_config: dict) -> None:
    """Enable ZeRO-2 communication overlap when the optimizer stays on GPU.

    Stage 3 already overlaps by default. CPU offload keeps its own path.
    Explicit ``overlap_comm: false`` in the config is left unchanged.
    """
    if not fast_runtime_enabled():
        return
    zero = ds_config.setdefault("zero_optimization", {})
    stage = int(zero.get("stage", 0))
    offload = zero.get("offload_optimizer") or {}
    device = str(offload.get("device", "none")).lower()
    if stage == 2 and device in ("none", ""):
        zero.setdefault("overlap_comm", True)
    logger.info(
        "Fast runtime: zero_stage=%s overlap_comm=%s",
        stage,
        zero.get("overlap_comm", False),
    )


def maybe_freeze_gc(global_step: int) -> None:
    """Collect once on steps 1, 2, and 4, then pin the survivors."""
    global _logged_gc_freeze
    if not fast_runtime_enabled() or global_step not in _GC_FREEZE_STEPS:
        return
    gc.collect()
    gc.freeze()
    if not _logged_gc_freeze:
        _logged_gc_freeze = True
        logger.info(
            "Froze garbage collected at early steps so later collections "
            "do not stall collectives."
        )
