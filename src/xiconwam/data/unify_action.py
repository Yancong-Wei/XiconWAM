"""Map heterogeneous per-robot action vectors into a unified action space.

Each dataset declares which raw dims map to which slots of a single unified
space of width ``UNIFY_DIM``. This module is the pure-mechanism layer:

  * :func:`parse_unify_spec`   — yaml spec -> ``dst_index`` array
  * :func:`map_to_unify`       — scatter raw action into the unified vector + mask
  * :func:`unmap_from_unify`   — gather the unified vector back to raw dims

Semantic layout (project contract, not enforced here)::

    0-6    left joints 6 + gripper
    7-13   right joints 6 + gripper
    14-23  left eef: xyz3 + rot6d6 + grip1
    24-33  right eef: xyz3 + rot6d6 + grip1
    34-47  reserved

Spec forms (both accepted by :func:`parse_unify_spec`)
------------------------------------------------------
* **single list** — source dims are implicitly ``0..N-1`` in order::

      ["0-13"]
      # raw 0..13 -> unified 0..13

* **paired list** — explicit ``[src_range, dst_range]`` pairs::

      [["0-13", "0-13"]]
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

# Fixed model action/state head width. Data configs only declare maps.
UNIFY_DIM = 48

ACTION_SLOT_LAYOUT = {
    "left_joint": (0, 7),
    "right_joint": (7, 14),
    "left_eef": (14, 24),
    "right_eef": (24, 34),
    "reserved": (34, 48),
}

Token = Union[int, str, Sequence]


def _expand_ranges(tokens: Token) -> List[int]:
    if isinstance(tokens, (list, tuple)):
        atoms = tokens
    else:
        atoms = [tokens]

    out: List[int] = []
    for atom in atoms:
        if isinstance(atom, (int, np.integer)):
            out.append(int(atom))
        elif isinstance(atom, str):
            s = atom.strip()
            if "-" in s:
                a_str, b_str = s.split("-", 1)
                try:
                    a, b = int(a_str), int(b_str)
                except ValueError as e:
                    raise ValueError(
                        f"unify spec: malformed range token {atom!r}"
                    ) from e
                if b < a:
                    raise ValueError(f"unify spec: range {atom!r} has end < start")
                out.extend(range(a, b + 1))
            else:
                try:
                    out.append(int(s))
                except ValueError as e:
                    raise ValueError(
                        f"unify spec: malformed index token {atom!r}"
                    ) from e
        else:
            raise ValueError(
                f"unify spec: unsupported token type {type(atom).__name__}: {atom!r}"
            )
    return out


def _coerce_spec(spec):
    try:
        from omegaconf import ListConfig, OmegaConf
    except Exception:
        return spec
    if isinstance(spec, ListConfig):
        return OmegaConf.to_container(spec, resolve=True)
    return spec


def _is_paired(spec: Sequence) -> bool:
    return all(isinstance(el, (list, tuple)) for el in spec)


def parse_unify_spec(spec: Sequence, unify_dim: int = UNIFY_DIM) -> np.ndarray:
    """Parse a yaml unify spec into a ``dst_index`` array of shape ``(N,)``."""
    spec = _coerce_spec(spec)
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError(f"unify spec must be a non-empty list, got {spec!r}")

    if _is_paired(spec):
        src_to_dst: dict[int, int] = {}
        for pair in spec:
            if len(pair) != 2:
                raise ValueError(
                    f"unify spec: paired entry must be [src, dst], got {pair!r}"
                )
            src = _expand_ranges(pair[0])
            dst = _expand_ranges(pair[1])
            if len(src) != len(dst):
                raise ValueError(
                    f"unify spec: src/dst length mismatch in {pair!r} "
                    f"(src has {len(src)}, dst has {len(dst)})"
                )
            for s, d in zip(src, dst):
                if s in src_to_dst:
                    raise ValueError(
                        f"unify spec: source dim {s} mapped more than once"
                    )
                src_to_dst[s] = d
        src_dims = sorted(src_to_dst)
        expected = list(range(len(src_dims)))
        if src_dims != expected:
            raise ValueError(
                "unify spec: source dims must cover 0..N-1 with no gaps; "
                f"got {src_dims} (expected {expected})"
            )
        dst_index = np.array([src_to_dst[s] for s in src_dims], dtype=np.int64)
    else:
        dst_index = np.array(_expand_ranges(list(spec)), dtype=np.int64)

    if dst_index.size == 0:
        raise ValueError("unify spec: parsed to an empty mapping")
    if dst_index.min() < 0 or dst_index.max() >= unify_dim:
        bad = dst_index[(dst_index < 0) | (dst_index >= unify_dim)]
        raise ValueError(
            f"unify spec: destination index/indices {bad.tolist()} out of range "
            f"[0, {unify_dim}); check unify_dim or the spec."
        )
    uniq, counts = np.unique(dst_index, return_counts=True)
    if np.any(counts > 1):
        dup = uniq[counts > 1].tolist()
        raise ValueError(
            f"unify spec: destination slot(s) {dup} mapped from more than one "
            "source dim"
        )

    return dst_index


def map_to_unify(
    action: np.ndarray, dst_index: np.ndarray, unify_dim: int = UNIFY_DIM
) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter raw action into unified space; return ``(unified, dim_valid)``.

    ``dim_valid`` is True at occupied slots (inverse of ``action_dim_is_pad``).
    """
    action = np.asarray(action)
    n = dst_index.shape[0]
    if action.shape[-1] != n:
        raise ValueError(
            f"map_to_unify: action last dim {action.shape[-1]} != mapping size {n} "
            f"(action shape {action.shape})"
        )
    unified = np.zeros((*action.shape[:-1], unify_dim), dtype=action.dtype)
    unified[..., dst_index] = action
    dim_valid = np.zeros(unify_dim, dtype=bool)
    dim_valid[dst_index] = True
    return unified, dim_valid


def unmap_from_unify(unified: np.ndarray, dst_index: np.ndarray) -> np.ndarray:
    """Gather unified vector back to raw action dims."""
    unified = np.asarray(unified)
    return unified[..., dst_index]


__all__ = [
    "UNIFY_DIM",
    "ACTION_SLOT_LAYOUT",
    "parse_unify_spec",
    "map_to_unify",
    "unmap_from_unify",
]
