"""Weighted mixture of already-packed EpisodeWindowDataset sources.

Used for multi-source pretrain. Single-source SFT keeps selecting
``data: robodojo`` (or another standalone yaml) directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from xiconwam.data.unify_action import UNIFY_DIM

logger = logging.getLogger(__name__)

_CONTRACT_FIELDS = (
    "num_frames",
    "action_video_freq_ratio",
    "video_size",
)


class MixtureDataset(torch.utils.data.Dataset):
    """Shuffled virtual index over weighted sub-datasets."""

    def __init__(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        names: Optional[Sequence[str]] = None,
    ):
        if not datasets:
            raise ValueError("MixtureDataset requires at least one sub-dataset")
        self._datasets = list(datasets)
        self._base_seed = int(seed)
        self._seed = int(seed)
        self._epoch = 0
        # Persistent DataLoader workers observe parent epoch changes too.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        if names is None:
            self._names = [f"source_{i}" for i in range(len(self._datasets))]
        else:
            names = list(names)
            if len(names) != len(self._datasets):
                raise ValueError(
                    f"names length ({len(names)}) != datasets length ({len(self._datasets)})"
                )
            self._names = [str(n) for n in names]
        if len(set(self._names)) != len(self._names):
            raise ValueError("MixtureDataset source names must be unique")

        self._validate_contract()

        if weights is None:
            weights = [float(len(d)) for d in self._datasets]
        if len(weights) != len(self._datasets):
            raise ValueError(
                f"weights length ({len(weights)}) != datasets length ({len(self._datasets)})"
            )
        weights = [float(w) for w in weights]
        if not all(np.isfinite(w) for w in weights):
            raise ValueError(f"MixtureDataset: weights must be finite: {weights}")
        if any(w < 0 for w in weights):
            raise ValueError(
                f"MixtureDataset: negative weights are not allowed: {weights}"
            )
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("MixtureDataset: all sub-sources are empty or zero-weight")
        self._weights = [w / total_w for w in weights]
        self._build_index_map()
        logger.info(
            "MixtureDataset: %d sub-datasets, total %d virtual samples, weights={%s}",
            len(self._datasets),
            len(self),
            ", ".join(f"{n}: {w:.3f}" for n, w in zip(self._names, self._weights)),
        )

    def _validate_contract(self) -> None:
        refs: Dict[str, Any] = {}
        for name, ds in zip(self._names, self._datasets):
            for field in _CONTRACT_FIELDS:
                if not hasattr(ds, field):
                    continue
                value = getattr(ds, field)
                if isinstance(value, (list, tuple)):
                    value = list(value)
                if field not in refs:
                    refs[field] = value
                elif refs[field] != value:
                    raise ValueError(
                        "MixtureDataset: enabled sub-sources must share the same "
                        f"'{field}', got {name}={value!r} vs {refs[field]!r}"
                    )
            # Prefer declared packer widths when present.
            action_dim = None
            processor = getattr(ds, "processor", None) or getattr(
                getattr(ds, "lerobot_dataset", None), "processor", None
            )
            if processor is not None and hasattr(processor, "action_output_dim"):
                action_dim = int(processor.action_output_dim)
            if action_dim is not None and action_dim != UNIFY_DIM:
                raise ValueError(
                    f"MixtureDataset: source '{name}' action_output_dim={action_dim} "
                    f"!= UNIFY_DIM={UNIFY_DIM}"
                )

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("Mixture epoch must be non-negative")
        if torch.utils.data.get_worker_info() is None:
            self._shared_epoch.fill_(int(epoch))
        new_seed = self._base_seed + int(epoch) * 7919
        if new_seed == self._seed:
            return
        self._epoch = int(epoch)
        self._seed = new_seed
        self._build_index_map()

    def _build_index_map(self) -> None:
        total_real = sum(len(d) for d in self._datasets)
        entries = []
        self._virtual_counts = [0] * len(self._datasets)
        for di, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            n_real = len(ds)
            if n_real == 0 or w <= 0.0:
                logger.warning(
                    "MixtureDataset: skipping source '%s' (len=%d, weight=%.4f)",
                    self._names[di],
                    n_real,
                    w,
                )
                continue
            virtual_n = max(1, round(total_real * w))
            self._virtual_counts[di] = virtual_n
            entries.append((di, n_real, virtual_n))
        if not entries:
            raise RuntimeError(
                "MixtureDataset: all sub-sources are empty or zero-weight"
            )
        combined = np.empty((sum(self._virtual_counts), 2), dtype=np.int32)
        offset = 0
        for di, n_real, virtual_n in entries:
            combined[offset : offset + virtual_n, 0] = di
            # Rotate through a seeded source permutation. A low-weight source
            # covers its full population across epochs rather than a prefix.
            permutation = np.random.default_rng([self._base_seed, di]).permutation(
                n_real
            )
            positions = (
                np.arange(virtual_n, dtype=np.int64) + self._epoch * virtual_n
            ) % n_real
            combined[offset : offset + virtual_n, 1] = permutation[positions]
            offset += virtual_n
        np.random.RandomState(self._seed).shuffle(combined.view(np.int64).reshape(-1))
        self._index_map = combined

    def __getitem__(self, idx: int) -> dict:
        epoch = int(self._shared_epoch.item())
        if epoch != self._epoch:
            self.set_epoch(epoch)
        di, si = self._index_map[idx]
        di, si = int(di), int(si)
        sample = dict(self._datasets[di][si])
        sample["_dataset_index"] = di
        sample["_dataset_name"] = self._names[di]
        return sample

    def __len__(self) -> int:
        return len(self._index_map)

    @property
    def datasets(self) -> List[torch.utils.data.Dataset]:
        return self._datasets

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def names(self) -> List[str]:
        return list(self._names)


def _as_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_mixture_dataset(
    sources: Mapping[str, Any],
    weight_strategy: str = "proportional",
    seed: int = 42,
    is_training_set: bool = True,
    pretrained_norm_stats: Optional[str] = None,
    source_norm_stats: Optional[Mapping[str, str]] = None,
    **unused,
) -> MixtureDataset:
    """Hydra entry: instantiate enabled sources then wrap as MixtureDataset.

    Each ``sources.<name>`` entry::

        robodojo:
          enabled: true
          weight: 1.0
          config: ${data.sources.robodojo.train}
    """
    del unused
    weight_strategy = str(weight_strategy)
    valid = ("manual", "uniform", "proportional")
    if weight_strategy not in valid:
        raise ValueError(
            f"MixtureDataset: unknown weight_strategy={weight_strategy!r}. "
            f"Valid options: {valid}."
        )

    enabled_names: List[str] = []
    enabled_cfgs: List[Any] = []
    manual_weights: List[float] = []
    for name, entry in dict(sources).items():
        entry = _as_plain(entry) or {}
        if isinstance(entry, DictConfig):
            entry = OmegaConf.to_container(entry, resolve=True)
        if not bool(entry.get("enabled", True)):
            logger.info("MixtureDataset: skipping disabled source '%s'", name)
            continue
        cfg = entry.get("config")
        if cfg is None:
            raise ValueError(
                f"MixtureDataset: source '{name}' is missing `config` "
                "(the EpisodeWindowDataset hydra node)."
            )
        enabled_names.append(str(name))
        enabled_cfgs.append(cfg)
        manual_weights.append(float(entry.get("weight", 1.0)))

    if not enabled_cfgs:
        raise RuntimeError("MixtureDataset: all sub-datasets are disabled")

    sub_datasets = []
    from xiconwam.kit import misc

    for name, cfg in zip(enabled_names, enabled_cfgs):
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Invalid mixture source name: {name!r}")
        kwargs = {
            "is_training_set": bool(is_training_set),
            "norm_stats_output": str(
                Path(misc.get_work_dir()) / "dataset_stats" / f"{name}.json"
            ),
        }
        if pretrained_norm_stats is not None:
            kwargs["pretrained_norm_stats"] = pretrained_norm_stats
        elif not cfg.get("pretrained_norm_stats") and source_norm_stats:
            if name in source_norm_stats:
                kwargs["pretrained_norm_stats"] = source_norm_stats[name]
        sub_datasets.append(instantiate(cfg, **kwargs))
        logger.info(
            "MixtureDataset: loaded source '%s' with %d samples",
            name,
            len(sub_datasets[-1]),
        )

    if weight_strategy == "uniform":
        weights = [1.0] * len(sub_datasets)
    elif weight_strategy == "proportional":
        weights = [float(len(ds)) for ds in sub_datasets]
    else:
        weights = manual_weights

    return MixtureDataset(
        datasets=sub_datasets,
        weights=weights,
        seed=int(seed),
        names=enabled_names,
    )


__all__ = ["MixtureDataset", "build_mixture_dataset"]
