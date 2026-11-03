import logging
import json
import inspect
import math
import os
import re
import shutil
from contextlib import nullcontext
from importlib.metadata import version as package_version
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from xiconwam.kit.fs import ensure_dir
from xiconwam.kit.logging_config import get_logger, setup_logging
from xiconwam.kit.pytorch_utils import set_global_seed, normalize_mixed_precision
from xiconwam.kit.samplers import ResumableEpochSampler
from xiconwam.kit.video_io import save_mp4
from xiconwam.kit.video_metrics import (
    pil_frames_to_video_tensor,
    video_psnr,
    video_ssim_triplet,
)

logger = get_logger(__name__)


def _cfg_none(value) -> bool:
    if value is None:
        return True
    # OmegaConf.is_none exists in 2.4+; 2.3 stores YAML null as Python None.
    is_none = getattr(OmegaConf, "is_none", None)
    return bool(is_none(value)) if callable(is_none) else False


class DataLoaderWorkerInit:
    def __init__(self, base_worker_init_fn, worker_threads: int):
        self.base_worker_init_fn = base_worker_init_fn
        self.worker_threads = int(worker_threads)

    def __call__(self, worker_id: int):
        if self.base_worker_init_fn is not None:
            self.base_worker_init_fn(worker_id)
        if self.worker_threads > 0:
            torch.set_num_threads(self.worker_threads)


def _count_parameters(module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters() if not trainable_only or p.requires_grad
    )


def _forward_training_model(model, sample):
    output = model(sample)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError(
            "Training model forward must return a `(loss, metrics)` tuple, "
            f"got {type(output).__name__}."
        )
    loss, metrics = output
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError(
            "Training model forward must return a scalar tensor as loss, "
            f"got {type(loss).__name__} with shape={getattr(loss, 'shape', None)}."
        )
    if not isinstance(metrics, dict):
        raise TypeError(
            "Training model forward must return a dict as metrics, "
            f"got {type(metrics).__name__}."
        )
    return loss, metrics


class CheckpointMixin:
    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path), optimizer=None
        )
        logger.warning(
            "Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2."
        )

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        temporary_path = f"{ckpt_path}.tmp"
        started_at = time.perf_counter()
        logger.info("[ckpt] saving weights: %s", ckpt_path)
        model.save_checkpoint(
            temporary_path,
            optimizer=None,
            step=self.global_step,
        )
        os.replace(temporary_path, ckpt_path)
        size_gib = os.path.getsize(ckpt_path) / (1024**3)
        logger.info(
            "[ckpt] saved weights: path=%s size=%.3fGiB elapsed=%.2fs",
            ckpt_path,
            size_gib,
            time.perf_counter() - started_at,
        )
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _prune_step_artifacts(directory: str, max_to_keep: int, *, suffix: str = ""):
        pattern = re.compile(rf"^step_(\d+){re.escape(suffix)}$")
        artifacts = []
        for path in Path(directory).iterdir():
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            if suffix:
                if not path.is_file():
                    continue
            elif not path.is_dir() or path.is_symlink():
                continue
            artifacts.append((int(match.group(1)), path))

        artifacts.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        for _, path in artifacts[max_to_keep:]:
            if suffix:
                path.unlink()
            else:
                shutil.rmtree(path)
            logger.info("[ckpt] removed old artifact: %s", path)

    def _prune_saved_checkpoints(self):
        if self.checkpoint_save_limit is None:
            return
        self._prune_step_artifacts(
            self.weights_dir,
            self.checkpoint_save_limit,
            suffix=".pt",
        )
        self._prune_step_artifacts(
            self.state_dir,
            self.checkpoint_save_limit,
        )

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        state_started_at = time.perf_counter()
        if self.accelerator.is_main_process:
            logger.info("[ckpt] saving distributed state: %s", state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            logger.info(
                "[ckpt] saved distributed state: path=%s elapsed=%.2fs",
                state_path,
                time.perf_counter() - state_started_at,
            )
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._prune_saved_checkpoints()
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch
                    * self.batch_size
                    * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d",
            state_dir,
            self.global_step,
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )
