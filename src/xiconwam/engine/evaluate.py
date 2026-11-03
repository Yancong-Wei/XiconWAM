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


class EvaluateMixin:
    @staticmethod
    def _to_batched_eval_sample(sample: dict) -> dict:
        video = sample["video"]
        if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] <= 1:
            raise ValueError(
                f"Evaluation dataset video must be [3,T,H,W] with T > 1, got {tuple(video.shape)}."
            )
        if not isinstance(sample["prompt"], str):
            raise TypeError("Evaluation dataset prompt must be a string.")
        result = {"video": video.unsqueeze(0), "prompt": [sample["prompt"]]}
        for key, rank in (
            ("action", 2),
            ("proprio", 2),
            ("context", 2),
            ("context_mask", 1),
        ):
            value = sample.get(key)
            if value is not None and value.ndim != rank:
                raise ValueError(
                    f"Evaluation dataset {key} must have rank {rank}, got {tuple(value.shape)}."
                )
            result[key] = None if value is None else value.unsqueeze(0)
        if (result["context"] is None) != (result["context_mask"] is None):
            raise ValueError("context and context_mask must both exist in eval sample.")
        action = result["action"]
        action_horizon = None if action is None else action.shape[1]
        if action_horizon is not None and action_horizon % (video.shape[1] - 1):
            raise ValueError("Action horizon must be divisible by video transitions.")
        result["action_horizon"] = action_horizon
        return result

    @torch.inference_mode()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        rng = torch.Generator(device="cpu").manual_seed(
            self.global_step + self.accelerator.process_index
        )
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        if (
            getattr(getattr(model, "video_dit", None), "visual_target", None)
            == "endpoint"
        ):
            video0 = video0[:, [0, -1]]
        action = (
            sample["action"][0]
            if "action" in sample and sample["action"] is not None
            else None
        )
        proprio = (
            sample["proprio"][0, 0]
            if "proprio" in sample and sample["proprio"] is not None
            else None
        )
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = (
            (video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
        ).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        metric_frames = getattr(
            getattr(model, "video_dit", None), "evaluation_metric_frames", None
        )

        def metric_view(video):
            return video if metric_frames is None else video[:, list(metric_frames)]

        psnr_rollout_vs_gt = video_psnr(
            pred=metric_view(pred_video_tensor), target=metric_view(gt_video_tensor)
        )

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError(
                    "Eval sample must contain `proprio` for action denormalization."
                )
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if raw_action.ndim != 2:
                    raise ValueError(
                        f"{action_name} action must be [T,D], got {tuple(raw_action.shape)}."
                    )
                action_btd = raw_action.unsqueeze(0)
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {
                        meta["key"]: batch["action"][meta["key"]].squeeze(0)
                        for meta in action_meta
                    },
                    "state": {
                        meta["key"]: batch["state"][meta["key"]].squeeze(0)
                        for meta in state_meta
                    },
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        gt_video_batch = video0.unsqueeze(0).to(
            device=model.device, dtype=model.torch_dtype
        )
        vae_latents = model._encode_video_latents(gt_video_batch)
        vae_recon_batch = model._decode_latents(vae_latents)
        if len(vae_recon_batch) != 1:
            raise ValueError(
                f"Eval VAE reconstruction must contain one video, got {len(vae_recon_batch)}."
            )
        vae_recon_video = vae_recon_batch[0]
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(
            pred=metric_view(vae_video_tensor), target=metric_view(gt_video_tensor)
        )

        psnr_rollout_vs_decode = video_psnr(
            pred=metric_view(pred_video_tensor), target=metric_view(vae_video_tensor)
        )
        ssim_rollout_vs_gt, ssim_decode_vs_gt, ssim_rollout_vs_decode = (
            video_ssim_triplet(
                metric_view(pred_video_tensor),
                metric_view(gt_video_tensor),
                metric_view(vae_video_tensor),
            )
        )

        video_path = self._save_eval_video(
            pred_video_tensor,
            vae_video_tensor,
            gt_video_tensor,
        )

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = (
            gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        )
        action_l1_mean = (
            gathered_metrics[:, 8].mean().item() if action_l1 is not None else None
        )

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_eval_video(
        self,
        pred_video_tensor: torch.Tensor,
        vae_video_tensor: torch.Tensor,
        gt_video_tensor: torch.Tensor,
    ) -> str | None:
        if not self.eval_save_video:
            return None

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (
                stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
                * 255.0
            ).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)
        return video_path
