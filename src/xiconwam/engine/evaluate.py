"""Deterministic, source-aware validation on fixed held-out windows."""

import json
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from accelerate.utils import gather_object
from PIL import Image

from xiconwam.engine.metrics import action_error_metrics
from xiconwam.kit.video_io import save_mp4
from xiconwam.kit.video_metrics import (
    pil_frames_to_video_tensor,
    video_psnr,
    video_ssim_triplet,
)


@contextmanager
def evaluation_rng(seed, device):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


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
            ("action_is_pad", 1),
            ("action_dim_is_pad", 1),
            ("image_is_pad", 1),
            ("proprio_is_pad", 1),
            ("proprio_dim_is_pad", 1),
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

    def _validation_sources(self):
        dataset = self.val_dataset
        if hasattr(dataset, "datasets") and hasattr(dataset, "names"):
            return list(zip(dataset.names, dataset.datasets))
        return [("default", dataset)]

    @torch.inference_mode()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.dit.training
        records, selection = [], {}
        self._eval_video_index = 0
        world = self.accelerator.num_processes
        rank = self.accelerator.process_index
        seed = int(self.cfg.get("eval_seed", 42))
        limit = int(self.cfg.get("eval_num_samples", 4))
        if limit < 0:
            raise ValueError("eval_num_samples must be >= 0 (0 evaluates all windows)")
        model.eval()
        try:
            for source, dataset in self._validation_sources():
                if len(dataset) == 0:
                    raise ValueError(f"Empty validation source: {source}")
                size = min(limit, len(dataset)) if limit else len(dataset)
                indices = (
                    np.random.default_rng(seed)
                    .choice(len(dataset), size=size, replace=False)
                    .tolist()
                )
                selection[source] = indices
                # Equal forward counts on all ranks, including ZeRO-3. Repeated
                # padding work is discarded rather than counted twice.
                for round_index in range(math.ceil(size / world)):
                    position = round_index * world + rank
                    index = indices[position % size]
                    with evaluation_rng(seed + index, model.device):
                        sample = self._to_batched_eval_sample(dataset[index])
                        if getattr(model, "video_only", False):
                            loss, _ = model.training_loss(sample)
                            result, weights = {"loss_video": float(loss.item())}, {}
                        else:
                            result, weights = self._evaluate_window(
                                model, sample, dataset, seed + index
                            )
                    if position < size:
                        records.append((source, result, weights))
            records = gather_object(records)
        finally:
            if was_training:
                self._set_dit_only_train_mode()
        totals = {}
        video_path = None
        for source, metrics, weights in records:
            path = metrics.pop("video_path", None)
            video_path = video_path or path
            for key, value in metrics.items():
                for name in (key, f"sources/{source}/{key}"):
                    summed, count = totals.get(name, (0.0, 0))
                    weight = weights.get(key, 1)
                    totals[name] = (summed + value * weight, count + weight)
        result = {
            name: value / count for name, (value, count) in totals.items() if count
        }
        result["video_path"] = video_path
        result["num_windows"] = len(records)
        if self.accelerator.is_main_process:
            report = {
                "step": self.global_step,
                "seed": seed,
                "selected_indices": selection,
                "metrics": result,
                "metric_counts": {key: count for key, (_, count) in totals.items()},
            }
            if not getattr(model, "video_only", False):
                report["action_l2_definition"] = "mean squared error in physical action coordinates"
            path = os.path.join(self.eval_dir, f"step_{self.global_step:06d}.json")
            temporary = path + ".tmp"
            with open(temporary, "w") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
            os.replace(temporary, path)
        return result

    def _evaluate_window(self, model, sample, dataset, seed):
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
            "seed": seed,
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

        action_metrics, action_weights = {}, {}
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval requires proprio for action denormalization")
            processor = dataset.lerobot_dataset.processor
            action_metrics, action_weights = action_error_metrics(
                processor,
                pred_action,
                action,
                sample["proprio"],
                None if sample["action_is_pad"] is None else sample["action_is_pad"][0],
                (
                    None
                    if sample["action_dim_is_pad"] is None
                    else sample["action_dim_is_pad"][0]
                ),
            )

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

        # Exclude padded frames after applying backbone-specific endpoint selection.
        frame_valid = torch.ones(video0.shape[1], dtype=torch.bool)
        if sample["image_is_pad"] is not None:
            frame_valid = ~sample["image_is_pad"][0].detach().cpu().bool()
            if frame_valid.numel() != video0.shape[1]:
                frame_valid = frame_valid[[0, -1]]
        if metric_frames is not None:
            frame_valid = frame_valid[list(metric_frames)]
        videos = [
            metric_view(v)[:, frame_valid]
            for v in (pred_video_tensor, gt_video_tensor, vae_video_tensor)
        ]
        metrics = {"val_loss": val_loss, **action_metrics}
        weights = {"val_loss": 1, **action_weights}
        if frame_valid.any():
            rollout, truth, reconstruction = videos
            ssim_rg, ssim_dg, ssim_rd = video_ssim_triplet(
                rollout, truth, reconstruction
            )
            metrics.update(
                psnr_rg=video_psnr(rollout, truth),
                ssim_rg=ssim_rg,
                psnr_rd=video_psnr(rollout, reconstruction),
                ssim_rd=ssim_rd,
                psnr_dg=video_psnr(reconstruction, truth),
                ssim_dg=ssim_dg,
            )
            weights.update(
                {
                    key: int(frame_valid.sum())
                    for key in metrics
                    if key.startswith(("psnr_", "ssim_"))
                }
            )
        metrics["video_path"] = self._save_eval_video(
            pred_video_tensor, vae_video_tensor, gt_video_tensor
        )
        return metrics, weights

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
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}_window_{getattr(self, '_eval_video_index', 0):04d}.mp4",
        )
        self._eval_video_index = getattr(self, "_eval_video_index", 0) + 1
        save_mp4(stitched_frames, video_path, fps=8)
        return video_path
