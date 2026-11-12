"""Real CPU optimizer/RNG/sampler recovery, including an epoch boundary."""

import copy

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from xiconwam.engine.session import TrainSession


class Windows(Dataset):
    def __init__(self, size=12):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor([index / 12], dtype=torch.float32)


class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(1, 1)

    def forward(self, sample):
        noise = torch.randn_like(sample) * 0.1
        loss = (self.dit(sample) - sample - noise).square().mean()
        return loss, {"loss_action": loss.detach()}

    def save_checkpoint(self, path, optimizer=None, step=None):
        torch.save(self.dit.state_dict(), path)


class Interrupted(Exception):
    pass


def fit(output, *, resume=None, interrupt_at=None, dataset_size=12, batch_size=1):
    cfg = OmegaConf.create(
        dict(
            output_dir=str(output),
            learning_rate=0.01,
            warmup_ratio=0.0,
            weight_decay=0.01,
            batch_size=batch_size,
            num_workers=0,
            log_every=10,
            save_every=1,
            checkpoint_save_limit=2,
            eval_every=0,
            eval_num_inference_steps=1,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            seed=42,
            resume=None if resume is None else str(resume),
            mixed_precision="no",
            wandb={"enabled": False},
            max_steps=8,
            lr_scheduler_type="cosine",
        )
    )
    torch.manual_seed(17)
    accelerator = Accelerator(
        cpu=True,
        mixed_precision="no",
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )
    try:
        session = TrainSession(
            StochasticModel(), Windows(dataset_size), cfg=cfg, accelerator=accelerator
        )
        if interrupt_at is not None:
            save = session.save_checkpoint

            def interrupt_after_save():
                result = save()
                if session.global_step == interrupt_at:
                    raise Interrupted()
                return result

            session.save_checkpoint = interrupt_after_save
            with pytest.raises(Interrupted):
                session.train()
        else:
            session.train()
        return copy.deepcopy(
            {
                "model": accelerator.unwrap_model(session.model).state_dict(),
                "optimizer": session.optimizer.state_dict(),
                "scheduler": session.scheduler.state_dict(),
                "step": session.global_step,
                "epoch": session.epoch,
                "batch_in_epoch": session.batch_in_epoch,
                "next_noise": torch.randn(3),
            }
        )
    finally:
        accelerator.free_memory()


@pytest.mark.parametrize(
    "dataset_size,batch_size,interrupt_at",
    [(12, 1, 2), (12, 1, 6), (12, 1, 7), (13, 2, 2), (13, 2, 4), (13, 2, 5)],
)
def test_resume_matches_continuous_training(
    tmp_path, dataset_size, batch_size, interrupt_at
):
    options = {"dataset_size": dataset_size, "batch_size": batch_size}
    reference = fit(tmp_path / "reference", **options)
    fit(tmp_path / "interrupted", interrupt_at=interrupt_at, **options)
    restored = fit(
        tmp_path / "resumed",
        resume=tmp_path / "interrupted/checkpoints/state" / f"step_{interrupt_at:06d}",
        **options,
    )
    torch.testing.assert_close(reference, restored, rtol=0, atol=0)
