import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from xiconwam.actor.mot import MoTPolicy
from xiconwam.engine.optimization import (
    policy_optimizer_param_groups,
    validate_global_batch_size,
)
from xiconwam.engine.session import TrainSession

ROOT = Path(__file__).resolve().parents[1]


def toy_policy():
    # Exercise real inherited grouping with aliased towers, as in StreamMixer.
    policy = MoTPolicy.__new__(MoTPolicy)
    torch.nn.Module.__init__(policy)
    policy.video_expert = torch.nn.Linear(2, 2)
    policy.action_expert = torch.nn.Linear(2, 2)
    policy.state_encoder = torch.nn.Linear(2, 2)
    policy.vae = torch.nn.Linear(2, 2).requires_grad_(False)
    policy.streams = torch.nn.ModuleDict(
        {"video": policy.video_expert, "action": policy.action_expert}
    )
    policy.group_learning_rate = {"video": 1e-5, "action": 1e-4}
    return policy


def test_grouping_includes_state_excludes_frozen_and_never_duplicates_aliases():
    policy = toy_policy()
    groups = policy.optimizer_param_groups(1e-4, 0.01)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in policy.parameters() if p.requires_grad}
    assert [g["name"] for g in groups] == ["video", "action"]
    assert [g["lr"] for g in groups] == [1e-5, 1e-4]
    assert {id(p) for p in policy.state_encoder.parameters()} <= {
        id(p) for p in groups[1]["params"]
    }
    policy.group_learning_rate = {}
    assert len(policy.optimizer_param_groups(1e-4, 0.01)) == 1


def test_optimizer_updates_both_towers_and_state_with_the_requested_rates():
    policy = toy_policy()
    optimizer = torch.optim.SGD(policy.optimizer_param_groups(1e-4, 0.0))
    before = {id(p): p.detach().clone() for p in policy.parameters()}
    for p in policy.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    optimizer.step()
    video_ids = {id(p) for p in policy.video_dit.parameters()}
    for p in policy.parameters():
        delta = before[id(p)] - p.detach()
        expected = (
            0.0 if not p.requires_grad else (1e-5 if id(p) in video_ids else 1e-4)
        )
        torch.testing.assert_close(
            delta, torch.full_like(p, expected), atol=3e-8, rtol=0
        )


@pytest.mark.parametrize(
    "rates", [{"video": float("nan")}, {"action": -1}, {"typo": 1e-5}]
)
def test_invalid_group_rates_fail(rates):
    policy = toy_policy()
    policy.group_learning_rate = rates
    with pytest.raises(ValueError):
        policy_optimizer_param_groups(policy, 1e-4, 0.01)


def scheduler_run():
    session = TrainSession.__new__(TrainSession)
    session.is_deepspeed = False
    session.learning_rate = 1e-4
    session.cfg = OmegaConf.create({"lr_min_ratio": 0.01})
    session.optimizer = torch.optim.AdamW(
        toy_policy().optimizer_param_groups(1e-4, 0.01)
    )
    scheduler = session._build_scheduler("cosine", total_train_steps=20, warmup_steps=2)
    return session.optimizer, scheduler


def test_cosine_preserves_group_ratio_and_resumes_the_same_curve():
    optimizer, scheduler = scheduler_run()
    saved = None
    trajectory = []
    for step in range(20):
        optimizer.step()
        scheduler.step()
        lrs = [g["lr"] for g in optimizer.param_groups]
        assert lrs[1] / lrs[0] == pytest.approx(10)
        trajectory.append(lrs)
        if step == 9:
            saved = (
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
            )
    assert trajectory[-1] == pytest.approx([1e-7, 1e-6])
    restored_optimizer, restored_scheduler = scheduler_run()
    restored_optimizer.load_state_dict(saved[0])
    restored_scheduler.load_state_dict(saved[1])
    for lrs in trajectory[10:]:
        restored_optimizer.step()
        restored_scheduler.step()
        assert [g["lr"] for g in restored_optimizer.param_groups] == pytest.approx(lrs)


def test_equal_group_peaks_use_their_own_cosine_floor():
    session = TrainSession.__new__(TrainSession)
    session.is_deepspeed = False
    session.learning_rate = 1e-4
    session.cfg = OmegaConf.create({"lr_min_ratio": 0.01})
    policy = toy_policy()
    policy.group_learning_rate = {"video": 1e-5, "action": 1e-5}
    session.optimizer = torch.optim.AdamW(policy.optimizer_param_groups(1e-4, 0.01))
    scheduler = session._build_scheduler("cosine", total_train_steps=20)
    for _ in range(20):
        session.optimizer.step()
        scheduler.step()
    assert [g["lr"] for g in session.optimizer.param_groups] == pytest.approx(
        [1e-7, 1e-7]
    )


def test_fresh_sft_recipe_preserves_budget_on_four_or_eight_gpus():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="task/robodojo_wan_sft90k")
    assert cfg.data.train._target_.endswith("EpisodeWindowDataset")
    assert cfg.model.architecture == "mot"
    assert cfg.max_steps == 90000 and cfg.resume is None and cfg.max_epochs is None
    assert validate_global_batch_size(cfg, 4) == 128
    with pytest.raises(ValueError, match="Global batch is 256, expected 128"):
        validate_global_batch_size(cfg, 8)
    cfg.gradient_accumulation_steps = 4
    assert validate_global_batch_size(cfg, 8) == 128
    assert cfg.model.group_learning_rate.video == 1e-5
    assert cfg.model.group_learning_rate.action == 1e-4
    assert cfg.eval_num_samples == 64
    assert cfg.model.backbone.use_gradient_checkpointing


def test_build_model_forwards_group_rates_to_mot(monkeypatch):
    from xiconwam import launch

    model = toy_policy()
    model.group_learning_rate = {}
    monkeypatch.setattr(launch, "_build_policy", lambda *args, **kwargs: model)
    monkeypatch.setattr(launch, "_apply_lora", lambda model, payload: model)
    launch.build_model(
        backbone={"name": "wan22"}, group_learning_rate={"video": 1e-5, "action": 1e-4}
    )
    assert model.optimizer_param_groups(1e-4, 0.01)[0]["lr"] == 1e-5
