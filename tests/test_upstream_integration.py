"""Compatibility between upstream training/evaluation and the deployment branch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from xiconwam.kit.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]
register_default_resolvers()


@pytest.mark.parametrize(
    "task,backbone",
    [
        ("robodojo_wan", "wan22"),
        ("robodojo_cosmos", "cosmos25"),
        ("robodojo_flux", "flux2"),
    ],
)
def test_recipes_keep_source_aware_validation(task, backbone):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name=f"task/{task}", overrides=["data=mixture"])
    OmegaConf.resolve(cfg)
    assert cfg.model.video_dit.name == backbone
    assert cfg.eval_num_samples == 4
    assert cfg.eval_seed == 42
    assert cfg.data.train.sources.robodojo.config.processor.action_output_dim == 48


def policy_module():
    spec = importlib.util.spec_from_file_location(
        "xiconwam_robodojo_eval_policy", ROOT / "eval/XPolicyLab_core/policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_fixture_needs_no_existing_training_run():
    module = policy_module()
    cfg = yaml.safe_load((ROOT / "eval/XPolicyLab_core/deploy.yml").read_text())
    assert cfg["run_dir"] is None and cfg["checkpoint"] is None
    with pytest.raises(ValueError, match="XICON_RUN_DIR"):
        module.Policy(cfg)
    cfg.update(dry_zeros=True, ckpt_name="does-not-exist")
    policy = module.Policy(cfg)
    policy.update_obs({})
    assert len(policy.get_action()) == cfg["replan_steps"]


def test_eval_selects_requested_checkpoint_instead_of_latest(tmp_path, monkeypatch):
    module = policy_module()
    weights = tmp_path / "checkpoints/weights"
    weights.mkdir(parents=True)
    for name in ("step_000100.pt", "step_000200.pt"):
        (weights / name).write_bytes(b"fixture")
    monkeypatch.setattr(
        module, "XiconServeRuntime", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    cfg = yaml.safe_load((ROOT / "eval/XPolicyLab_core/deploy.yml").read_text())
    cfg.update(run_dir=str(tmp_path), ckpt_name="step_000100")
    assert module.Policy(cfg).runtime.checkpoint_path == weights / "step_000100.pt"
    cfg["ckpt_name"] = "missing"
    with pytest.raises(FileNotFoundError, match="ckpt_name"):
        module.Policy(cfg)


def test_action_initialization_cli_keeps_explicit_output(tmp_path, monkeypatch):
    from xiconwam import prepare_action_expert as entry

    output = tmp_path / "custom.pt"
    monkeypatch.setattr(
        entry,
        "build_initialization",
        lambda *args, **kwargs: {
            "initialization": {},
            "policy": {"source_backbone": "wan22"},
            "weight": torch.ones(1),
        },
    )
    entry.main(["--config-name", "task/robodojo_wan", "--output", str(output)])
    assert torch.load(output, weights_only=True)["weight"].item() == 1


@pytest.mark.parametrize("supports_proxy", [False, True])
def test_eval_websocket_defaults_to_direct_connection(monkeypatch, supports_proxy):
    import asyncio

    monkeypatch.syspath_prepend(str(ROOT / "eval/XPolicyLab_core"))
    from client_server.ws.protocol import client as module

    calls = []

    async def modern_connect(url, *, proxy=True, **kwargs):
        calls.append(proxy)
        raise RuntimeError("connection probe")

    async def legacy_connect(url, **kwargs):
        assert "proxy" not in kwargs
        calls.append(None)
        raise RuntimeError("connection probe")

    monkeypatch.setattr(
        module.websockets,
        "connect",
        modern_connect if supports_proxy else legacy_connect,
    )
    client = module.PolicyEvalClient(
        module.PolicyEvalClientConfig(
            url="ws://127.0.0.1:1",
            evaluation_id="fixture",
            max_connect_attempts=1,
        )
    )
    with pytest.raises(ConnectionError, match="connection probe"):
        asyncio.run(client.connect())
    assert calls == [None]
