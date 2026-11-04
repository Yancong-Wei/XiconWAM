# XPolicyLab_core

Minimal RoboDojo closed-loop entry for XiconWAM.

This directory vendors only the XPolicyLab websocket transport and JPEG
observation decode path (see `NOTICE`). Policy inference stays in
`deployment/model_server.py`.

The simulator client lives next door in [`../RoboDojo`](../RoboDojo)
(vendored from RoboDojo-Benchmark/RoboDojo). `eval.sh` defaults
`ROBODOJO_ROOT` to that path — you do not need `third_party/XPolicyLab`.

## Layout

```text
eval/
  XPolicyLab_core/     websocket policy shell (this package)
  RoboDojo/            vendored sim client + tasks
    XPolicyLab/policy/XiconWAM -> ../../XPolicyLab_core
```

Paths in `deploy.yml` are relative to the XiconWAM repo root. Override with
`XICON_RUN_DIR`, `XICON_CHECKPOINT`, `XICON_DATASET_STATS`, or
`server.py --overrides key=value`.

## Offline wiring check

No Isaac / Assets needed:

```bash
cd /path/to/XiconWAM
XICON_DRY_ZEROS=1 EVAL_ENV_TYPE=debug bash eval/XPolicyLab_core/eval.sh \
  RoboDojo stack_bowls step_015000 arx_x5 joint 0 0 0 \
  .venv/bin/python unused
```

## Simulation scores

1. One-time RoboDojo + Isaac setup (see `../RoboDojo/README.XiconWAM.md`):

```bash
cd eval/RoboDojo
bash scripts/install.sh -i
conda activate RoboDojo
bash scripts/init_assets.sh
```

2. Launch (policy GPU and env GPU should differ). No `ROBODOJO_ROOT` export:

```bash
bash eval/XPolicyLab_core/eval.sh \
  RoboDojo <task> step_015000 arx_x5 joint 0 \
  <policy_gpu> <env_gpu> .venv/bin/python RoboDojo
```

`action_type` must be `joint`. The server predicts 32 joint steps and returns
the first `replan_steps` (default 10) for the client to execute before
replanning.

Cloning the sim client is not enough for leaderboard numbers. On a fresh
machine you also need the policy `.venv`, the full training run directory
(`config.yaml` + `dataset_stats.json` + `step_*.pt`), Wan2.2 frozen weights
under `playground/pretrained_models/`, and a one-time
`eval/RoboDojo` Isaac + `Assets/` install. See
[`../RoboDojo/README.XiconWAM.md`](../RoboDojo/README.XiconWAM.md).
