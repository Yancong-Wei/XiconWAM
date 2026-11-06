# Vendored RoboDojo (sim client)

Shallow snapshot of [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
used by `eval/XPolicyLab_core` for closed-loop scoring.

- Policy adapter: `XPolicyLab/policy/XiconWAM` -> symlink to `../XPolicyLab_core`
- Nested `XPolicyLab` / Isaac / CuRobo git checkouts are **not** vendored here
- Downstream assets and Isaac installs stay local (see below)

## What this tree is

Simulator-side client only: `scripts/eval_policy.sh`, tasks, env configs.
It does **not** include Isaac Sim weights or HF Assets.

## One-time setup (required for real scores)

From this directory, after a working conda:

```bash
cd eval/RoboDojo
bash scripts/install.sh -i          # creates conda env RoboDojo + Isaac Sim/Lab
conda activate RoboDojo
bash scripts/init_assets.sh         # downloads Assets/ from Hugging Face
```

`install.sh` will pull Isaac Lab / CuRobo into `third_party/` (large).
Those directories are gitignored under XiconWAM.

## Run through XiconWAM

No `ROBODOJO_ROOT` needed; `eval/XPolicyLab_core/eval.sh` defaults here:

```bash
# from XiconWAM root
bash eval/XPolicyLab_core/eval.sh \
  RoboDojo <task> step_015000 arx_x5 joint 0 \
  <policy_gpu> <env_gpu> .venv/bin/python RoboDojo
```

Override only if you keep a separate RoboDojo install:

```bash
export ROBODOJO_ROOT=/other/RoboDojo
```

## Fresh machine checklist (not just clone + ckpt)

`git clone` + a `step_*.pt` is **not** enough to score. You also need:

1. Policy env: `uv sync` / install `pyproject.toml` into `.venv`
2. The full run directory (not weights alone):
   - `config.yaml`
   - `dataset_stats.json`
   - `checkpoints/weights/step_XXXXXX.pt`
3. Frozen Wan assets under `playground/pretrained_models/`
   (`Wan2.2-TI2V-5B` including UMT5 + VAE, and `ActionDiT_Wan22_5b.pt`)
4. This RoboDojo tree: `bash scripts/install.sh -i` then
   `bash scripts/init_assets.sh` and
   `python utils/update_embodiment_config_path.py`
5. Vulkan + NVIDIA driver suitable for Isaac Sim 5.1 headless

Then point the server at the run:

```bash
export XICON_RUN_DIR=runs/<your_run>
export XICON_CHECKPOINT=$XICON_RUN_DIR/checkpoints/weights/step_015000.pt
```
