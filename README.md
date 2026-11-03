# XiconWAM

XiconWAM trains dual-stream world-action policies on frozen video backbones for robot manipulation.

Supported vision towers: **Wan2.2-TI2V-5B**, **Cosmos-Predict2.5-2B**, **FLUX.2-klein-4B**.  
Supported policy heads: **`mot`** (default), **`idm`**, **`hide`**.  
Primary benchmark integration: **RoboDojo** (LeRobot v3.0, joint14, three cameras).

Validated stack: Linux, Python 3.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128 (CUDA 12.8).  
Single-GPU 80GB recipe: `configs/task/robodojo_a800.yaml`.

> RoboDojo alone is about 112 GiB before vision weights. Keep datasets and checkpoints on a data disk, then symlink them into `playground/`.

## Contents

1. [Install](#1-install)
2. [Data and pretrained weights](#2-data-and-pretrained-weights)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Project layout](#5-project-layout)
6. [License and attribution](#6-license-and-attribution)

## 1. Install

All commands below assume the repository root. Dependencies are declared in `pyproject.toml`. `torch==2.7.1+cu128` is not on the default PyPI index; install with the CUDA 12.8 wheel source.

### conda

```bash
conda create -n xiconwam python=3.10 -y
conda activate xiconwam
python -m pip install -U pip
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

### uv

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify GPU visibility:

```bash
python -c "import torch, xiconwam; print(torch.__version__, torch.cuda.get_device_name(0))"
```

FlashAttention is optional. With `attention_backend: auto`, the code falls back to PyTorch SDPA when FlashAttention is unavailable.

```bash
python -m pip install flash-attn --no-build-isolation
```

## 2. Data and pretrained weights

Large assets should live outside the git tree:

```bash
mkdir -p /data/xiconwam
ln -sfn /data/xiconwam/datasets playground/datasets
ln -sfn /data/xiconwam/pretrained_models playground/pretrained_models
```

Expected pretrained layout (no extra nesting):

```text
playground/pretrained_models/Wan2.2-TI2V-5B/
playground/pretrained_models/ActionDiT_Wan22_5b.pt
playground/pretrained_models/Cosmos-Predict2.5-2B/
playground/pretrained_models/Cosmos-Reason1-7B/
playground/pretrained_models/ActionDiT_Cosmos25_2b.pt
```

### ModelScope helpers

```bash
python -m pip install 'modelscope==1.34.0' 'setuptools<81'
export MODELSCOPE_API_TOKEN=...   # never commit the token
export MODELSCOPE_MODEL_REPO=owner/WAM_pretrained
export MODELSCOPE_DATASET_REPO=owner/robodojo

bash scripts/io/download_models.sh
bash scripts/io/download_datasets.sh
# upload counterparts: scripts/io/upload_models.sh / upload_datasets.sh
```

You can also pass `owner/name` as the first CLI argument instead of exporting the repo env vars.

### Hugging Face dataset mirror

```bash
hf download OpenMOSS-Team/robodojo-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir playground/datasets/robodojo
```

FLUX weights belong under `playground/pretrained_models/flux2/`; see `configs/model/backbone/flux2.yaml`. The `hide` policy needs cross-attention on the action expert and is not wired for FLUX.

If the action-expert checkpoint is missing or the vision tower changed, rebuild it:

```bash
bash scripts/prepare_action_expert.sh configs/task/robodojo.yaml wan22 \
  playground/pretrained_models/ActionDiT_Wan22_5b.pt
bash scripts/prepare_action_expert.sh configs/task/robodojo.yaml cosmos25 \
  playground/pretrained_models/ActionDiT_Cosmos25_2b.pt
```

Training reads cached text embeddings only. Rebuild after switching towers:

```bash
bash scripts/precompute_text_embeds.sh configs/task/robodojo.yaml wan22
bash scripts/precompute_text_embeds.sh configs/task/robodojo.yaml cosmos25
```

Caches are written to `playground/datasets/text_embeds_cache/robodojo/<backbone>/`.

## 3. Training

A task config selects data, model, batch size, learning rate, schedule, and ZeRO settings. `robodojo_a800` inherits from `robodojo`. Validation reuses the train dataset config with `is_training_set=false`.

```text
configs/task/robodojo.yaml
  data/robodojo          # single-source SFT
  data/mixture           # multi-source pretrain shell
  model/mot | idm | hide
    backbone  wan22 | cosmos25 | flux2
```

Action / state heads use a fixed `UNIFY_DIM=48`. Datasets only declare `unify_action_map` (RoboDojo joints: `["0-13"]`). Missing cameras are filled with black frames. Text-cache misses can fall back to online encoding when `load_text_encoder_on_miss=true`.

```bash
# single-source SFT (default)
bash scripts/train_single_gpu.sh configs/task/robodojo_a800.yaml

# multi-source pretrain shell
bash scripts/train_single_gpu.sh configs/task/robodojo_a800.yaml data=mixture

# short smoke run
bash scripts/train_single_gpu.sh configs/task/robodojo_a800.yaml max_steps=10 log_every=1
```

Common overrides:

```bash
model=idm
model=hide
model/backbone=cosmos25
model/backbone=flux2
model.lora.enable=true model.lora.rank=16
zero=3 zero_offload=true
```

Default ZeRO stage is 2 with the optimizer on GPU. Launch scripts materialize Accelerate config under the run directory and point to `configs/deepspeed/`. `ACCELERATE_CONFIG=none` forces single-process PyTorch (`NPROC=1` only).

Run outputs:

```text
runs/<task>/<run_id>/
  task.yaml
  config.yaml          # fully resolved config, including CLI overrides
  accelerate.yaml
  train.log
  checkpoints/weights
  checkpoints/state
```

Resume and evaluation should use this run's `config.yaml`, not the original task file in the repo.

VAE and text encoders stay frozen. Training updates DiT and the state encoder. Cosmos additionally freezes `crossattn_proj`. LoRA is off by default and, when enabled, attaches only to video-tower `lora_target_modules`.

## 4. Evaluation

Closed-loop RoboDojo evaluation uses the XPolicyLab websocket server:

```bash
XICON_CHECKPOINT_PATH=/path/to/run/checkpoints/weights/step_003000.pt \
  bash XPolicyLab/policy/XiconWAM/eval.sh \
  RoboDojo <task> <ckpt> arx_x5 joint 0 0 0 <policy_env> <eval_env>
```

The server loads the run `config.yaml`, so architecture, vision tower, and LoRA settings stay aligned with the checkpoint. With `num_frames: 33` and `action_video_freq_ratio: 4`, the model observes 9 frames and predicts 32 action steps; the robot executes `replan_steps` (default 10) before replanning.

## 5. Project layout

```text
configs/          Hydra data / model / task / deepspeed recipes
src/xiconwam/     training entry, actors, vision backends, data, engine
scripts/          launchers and ModelScope IO helpers
eval/             observation packing helpers for XPolicyLab
tests/            unit / composition checks
playground/       local symlinks to datasets and pretrained weights (gitignored)
```

## 6. License and attribution

Add a top-level `LICENSE` before publishing. If this tree redistributes or derives from upstream model/code assets (Wan, Cosmos, FLUX, LeRobot, RoboDojo, or related research code), keep those upstream licenses and attributions intact and cite the corresponding papers/repositories in your release notes.

This repository is a research codebase. Presence of configs or backends does not imply that every combination has been validated end-to-end on RoboDojo or Isaac Sim.
