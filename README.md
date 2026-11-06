# XiconWAM

XiconWAM trains dual-stream world-action policies with pretrained video backbones for robot manipulation. The default trainer updates the joint DiT and state encoder while keeping the VAE and text encoder frozen.

Supported vision towers: **Wan2.2-TI2V-5B**, **Cosmos-Predict2.5-2B**, **FLUX.2-klein-4B**.  
Supported policy heads: **`mot`** (default), **`idm`**, **`hide`**.  
Primary benchmark integration: **RoboDojo** (LeRobot v3.0, joint14, three cameras).

Validated stack: Linux, Python 3.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128 (CUDA 12.8).  
Default launch uses **8 GPUs**. Single-GPU 80GB recipe: `configs/task/robodojo_a800.yaml`.

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

The lockfile uses PyPI for ordinary packages and the explicit CUDA 12.8 index for `torch==2.7.1+cu128` and `torchvision==0.22.1+cu128`. Use this recipe on the Linux training host. Engine is installed separately from the reviewed wheel as described in the [integration guide](docs/xicon_engine_model_side.md).

```bash
uv python install 3.10
uv venv --python 3.10 .venv
uv sync --locked
source .venv/bin/activate
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

Dataset paths may be individual LeRobot v3 roots or nested containers discovered
through `meta/info.json`. See [nested dataset discovery](docs/dataset_discovery.md)
for inventory commands, deterministic traversal, and cross-robot limitations.

Large assets should live outside the git tree, then symlink into `playground/`:

```bash
mkdir -p playground/datasets playground/pretrained_models
# RoboDojo joint14 (this machine)
ln -sfn /mnt/pangyunyi/datasets/robodojo-joint playground/datasets/robodojo
# Or a generic data root:
# ln -sfn /data/xiconwam/datasets/robodojo playground/datasets/robodojo
# ln -sfn /data/xiconwam/pretrained_models playground/pretrained_models
```

Expected pretrained layout (no extra nesting):

```text
playground/pretrained_models/Wan2.2-TI2V-5B/
playground/pretrained_models/ActionDiT_Wan22_5b.pt
playground/pretrained_models/Cosmos-Predict2.5-2B/
playground/pretrained_models/Cosmos-Reason1-7B/
playground/pretrained_models/ActionDiT_Cosmos25_2b.pt
playground/pretrained_models/flux2/
playground/pretrained_models/ActionDiT_Flux2_4b.pt
```

### Dataset normalization stats

Training reads `playground/datasets/robodojo/dataset_stats.json` (XiconWAM format with
`action` / `state` → `default` → `global_mean` / `global_std` / …).  
Do **not** rename LeRobot `meta/stats.json`. Compute it once:

```bash
python scripts/compute_dataset_stats.py
# or:
python scripts/compute_dataset_stats.py --config-name task/robodojo_wan \
  --output playground/datasets/robodojo/dataset_stats.json
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

Action DiT files are initializations resized from the video tower, not trained checkpoints. The task YAML selects the tower and `action_dit_pretrained_path`, and the script saves there. Run from the repository root with `.venv` active (`source .venv/bin/activate`); the script calls `python` on `PATH`.

```bash
bash scripts/prepare_action_expert.sh configs/task/robodojo_wan.yaml
bash scripts/prepare_action_expert.sh configs/task/robodojo_cosmos.yaml
bash scripts/prepare_action_expert.sh configs/task/robodojo_flux.yaml
```

`robodojo_wan` uses Wan2.2 and writes `ActionDiT_Wan22_5b.pt`. `robodojo_cosmos` uses Cosmos-Predict2.5-2B and writes `ActionDiT_Cosmos25_2b.pt`. `robodojo_flux` uses FLUX.2-klein-base-4B and writes `ActionDiT_Flux2_4b.pt`. All three files land in `playground/pretrained_models/`. FLUX also reads `flux2/FLUX.2-dev/ae.safetensors`. Its text encoder spec is `Qwen/Qwen3-4B`, loaded only when `load_text_encoder` is true.

Training reads cached text embeddings. Rebuild the cache after switching towers:

```bash
bash scripts/precompute_text_embeds.sh configs/task/robodojo_wan.yaml
bash scripts/precompute_text_embeds.sh configs/task/robodojo_cosmos.yaml
bash scripts/precompute_text_embeds.sh configs/task/robodojo_flux.yaml
```

Caches are written to `playground/datasets/text_embeds_cache/robodojo/<backbone>/`.

## 3. Training

A task config selects data, model, batch size, learning rate, schedule, and ZeRO settings. `robodojo_wan`, `robodojo_cosmos`, and `robodojo_flux` share those settings and pin the vision tower. `robodojo_a800` inherits from `robodojo_wan` for single-GPU 80GB. Validation reuses the train dataset config with `is_training_set=false`.

For a fresh 90k-update RoboDojo SFT on four GPUs with global batch 128, use
[`robodojo_wan_sft90k`](docs/robodojo_sft90k.md). It adds video/action learning-rate
groups, gradient checkpointing, a batch-size guard, and broader offline validation.

```text
configs/task/robodojo_wan.yaml       # backbone wan22 (default 8-GPU entry)
configs/task/robodojo_cosmos.yaml    # backbone cosmos25
configs/task/robodojo_flux.yaml      # backbone flux2
configs/task/robodojo_a800.yaml      # single-GPU 80GB override
  data/robodojo          # single-source SFT
  data/mixture           # multi-source pretrain shell
  model/mot | idm | hide
```

Action / state heads use a fixed `UNIFY_DIM=48`. Datasets only declare `unify_action_map` (RoboDojo joints: `["0-13"]`). Missing cameras are filled with black frames. Text-cache misses can fall back to online encoding when `load_text_encoder_on_miss=true`.

```bash
# default: 8 GPUs (GPU_IDS=0,1,2,3,4,5,6,7)
bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml

# fewer GPUs
GPU_IDS=0,1,2,3 bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml

# single-GPU recipe
bash scripts/train_single_gpu.sh configs/task/robodojo_a800.yaml

# short smoke on one GPU
GPU_IDS=0 bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml max_steps=10 log_every=1
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

Default ZeRO stage is 2 with the optimizer on GPU. `scripts/train_from_yaml.sh` defaults to 8 processes and materializes Accelerate config under the run directory pointing at `configs/deepspeed/`. `ACCELERATE_CONFIG=none` forces single-process PyTorch (`NPROC=1` only).

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

Closed-loop scoring lives under [`eval/`](eval/):

- [`eval/XPolicyLab_core`](eval/XPolicyLab_core): websocket policy shell around
  `deployment/model_server.py` (no `third_party/XPolicyLab` needed)
- [`eval/RoboDojo`](eval/RoboDojo): vendored RoboDojo sim client + tasks
  (`ROBODOJO_ROOT` defaults here)

Offline wiring check (no Isaac; dry-zeros skips weight load):

```bash
XICON_DRY_ZEROS=1 EVAL_ENV_TYPE=debug bash eval/XPolicyLab_core/eval.sh \
  RoboDojo stack_bowls step_015000 arx_x5 joint 0 0 0 \
  .venv/bin/python unused
```

Simulation (one-time Isaac + assets under `eval/RoboDojo`, see
[`eval/RoboDojo/README.XiconWAM.md`](eval/RoboDojo/README.XiconWAM.md)):

```bash
cd eval/RoboDojo && bash scripts/install.sh -i && bash scripts/init_assets.sh
# then from repo root:
XICON_RUN_DIR=/path/to/training/run bash eval/XPolicyLab_core/eval.sh \
  RoboDojo <task> step_015000 arx_x5 joint 0 \
  <policy_gpu> <env_gpu> .venv/bin/python RoboDojo
```

The server loads the run `config.yaml`, so architecture stays aligned with the
checkpoint. Use `action_type=joint`. Real-robot ZMQ serving remains under
`deployment/` and is separate from this websocket path.

### Xicon Engine 开环验证与真机接口

模型端服务、无权重三相机/14D 演示、Engine pred/GT 开环验证及 robot_control 配合见
[模型端接入方案](docs/xicon_engine_model_side.md)。RoboDojo 打榜继续使用上面的 XPolicyLab 入口。
当前已验证通信和合成数据链路，尚未训练/验证实际部署 checkpoint。

可迁移模型包的导出、校验、加载，以及固定验证集和数据错误策略，见
[部署模型包](docs/deployment_bundle.md)。入口：`xiconwam-bundle export|verify`、
`xiconwam-serve --bundle /path/to/bundle`。模型包绑定资源完整性，保留当前研究权重加载的兼容行为。

## 5. Project layout

```text
configs/          Hydra data / model / task / deepspeed recipes
src/xiconwam/     training entry, actors, vision backends, data, engine
scripts/          launchers, compute_dataset_stats.py, ModelScope IO helpers
deployment/       ZMQ model_server / robot_client / fake_lerobot_client
eval/             XPolicyLab_core + vendored RoboDojo sim client
tests/            unit / composition checks
playground/       local symlinks to datasets and pretrained weights (gitignored)
```

## 6. License and attribution

The project license is in [LICENSE](LICENSE). If this tree redistributes or derives from upstream model/code assets (Wan, Cosmos, FLUX, LeRobot, RoboDojo, or related research code), keep those upstream licenses and attributions intact and cite the corresponding papers/repositories in your release notes.

This repository is a research codebase. Presence of configs or backends does not imply that every combination has been validated end-to-end on RoboDojo or Isaac Sim.
