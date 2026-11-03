#!/usr/bin/env bash
# Train from a task YAML. The task file selects data and model configs.
# GPU ids follow StarVLA: CUDA_VISIBLE_DEVICES / GPU_IDS.
#
#   bash scripts/train_from_yaml.sh configs/task/robodojo_a800.yaml
#   GPU_IDS=0,1 bash scripts/train_from_yaml.sh configs/task/robodojo.yaml max_steps=1000
#   ACCELERATE_CONFIG=none GPU_IDS=0 bash scripts/train_from_yaml.sh configs/task/robodojo_a800.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_YAML="${1:-}"
if [[ -z "${CONFIG_YAML}" ]]; then
  echo "Usage: bash scripts/train_from_yaml.sh configs/task/<task>.yaml [hydra overrides...]" >&2
  exit 2
fi
shift

if [[ "${CONFIG_YAML}" != /* ]]; then
  CONFIG_YAML="${ROOT}/${CONFIG_YAML}"
fi
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Task YAML not found: ${CONFIG_YAML}" >&2
  exit 2
fi

TASK="$(basename "${CONFIG_YAML}" .yaml)"
TASK_DIR="${ROOT}/configs/task"
if [[ "$(cd "$(dirname "${CONFIG_YAML}")" && pwd)" != "${TASK_DIR}" ]]; then
  echo "Task YAML must live in configs/task so Hydra can select it. Got: ${CONFIG_YAML}" >&2
  exit 2
fi

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer, got: ${NPROC}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="${ROOT}/runs/${TASK}/${RUN_ID}"
mkdir -p "${OUT}/checkpoints"
cp "${CONFIG_YAML}" "${OUT}/task.yaml"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ARGS=("--config-name=task/${TASK}" "output_dir=${OUT}" "$@")
ZERO_SPEC="$(
  PYTHONPATH="${ROOT}/src" python - "$ROOT" "$TASK" "$OUT" "$@" << 'PY'
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from xiconwam.kit.config_resolvers import register_default_resolvers

register_default_resolvers()
root, task, output, *extra = sys.argv[1:]
overrides = [f"output_dir={output}", *extra]
with initialize_config_dir(config_dir=str(Path(root) / "configs"), version_base="1.3"):
    cfg = compose(config_name=f"task/{task}", overrides=overrides)
zero = cfg.get("zero", 2)
offload = bool(cfg.get("zero_offload", False))
text = str(zero)
if text.endswith(".json") or "/" in text:
    path = Path(text)
    if not path.is_absolute():
        path = Path(root) / path
    print(path)
else:
    stage = int(text)
    name = {
        (1, False): "ds_zero1_config.json",
        (2, False): "ds_zero2_config.json",
        (2, True): "ds_zero2_offload_config.json",
        (3, False): "ds_zero3_config.json",
        (3, True): "ds_zero3_offload_config.json",
    }.get((stage, offload))
    if name is None:
        raise SystemExit(f"Unsupported zero={stage} zero_offload={offload}")
    print(Path(root) / "configs" / "deepspeed" / name)
PY
)"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
if [[ -z "${ACCELERATE_CONFIG}" ]]; then
  ACCELERATE_CONFIG="${OUT}/accelerate.yaml"
  cat > "${ACCELERATE_CONFIG}" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ${ZERO_SPEC}
  zero3_init_flag: false
mixed_precision: null
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: ${NPROC}
rdzv_backend: static
same_network: true
use_cpu: false
EOF
fi

echo "task=${TASK} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} zero=${ZERO_SPEC} output=${OUT}"
set -o pipefail
if [[ "${ACCELERATE_CONFIG}" == "none" ]]; then
  if [[ "${NPROC}" != "1" ]]; then
    echo "ACCELERATE_CONFIG=none only supports one process." >&2
    exit 2
  fi
  python -m xiconwam.train "${ARGS[@]}" 2>&1 | tee -a "${OUT}/train.log"
else
  accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NPROC}" \
    --num_machines 1 \
    -m xiconwam.train "${ARGS[@]}" 2>&1 | tee -a "${OUT}/train.log"
fi
