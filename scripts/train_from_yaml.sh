#!/usr/bin/env bash
# Train from a task YAML. The task file selects data and model configs.
# Default is 8 GPUs. Override with GPU_IDS / CUDA_VISIBLE_DEVICES.
#
#   bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml
#   GPU_IDS=0,1 bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml max_steps=1000
#   ACCELERATE_CONFIG=none GPU_IDS=0 bash scripts/train_from_yaml.sh configs/task/robodojo_a800.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${1:-}" == resume=* ]]; then
  CONFIG_YAML="${ROOT}/configs/task/robodojo_wan.yaml"
else
  CONFIG_YAML="${1:-}"
  if [[ -z "${CONFIG_YAML}" ]]; then
    echo "Usage: bash scripts/train_from_yaml.sh configs/task/<task>.yaml [--dry-run] [overrides...]" >&2
    echo "Resume: bash scripts/train_from_yaml.sh resume=/path/to/run [overrides...]" >&2
    exit 2
  fi
  shift
fi
DRY_RUN=false
OVERRIDES=()
for argument in "$@"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    DRY_RUN=true
  else
    OVERRIDES+=("${argument}")
  fi
done
if [[ "${#OVERRIDES[@]}" -gt 0 ]]; then
  set -- "${OVERRIDES[@]}"
else
  set --
fi

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

DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${DEFAULT_GPU_IDS}}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer, got: ${NPROC}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="${ROOT}/runs/${TASK}/${RUN_ID}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ARGS=("--config-name=task/${TASK}" "output_dir=${OUT}" "$@")
if [[ "${DRY_RUN}" == "true" ]]; then
  python -m xiconwam.engine.launch_config --task "${TASK}" --output "${OUT}" \
    --processes "${NPROC}" --dry-run "$@"
  exit 0
fi
mkdir -p "${OUT}/checkpoints"
cp "${CONFIG_YAML}" "${OUT}/task.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
if [[ -z "${ACCELERATE_CONFIG}" ]]; then
  ACCELERATE_CONFIG="${OUT}/accelerate.yaml"
  python -m xiconwam.engine.launch_config --task "${TASK}" --output "${OUT}" \
    --processes "${NPROC}" --launcher-file "${ACCELERATE_CONFIG}" "$@"
fi

echo "task=${TASK} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} launcher_config=${ACCELERATE_CONFIG} output=${OUT}"
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
