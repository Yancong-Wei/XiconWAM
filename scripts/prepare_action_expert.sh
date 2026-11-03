#!/usr/bin/env bash
# Read the task, vision tower, and output file. Resizing stays in the package.
#
#   bash scripts/prepare_action_expert.sh configs/task/robodojo.yaml wan22 \
#     playground/pretrained_models/ActionDiT_Wan22_5b.pt
#   bash scripts/prepare_action_expert.sh configs/task/robodojo.yaml cosmos25 \
#     playground/pretrained_models/ActionDiT_Cosmos25_2b.pt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_YAML="${1:-}"
BACKBONE="${2:-}"
OUTPUT="${3:-}"
if [[ -z "${CONFIG_YAML}" || -z "${BACKBONE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: bash scripts/prepare_action_expert.sh configs/task/<task>.yaml <backbone> <output.pt> [extra args...]" >&2
  exit 2
fi
shift 3

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
if [[ "${OUTPUT}" != /* ]]; then
  OUTPUT="${ROOT}/${OUTPUT}"
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m xiconwam.prepare_action_expert \
  --config-name "task/${TASK}" \
  --output "${OUTPUT}" \
  "model/backbone=${BACKBONE}" \
  "$@"
