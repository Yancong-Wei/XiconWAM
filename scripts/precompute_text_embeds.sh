#!/usr/bin/env bash
# Read the task and vision tower. Text encoding stays in the package.
# The cache directory comes from the task config.
#
#   bash scripts/precompute_text_embeds.sh configs/task/robodojo.yaml wan22
#   bash scripts/precompute_text_embeds.sh configs/task/robodojo.yaml cosmos25 --overwrite
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_YAML="${1:-}"
BACKBONE="${2:-}"
if [[ -z "${CONFIG_YAML}" || -z "${BACKBONE}" ]]; then
  echo "Usage: bash scripts/precompute_text_embeds.sh configs/task/<task>.yaml <backbone> [extra args...]" >&2
  exit 2
fi
shift 2

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

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m xiconwam.precompute_text \
  --config-name "task/${TASK}" \
  "model/backbone=${BACKBONE}" \
  "$@"
