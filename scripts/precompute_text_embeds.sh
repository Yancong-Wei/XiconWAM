#!/usr/bin/env bash
# The task YAML selects the vision tower and text-embedding cache directory.
#
#   bash scripts/precompute_text_embeds.sh configs/task/robodojo_wan.yaml
#   bash scripts/precompute_text_embeds.sh configs/task/robodojo_cosmos.yaml --overwrite
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_YAML="${1:-}"
if [[ -z "${CONFIG_YAML}" ]]; then
  echo "Usage: bash scripts/precompute_text_embeds.sh configs/task/<task>.yaml [--overwrite]" >&2
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

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m xiconwam.precompute_text --config-name "task/${TASK}" "$@"
