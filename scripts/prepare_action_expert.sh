#!/usr/bin/env bash
# The task YAML selects the vision tower and Action DiT output path.
#
#   bash scripts/prepare_action_expert.sh configs/task/robodojo_wan.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_YAML="${1:-}"
if [[ -z "${CONFIG_YAML}" || $# -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_action_expert.sh configs/task/<task>.yaml" >&2
  exit 2
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

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m xiconwam.prepare_action_expert --config-name "task/${TASK}"
