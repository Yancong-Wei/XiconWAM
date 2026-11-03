#!/usr/bin/env bash
# One GPU. GPU_ID selects the device; a comma list keeps only the first id.
#
#   bash scripts/train_single_gpu.sh configs/task/robodojo_a800.yaml
#   GPU_ID=1 bash scripts/train_single_gpu.sh configs/task/robodojo_idm.yaml max_steps=20
set -euo pipefail

RAW_GPU="${GPU_ID:-${GPU_IDS:-0}}"
export GPU_IDS="${RAW_GPU%%,*}"
export NPROC=1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_from_yaml.sh" "$@"
