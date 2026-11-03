#!/usr/bin/env bash
# Download a ModelScope dataset into playground/datasets.
#
#   export MODELSCOPE_API_TOKEN=...
#   bash scripts/io/download_datasets.sh [owner/dataset] [local_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/io/_common.sh
source "${SCRIPT_DIR}/_common.sh"

REPO="${1:-${DEFAULT_DATASET_REPO}}"
LOCAL_DIR="${2:-${DEFAULT_DATASET_DIR}}"
TOKEN="$(resolve_token)"
REVISION="${MODELSCOPE_REVISION:-master}"
MAX_WORKERS="${MODELSCOPE_MAX_WORKERS:-8}"
EXTRA=()
[[ -n "${TOKEN}" ]] && EXTRA+=(--token "${TOKEN}")

require_repo_id "${REPO}"
require_modelscope
ensure_playground_links
mkdir -p "${LOCAL_DIR}"
login_modelscope "${TOKEN}"

echo "download dataset | ${REPO}@${REVISION} -> ${LOCAL_DIR}"
exec modelscope download \
  --dataset "${REPO}" \
  --repo-type dataset \
  "${EXTRA[@]}" \
  --revision "${REVISION}" \
  --local_dir "${LOCAL_DIR}" \
  --max-workers "${MAX_WORKERS}"
