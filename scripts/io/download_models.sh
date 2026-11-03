#!/usr/bin/env bash
# Download ModelScope pretrained weights into playground/pretrained_models.
#
#   export MODELSCOPE_API_TOKEN=...
#   bash scripts/io/download_models.sh [owner/model] [local_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/io/_common.sh
source "${SCRIPT_DIR}/_common.sh"

REPO="${1:-${DEFAULT_MODEL_REPO}}"
LOCAL_DIR="${2:-${DEFAULT_MODEL_DIR}}"
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

echo "download models | ${REPO}@${REVISION} -> ${LOCAL_DIR}"
modelscope download \
  --model "${REPO}" \
  --repo-type model \
  "${EXTRA[@]}" \
  --revision "${REVISION}" \
  --local_dir "${LOCAL_DIR}" \
  --max-workers "${MAX_WORKERS}"

link_action_dit_aliases "${LOCAL_DIR}"
echo "done | models at ${LOCAL_DIR}"
ls -la "${LOCAL_DIR}"
