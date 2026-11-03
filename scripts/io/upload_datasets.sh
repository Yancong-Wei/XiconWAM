#!/usr/bin/env bash
# Upload a local dataset directory to ModelScope.
#
#   export MODELSCOPE_API_TOKEN=...
#   bash scripts/io/upload_datasets.sh [local_dir] [owner/dataset]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/io/_common.sh
source "${SCRIPT_DIR}/_common.sh"

LOCAL_DIR="$(readlink -f "${1:-${DEFAULT_DATASET_DIR}}")"
REPO="${2:-${DEFAULT_DATASET_REPO}}"
TOKEN="$(resolve_token)"
MAX_WORKERS="${MODELSCOPE_MAX_WORKERS:-8}"
CHINESE_NAME="${MODELSCOPE_CHINESE_NAME:-RoboDojo dataset}"
DESCRIPTION="${MODELSCOPE_DESCRIPTION:-RoboDojo LeRobot dataset mirror}"
EXTRA=()
[[ -n "${TOKEN}" ]] && EXTRA+=(--token "${TOKEN}")

[[ -e "${LOCAL_DIR}" ]] || { echo "path not found: ${LOCAL_DIR}" >&2; exit 2; }
require_repo_id "${REPO}"
require_modelscope
login_modelscope "${TOKEN}"
create_repo_if_needed "${REPO}" dataset "${TOKEN}" "${CHINESE_NAME}" "${DESCRIPTION}"

echo "upload dataset | ${LOCAL_DIR} -> ${REPO}"
exec modelscope upload "${REPO}" "${LOCAL_DIR}" \
  --repo-type dataset \
  "${EXTRA[@]}" \
  --max-workers "${MAX_WORKERS}" \
  --commit-message "Upload dataset from ${HOSTNAME:-local}"
