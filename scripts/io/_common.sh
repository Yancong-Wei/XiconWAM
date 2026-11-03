#!/usr/bin/env bash
# Shared helpers for ModelScope dataset/model IO under scripts/io/.
# Auth: export MODELSCOPE_API_TOKEN=...  (never hardcode tokens in these scripts)
# shellcheck disable=SC2034

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAYGROUND="${ROOT}/playground"

# Public defaults stay empty; set MODELSCOPE_*_REPO or pass owner/name on the CLI.
DEFAULT_MODEL_REPO="${MODELSCOPE_MODEL_REPO:-}"
DEFAULT_DATASET_REPO="${MODELSCOPE_DATASET_REPO:-}"
DEFAULT_MODEL_DIR="${MODELSCOPE_MODEL_DIR:-${PLAYGROUND}/pretrained_models}"
DEFAULT_DATASET_DIR="${MODELSCOPE_DATASET_DIR:-${PLAYGROUND}/datasets/robodojo}"
DEFAULT_VISIBILITY="${MODELSCOPE_VISIBILITY:-private}"

require_repo_id() {
  local repo="$1"
  [[ -n "${repo}" ]] || {
    echo "repo is required: pass owner/name or export MODELSCOPE_MODEL_REPO / MODELSCOPE_DATASET_REPO" >&2
    exit 2
  }
  [[ "${repo}" =~ ^[^/]+/[^/]+$ ]] || {
    echo "repo must be owner/name, got: ${repo}" >&2
    exit 2
  }
}

# Prefer export MODELSCOPE_API_TOKEN=...; fall back to local modelscope credentials.
resolve_token() {
  local token="${MODELSCOPE_API_TOKEN:-}"
  if [[ -n "${token}" ]]; then
    printf '%s' "${token}"
    return 0
  fi
  if [[ -d "${HOME}/.modelscope/credentials" ]]; then
    echo "MODELSCOPE_API_TOKEN unset; using local modelscope credentials" >&2
    printf ''
    return 0
  fi
  echo "export MODELSCOPE_API_TOKEN=... first (or login once via modelscope login)" >&2
  exit 2
}

require_modelscope() {
  if ! command -v modelscope >/dev/null 2>&1; then
    echo "modelscope CLI not found. Install: pip install 'modelscope==1.34.0' 'setuptools<81'" >&2
    exit 1
  fi
  if [[ -z "${OMP_NUM_THREADS:-}" ]] || ! [[ "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    unset OMP_NUM_THREADS || true
  fi
}

login_modelscope() {
  local token="$1"
  [[ -n "${token}" ]] || return 0
  export MODELSCOPE_API_TOKEN="${token}"
  echo "login modelscope"
  modelscope login --token "${token}"
}

ensure_playground_links() {
  if [[ ! -e "${PLAYGROUND}/datasets" || ! -e "${PLAYGROUND}/pretrained_models" ]]; then
    echo "playground links missing under ${PLAYGROUND}" >&2
    echo "example:" >&2
    echo "  ln -sfn /data/xiconwam/datasets ${PLAYGROUND}/datasets" >&2
    echo "  ln -sfn /data/xiconwam/pretrained_models ${PLAYGROUND}/pretrained_models" >&2
    exit 1
  fi
}

create_repo_if_needed() {
  local repo="$1"
  local repo_type="$2"
  local token="$3"
  local chinese_name="${4:-}"
  local description="${5:-}"
  local args=(create "${repo}" --repo_type "${repo_type}" --visibility "${DEFAULT_VISIBILITY}")
  [[ -n "${token}" ]] && args+=(--token "${token}")
  [[ -n "${chinese_name}" ]] && args+=(--chinese_name "${chinese_name}")
  [[ -n "${description}" ]] && args+=(--description "${description}")
  echo "ensure ${repo_type} repo | ${repo}"
  modelscope "${args[@]}" || echo "create skipped/non-zero; continuing with upload to ${repo}"
}

link_action_dit_aliases() {
  local root="$1"
  if [[ -f "${root}/ActionDiT_Wan22_5b.pt" && ! -e "${root}/ActionDiT_Wan22_5B_alphascale_1024hdim.pt" ]]; then
    ln -sfn ActionDiT_Wan22_5b.pt "${root}/ActionDiT_Wan22_5B_alphascale_1024hdim.pt"
  elif [[ -f "${root}/ActionDiT_Wan22_5B_alphascale_1024hdim.pt" && ! -e "${root}/ActionDiT_Wan22_5b.pt" ]]; then
    ln -sfn ActionDiT_Wan22_5B_alphascale_1024hdim.pt "${root}/ActionDiT_Wan22_5b.pt"
  fi
  if [[ -f "${root}/ActionDiT_Cosmos25_2b.pt" && ! -e "${root}/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt" ]]; then
    ln -sfn ActionDiT_Cosmos25_2b.pt "${root}/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt"
  elif [[ -f "${root}/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt" && ! -e "${root}/ActionDiT_Cosmos25_2b.pt" ]]; then
    ln -sfn ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt "${root}/ActionDiT_Cosmos25_2b.pt"
  fi
}
