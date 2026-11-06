#!/usr/bin/env bash
# RoboDojo closed-loop entry for XiconWAM.
#
# Usage (matches the usual 10-arg RoboDojo policy-dir contract):
#   bash eval/XPolicyLab_core/eval.sh \
#     RoboDojo <task> <ckpt> arx_x5 joint <seed> <policy_gpu> <env_gpu> \
#     <policy_python_or_venv> <eval_env>
#
# Offline wiring check (no Isaac / RoboDojo):
#   EVAL_ENV_TYPE=debug bash eval/XPolicyLab_core/eval.sh \
#     RoboDojo stack_bowls step_015000 arx_x5 joint 0 0 0 \
#     .venv/bin/python unused
#
# Simulation defaults to the vendored checkout at eval/RoboDojo. Override with
# ROBODOJO_ROOT only if you keep a separate install.
set -euo pipefail

bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
policy_gpu_id=${7:?policy_gpu_id}
env_gpu_id=${8:?env_gpu_id}
policy_python_arg=${9:?policy_python_or_venv}
eval_env_conda_env=${10:?eval_env}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML_FILE="${SCRIPT_DIR}/deploy.yml"
SERVER_SCRIPT="${SCRIPT_DIR}/server.py"
DEBUG_CLIENT="${SCRIPT_DIR}/debug_client.py"

resolve_python() {
  local candidate="${1}"
  if [[ -z "${candidate}" ]]; then
    :
  elif [[ -x "${candidate}" && -f "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  elif [[ -x "${candidate}/bin/python" ]]; then
    printf '%s\n' "${candidate}/bin/python"
    return 0
  elif [[ -x "${REPO_ROOT}/${candidate}" && -f "${REPO_ROOT}/${candidate}" ]]; then
    printf '%s\n' "${REPO_ROOT}/${candidate}"
    return 0
  elif [[ -x "${REPO_ROOT}/${candidate}/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/${candidate}/bin/python"
    return 0
  fi
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
    return 0
  fi
  command -v python3
}

POLICY_PYTHON="$(resolve_python "${policy_python_arg}")"
if [[ ! -x "${POLICY_PYTHON}" ]]; then
  echo "[MAIN] policy python not executable: ${POLICY_PYTHON}" >&2
  exit 1
fi

if [[ "${action_type}" != "joint" ]]; then
  echo "[MAIN] XiconWAM RoboDojo eval requires action_type=joint, got ${action_type}" >&2
  exit 1
fi

policy_server_port="$(
  "${POLICY_PYTHON}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
policy_server_ip="127.0.0.1"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    echo "[MAIN] kill server ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MAIN] start server port=${policy_server_port} python=${POLICY_PYTHON}"

export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

OVERRIDE_ARGS=(
  port="${policy_server_port}"
  host="0.0.0.0"
  bench_name="${bench_name}"
  task_name="${task_name}"
  ckpt_name="${ckpt_name}"
  env_cfg_type="${env_cfg_type}"
  seed="${seed}"
  action_type="${action_type}"
  policy_name="XiconWAM"
)

if [[ -n "${XICON_DRY_ZEROS:-}" ]]; then
  OVERRIDE_ARGS+=(dry_zeros=true)
fi

env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  "${POLICY_PYTHON}" "${SERVER_SCRIPT}" \
    --config_path "${YAML_FILE}" \
    --overrides "${OVERRIDE_ARGS[@]}" &
SERVER_PID=$!

"${POLICY_PYTHON}" - <<PY
import socket
import sys
import time

host = "${policy_server_ip}"
port = int("${policy_server_port}")
deadline = time.time() + 1200
while time.time() < deadline:
    # TCP accept only: the websocket handshake is done by the real client.
    # Closing immediately after connect can log a benign handshake error on
    # the server, so wait briefly once the port is open.
    try:
        with socket.create_connection((host, port), timeout=2.0):
            time.sleep(0.5)
            print(f"[MAIN] policy server ready on {host}:{port}", flush=True)
            sys.exit(0)
    except OSError:
        time.sleep(1.0)
print(f"[MAIN] timed out waiting for policy server {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[MAIN] policy server exited early" >&2
  exit 1
fi

eval_env_mode="${EVAL_ENV_TYPE:-sim}"
case "${eval_env_mode}" in
  debug)
    echo "[MAIN] EVAL_ENV_TYPE=debug -> offline wiring check"
    "${POLICY_PYTHON}" "${DEBUG_CLIENT}" \
      --host "${policy_server_ip}" \
      --port "${policy_server_port}" \
      --replan-steps "${XICON_REPLAN_STEPS:-10}"
    ;;
  sim)
    ROBODOJO_ROOT="${ROBODOJO_ROOT:-${SCRIPT_DIR}/../RoboDojo}"
    ROBODOJO_ROOT="$(cd "${ROBODOJO_ROOT}" && pwd)"
    EVAL_SCRIPT="${ROBODOJO_ROOT}/scripts/eval_policy.sh"
    if [[ ! -f "${EVAL_SCRIPT}" ]]; then
      echo "[MAIN] missing ${EVAL_SCRIPT}" >&2
      echo "[MAIN] expected vendored checkout at eval/RoboDojo" >&2
      exit 1
    fi
    additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
    echo "[MAIN] start RoboDojo client | root=${ROBODOJO_ROOT}"
    if command -v conda >/dev/null 2>&1; then
      # shellcheck disable=SC1091
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda deactivate >/dev/null 2>&1 || true
      conda activate "${eval_env_conda_env}"
    fi
    bash "${EVAL_SCRIPT}" \
      --task_name "${task_name}" \
      --env_cfg_type "${env_cfg_type}" \
      --policy_name "XiconWAM" \
      --host "${policy_server_ip}" \
      --port "${policy_server_port}" \
      --protocol ws \
      --eval_batch false \
      --root_dir "${ROBODOJO_ROOT}" \
      --device_id "${env_gpu_id}" \
      --additional_info "${additional_info}" \
      --seed "${seed}"
    ;;
  *)
    echo "[MAIN] unsupported EVAL_ENV_TYPE=${eval_env_mode} (use debug or sim)" >&2
    exit 2
    ;;
esac

echo "[MAIN] eval finished"
