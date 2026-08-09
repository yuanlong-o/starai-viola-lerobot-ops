#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_FILE="${VIOLA_OPERATION_CONFIG:-${REPO_DIR}/config/operation.env}"
[[ -f "${CONFIG_FILE}" ]] && source "${CONFIG_FILE}"

readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
readonly ROBOT_PORT="${VIOLA_ROBOT_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0}"
readonly TELEOP_PORT="${VIOLA_TELEOP_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0}"
readonly FRONT_CAMERA="${VIOLA_FRONT_CAMERA:-/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0}"
readonly UP_CAMERA="${VIOLA_UP_CAMERA:-/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0}"
readonly POLICY_DIR="${VIOLA_POLICY_DIR:-${HOME}/models/act_viola_val20_step080000}"
readonly CALIBRATION_ROOT="${HF_LEROBOT_CALIBRATION:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot/calibration}"
readonly ROBOT_CALIBRATION="${CALIBRATION_ROOT}/robots/starai_viola/my_awesome_staraiviola_arm.json"
readonly TELEOP_CALIBRATION="${CALIBRATION_ROOT}/teleoperators/starai_violin/my_awesome_staraiviolin_arm.json"
readonly ROBOT_CALIBRATION_SHA256="7e580ce32f1f4b9a37367d42ff1563edccd4d9130ea4e122febd0558517d30ac"
readonly TELEOP_CALIBRATION_SHA256="31d9cbe5471219df2087f4bd104dada86d3279965cee0666837c9dd880764fe7"
readonly MODEL_SHA256="1093aaeddfb902e7e596425d87676baba58cb8ab617a52c954ec11940726b886"

if [[ -n "${LEROBOT_PYTHON_BIN:-}" ]]; then
  python_bin="${LEROBOT_PYTHON_BIN}"
elif command -v conda >/dev/null 2>&1; then
  env_prefix="$(conda env list --json | python -c 'import json,sys; data=json.load(sys.stdin); name=sys.argv[1]; print(next((p for p in data["envs"] if p.rsplit("/",1)[-1] == name), ""))' "${ENV_NAME}")"
  python_bin="${env_prefix}/bin/python"
else
  python_bin="${HOME}/anaconda3/envs/${ENV_NAME}/bin/python"
fi

errors=0
fail() { echo "FAIL: $*" >&2; errors=$((errors + 1)); }
pass() { echo " OK : $*"; }
check_exists() { [[ -e "$2" ]] && pass "$1: $2 -> $(readlink -f -- "$2")" || fail "$1 is missing: $2"; }
check_rw() { [[ ! -e "$2" ]] || { [[ -r "$2" && -w "$2" ]] && pass "$1 is readable/writable" || fail "$1 is not readable/writable: $2"; }; }
check_hash() {
  [[ ! -f "$2" ]] || {
    actual="$(sha256sum -- "$2" | awk '{print $1}')"
    [[ "${actual}" == "$3" ]] && pass "$1 checksum matches" || fail "$1 checksum ${actual}; expected $3"
  }
}

echo "StarAI Viola operation/inference read-only preflight"
echo "No camera, serial port, motor, or policy inference will be opened."
echo
check_exists "Operations repository" "${REPO_DIR}"
check_exists "Python" "${python_bin}"
check_exists "Viola follower port" "${ROBOT_PORT}"
check_exists "Violin teacher port" "${TELEOP_PORT}"
check_exists "front camera" "${FRONT_CAMERA}"
check_exists "up camera" "${UP_CAMERA}"
check_exists "Viola calibration" "${ROBOT_CALIBRATION}"
check_exists "Violin calibration" "${TELEOP_CALIBRATION}"
check_exists "ACT checkpoint" "${POLICY_DIR}/model.safetensors"
check_rw "Viola follower port" "${ROBOT_PORT}"
check_rw "Violin teacher port" "${TELEOP_PORT}"
check_rw "front camera" "${FRONT_CAMERA}"
check_rw "up camera" "${UP_CAMERA}"
check_hash "Viola calibration" "${ROBOT_CALIBRATION}" "${ROBOT_CALIBRATION_SHA256}"
check_hash "Violin calibration" "${TELEOP_CALIBRATION}" "${TELEOP_CALIBRATION_SHA256}"
check_hash "ACT model" "${POLICY_DIR}/model.safetensors" "${MODEL_SHA256}"

if [[ -x "${python_bin}" ]]; then
  if "${python_bin}" - <<'PY'
import importlib
from importlib.metadata import PackageNotFoundError, version
import sys

expected = {
    "fashionstar_uart_sdk": "1.3.12",
    "huggingface-hub": "1.27.0",
    "lerobot": "0.6.1",
    "lerobot_motor_starai": "0.0.4",
    "lerobot_robot_viola": "0.0.4",
    "lerobot_teleoperator_violin": "0.0.4",
    "rerun-sdk": "0.26.2",
    "torch": "2.7.1",
    "transformers": "5.5.4",
    "wandb": "0.27.2",
}
errors = []
for package, wanted in expected.items():
    try:
        actual = version(package)
    except PackageNotFoundError:
        errors.append(f"{package} is not installed")
        continue
    if actual != wanted:
        errors.append(f"{package}=={actual}, expected {wanted}")
for module in ("cv2", "lerobot", "lerobot_robot_viola", "rerun", "torch", "transformers"):
    try:
        importlib.import_module(module)
    except Exception as error:
        errors.append(f"cannot import {module}: {error}")
if errors:
    print("; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("Python operation/inference package versions match")
PY
  then pass "Python imports and critical versions"; else fail "Python environment mismatch"; fi
fi

echo
if (( errors > 0 )); then
  echo "Preflight failed with ${errors} problem(s)." >&2
  exit 1
fi
echo "Preflight passed. Hardware was not opened."
