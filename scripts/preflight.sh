#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="${LEROBOT_PROJECT_DIR:-/home/yz/lerobot}"
readonly PYTHON_BIN="${LEROBOT_PYTHON_BIN:-/home/yz/anaconda3/envs/lerobot/bin/python}"
readonly ROBOT_PORT="${VIOLA_ROBOT_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0}"
readonly TELEOP_PORT="${VIOLA_TELEOP_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0}"
readonly FRONT_CAMERA="${VIOLA_FRONT_CAMERA:-/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0}"
readonly UP_CAMERA="${VIOLA_UP_CAMERA:-/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0}"
readonly CALIBRATION_ROOT="${HF_LEROBOT_CALIBRATION:-/home/yz/.cache/huggingface/lerobot/calibration}"
readonly ROBOT_CALIBRATION="${CALIBRATION_ROOT}/robots/starai_viola/my_awesome_staraiviola_arm.json"
readonly TELEOP_CALIBRATION="${CALIBRATION_ROOT}/teleoperators/starai_violin/my_awesome_staraiviolin_arm.json"
readonly ROBOT_CALIBRATION_SHA256="7e580ce32f1f4b9a37367d42ff1563edccd4d9130ea4e122febd0558517d30ac"
readonly TELEOP_CALIBRATION_SHA256="31d9cbe5471219df2087f4bd104dada86d3279965cee0666837c9dd880764fe7"
readonly EXPECTED_LEROBOT_COMMIT="d9e74a9d374a8f26582ad326c699740a227b483c"

errors=0

fail() {
  echo "FAIL: $*" >&2
  errors=$((errors + 1))
}

pass() {
  echo " OK : $*"
}

check_exists() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    pass "${label}: ${path} -> $(readlink -f -- "${path}")"
  else
    fail "${label} is missing: ${path}"
  fi
}

check_rw() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    return
  fi
  if [[ -r "${path}" && -w "${path}" ]]; then
    pass "${label} is readable and writable by this process"
  else
    fail "${label} exists but this process cannot read and write it: ${path}"
  fi
}

check_hash() {
  local label="$1"
  local path="$2"
  local expected="$3"
  if [[ ! -f "${path}" ]]; then
    return
  fi
  local actual
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  if [[ "${actual}" == "${expected}" ]]; then
    pass "${label} checksum matches ${actual}"
  else
    fail "${label} checksum is ${actual}; expected ${expected}"
  fi
}

echo "StarAI Viola/Violin read-only preflight"
echo "No camera, serial port, motor, or dataset will be opened."
echo

check_exists "LeRobot project" "${PROJECT_DIR}"
check_exists "Python" "${PYTHON_BIN}"
check_exists "Viola follower port" "${ROBOT_PORT}"
check_exists "Violin teacher port" "${TELEOP_PORT}"
check_exists "front camera" "${FRONT_CAMERA}"
check_exists "up camera" "${UP_CAMERA}"
check_exists "Viola follower calibration" "${ROBOT_CALIBRATION}"
check_exists "Violin teacher calibration" "${TELEOP_CALIBRATION}"

check_rw "Viola follower port" "${ROBOT_PORT}"
check_rw "Violin teacher port" "${TELEOP_PORT}"
check_rw "front camera" "${FRONT_CAMERA}"
check_rw "up camera" "${UP_CAMERA}"

check_hash "Viola follower calibration" "${ROBOT_CALIBRATION}" "${ROBOT_CALIBRATION_SHA256}"
check_hash "Violin teacher calibration" "${TELEOP_CALIBRATION}" "${TELEOP_CALIBRATION_SHA256}"

if [[ -d "${PROJECT_DIR}/.git" ]]; then
  actual_commit="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${actual_commit}" == "${EXPECTED_LEROBOT_COMMIT}" ]]; then
    pass "LeRobot checkout is the validated commit ${actual_commit}"
  elif [[ "${VIOLA_ALLOW_UNVALIDATED_SOFTWARE:-0}" == "1" ]]; then
    echo "WARN: LeRobot commit ${actual_commit:-unknown} is not the validated commit; override accepted."
  else
    fail "LeRobot commit is ${actual_commit:-unknown}; expected ${EXPECTED_LEROBOT_COMMIT}"
  fi
else
  fail "LeRobot project is not a Git checkout: ${PROJECT_DIR}"
fi

if [[ -x "${PYTHON_BIN}" ]]; then
  if "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import importlib
import sys

expected = {
    "av": "15.1.0",
    "fashionstar_uart_sdk": "1.3.8",
    "lerobot": "0.4.2",
    "lerobot_motor_starai": "0.0.4",
    "lerobot_robot_viola": "0.0.4",
    "lerobot_teleoperator_violin": "0.0.4",
    "numpy": "2.2.6",
    "opencv-python-headless": "4.12.0.88",
    "pillow": "12.0.0",
    "pyarrow": "22.0.0",
    "pynput": "1.8.1",
    "rerun-sdk": "0.26.2",
    "torch": "2.7.1",
    "torchcodec": "0.5",
    "torchvision": "0.22.1",
}
errors = []
for distribution, wanted in expected.items():
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        errors.append(f"{distribution} is not installed")
        continue
    if actual != wanted:
        errors.append(f"{distribution}=={actual}, expected {wanted}")
for module in (
    "av",
    "cv2",
    "lerobot",
    "lerobot_robot_viola",
    "lerobot_teleoperator_violin",
    "PIL",
    "pyarrow",
    "rerun",
    "tkinter",
):
    try:
        importlib.import_module(module)
    except Exception as error:
        errors.append(f"cannot import {module}: {error}")
if errors:
    print("; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("Python package versions match the validated setup")
PY
  then
    pass "Python imports and critical package versions"
  elif [[ "${VIOLA_ALLOW_UNVALIDATED_SOFTWARE:-0}" == "1" ]]; then
    echo "WARN: critical Python versions differ; override accepted."
  else
    fail "critical Python imports or versions do not match"
  fi
fi

echo
if (( errors > 0 )); then
  echo "Preflight failed with ${errors} problem(s)." >&2
  echo "For serial permission errors, use the documented sg dialout command." >&2
  exit 1
fi

echo "Preflight passed. Hardware was not opened."
