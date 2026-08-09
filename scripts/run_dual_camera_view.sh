#!/usr/bin/env bash
set -Eeuo pipefail
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_FILE="${VIOLA_OPERATION_CONFIG:-${REPO_DIR}/config/operation.env}"
[[ -f "${CONFIG_FILE}" ]] && source "${CONFIG_FILE}"
exec conda run --no-capture-output -n "${LEROBOT_ENV_NAME:-lerobot}" python "${SCRIPT_DIR}/dual_camera_view.py" \
  --devices "${VIOLA_FRONT_CAMERA:?Set VIOLA_FRONT_CAMERA}" "${VIOLA_UP_CAMERA:?Set VIOLA_UP_CAMERA}" \
  --fourccs MJPG YUYV --width 640 --height 480 --fps 30
