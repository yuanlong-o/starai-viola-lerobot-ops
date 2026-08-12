#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_FILE="${VIOLA_OPERATION_CONFIG:-${REPO_DIR}/config/operation.env}"
[[ -f "${CONFIG_FILE}" ]] && source "${CONFIG_FILE}"

readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
readonly ROBOT_PORT="${VIOLA_ROBOT_PORT:?Set VIOLA_ROBOT_PORT in config/operation.env}"
readonly TELEOP_PORT="${VIOLA_TELEOP_PORT:?Set VIOLA_TELEOP_PORT in config/operation.env}"
readonly FRONT_CAMERA="${VIOLA_FRONT_CAMERA:?Set VIOLA_FRONT_CAMERA in config/operation.env}"
readonly UP_CAMERA="${VIOLA_UP_CAMERA:?Set VIOLA_UP_CAMERA in config/operation.env}"
readonly ROBOT_ID="${VIOLA_ROBOT_ID:-my_awesome_staraiviola_arm}"
readonly TELEOP_ID="${VIOLA_TELEOP_ID:-my_awesome_staraiviolin_arm}"
readonly MAX_STEP="${VIOLA_MAX_STEP:-3.0}"

if [[ -n "${LEROBOT_PYTHON_BIN:-}" ]]; then
  python_bin="${LEROBOT_PYTHON_BIN}"
else
  env_prefix="$(conda env list --json | python -c 'import json,sys; data=json.load(sys.stdin); name=sys.argv[1]; print(next((p for p in data["envs"] if p.rsplit("/",1)[-1] == name), ""))' "${ENV_NAME}")"
  python_bin="${env_prefix}/bin/python"
fi
[[ -x "${python_bin}" ]] || { echo "Environment Python not found: ${python_bin}" >&2; exit 2; }

exec sg dialout -c "exec '${python_bin}' '${SCRIPT_DIR}/teleoperate_keep_pose.py' \
  --robot.type=lerobot_robot_viola \
  --robot.port='${ROBOT_PORT}' \
  --robot.id='${ROBOT_ID}' \
  --robot.disable_torque_on_disconnect=true \
  --robot.use_degrees=false \
  --robot.cameras='{\"front\":{\"type\":\"opencv\",\"index_or_path\":\"${FRONT_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\",\"warmup_s\":8},\"up\":{\"type\":\"opencv\",\"index_or_path\":\"${UP_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"YUYV\",\"warmup_s\":8}}' \
  --teleop.type=lerobot_teleoperator_violin \
  --teleop.port='${TELEOP_PORT}' \
  --teleop.id='${TELEOP_ID}' \
  --teleop.use_degrees=false \
  --fps=30 --max_step=${MAX_STEP} --display_data=true"
