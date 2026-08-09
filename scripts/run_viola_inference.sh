#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_FILE="${VIOLA_OPERATION_CONFIG:-${REPO_DIR}/config/operation.env}"
[[ -f "${CONFIG_FILE}" ]] && source "${CONFIG_FILE}"

readonly RUN_SECONDS="${1:-10}"
readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
readonly ROBOT_PORT="${VIOLA_ROBOT_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0}"
readonly FRONT_CAMERA="${VIOLA_FRONT_CAMERA:-/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0}"
readonly UP_CAMERA="${VIOLA_UP_CAMERA:-/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0}"
readonly ROBOT_ID="${VIOLA_ROBOT_ID:-my_awesome_staraiviola_arm}"
readonly POLICY_DIR="${VIOLA_POLICY_DIR:-${HOME}/models/act_viola_val20_step080000}"
readonly CUDA_DEVICE="${VIOLA_CUDA_DEVICE:-0}"
readonly MAX_STEP="${VIOLA_MAX_STEP:-3.0}"

if [[ ! "${RUN_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${RUN_SECONDS}" == "0" ]]; then
  echo "Usage: $0 [positive-duration-seconds]" >&2
  exit 2
fi

if [[ -n "${LEROBOT_PYTHON_BIN:-}" ]]; then
  python_bin="${LEROBOT_PYTHON_BIN}"
elif command -v conda >/dev/null 2>&1; then
  env_prefix="$(conda env list --json | python -c 'import json,sys; data=json.load(sys.stdin); name=sys.argv[1]; print(next((p for p in data["envs"] if p.rsplit("/",1)[-1] == name), ""))' "${ENV_NAME}")"
  python_bin="${env_prefix}/bin/python"
else
  python_bin="${HOME}/anaconda3/envs/${ENV_NAME}/bin/python"
fi

for required in "${python_bin}" "${POLICY_DIR}/model.safetensors" "${ROBOT_PORT}" "${FRONT_CAMERA}" "${UP_CAMERA}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 3
  fi
done

exec sg dialout -c "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} exec '${python_bin}' \
  '${SCRIPT_DIR}/infer_keep_pose.py' \
  --max_step=${MAX_STEP} \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path='${POLICY_DIR}' \
  --robot.type=lerobot_robot_viola \
  --robot.port='${ROBOT_PORT}' \
  --robot.id='${ROBOT_ID}' \
  --robot.disable_torque_on_disconnect=true \
  --robot.use_degrees=false \
  --robot.cameras='{\"front\":{\"type\":\"opencv\",\"index_or_path\":\"${FRONT_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\",\"warmup_s\":8},\"up\":{\"type\":\"opencv\",\"index_or_path\":\"${UP_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"YUYV\",\"warmup_s\":8}}' \
  --fps=30 \
  --duration=${RUN_SECONDS} \
  --device=cuda \
  --task='move the blue cube to the gray platform before the red cube' \
  --display_data=true \
  --display_mode=rerun \
  --play_sounds=false"
