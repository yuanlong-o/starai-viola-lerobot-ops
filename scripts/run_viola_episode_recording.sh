#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_FILE="${VIOLA_OPERATION_CONFIG:-${REPO_DIR}/config/operation.env}"
[[ -f "${CONFIG_FILE}" ]] && source "${CONFIG_FILE}"

direction="${1:-}"
check="${2:-}"
[[ "${direction}" == "right-to-left" || "${direction}" == "left-to-right" ]] || {
  echo "Usage: $0 {right-to-left|left-to-right} [--check]" >&2; exit 2;
}
[[ -z "${check}" || "${check}" == "--check" ]] || { echo "Unknown option: ${check}" >&2; exit 2; }

readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
env_prefix="$(conda env list --json | python -c 'import json,sys; d=json.load(sys.stdin); n=sys.argv[1]; print(next((p for p in d["envs"] if p.rsplit("/",1)[-1] == n), ""))' "${ENV_NAME}")"
python_bin="${env_prefix}/bin/python"
[[ -x "${python_bin}" ]] || { echo "Environment Python not found for ${ENV_NAME}" >&2; exit 2; }

robot_port="${VIOLA_ROBOT_PORT:?Set VIOLA_ROBOT_PORT in config/operation.env}"
teleop_port="${VIOLA_TELEOP_PORT:?Set VIOLA_TELEOP_PORT in config/operation.env}"
front_camera="${VIOLA_FRONT_CAMERA:?Set VIOLA_FRONT_CAMERA in config/operation.env}"
up_camera="${VIOLA_UP_CAMERA:?Set VIOLA_UP_CAMERA in config/operation.env}"
robot_id="${VIOLA_ROBOT_ID:-my_awesome_staraiviola_arm}"
teleop_id="${VIOLA_TELEOP_ID:-my_awesome_staraiviolin_arm}"
dataset_dir="${VIOLA_DATASET_DIR:-${HOME}/lerobot-data}"
namespace="${VIOLA_DATASET_NAMESPACE:-local}"

if [[ "${direction}" == "right-to-left" ]]; then
  slug="viola_cubes_right_to_left_keep_pose_v1"
  task="Move the blue cube, then the red cube, from the white pad on the right to the gray platform on the left."
else
  slug="viola_cubes_left_to_right_keep_pose_v1"
  task="Move the blue cube, then the red cube, from the gray platform on the left to the white pad on the right."
fi
repo_id="${namespace}/${slug}"
root="${dataset_dir}/${namespace}/${slug}"
resume=false
[[ -f "${root}/meta/info.json" ]] && resume=true
num_episodes=1
[[ "${resume}" == true ]] && num_episodes=10

for path in "${robot_port}" "${teleop_port}" "${front_camera}" "${up_camera}"; do
  [[ -e "${path}" ]] || { echo "Preflight failed: missing ${path}" >&2; exit 2; }
done
"${SCRIPT_DIR}/install_calibrations.sh" --check >/dev/null
"${python_bin}" "${SCRIPT_DIR}/record_episodes_keep_pose.py" --help >/dev/null
if [[ "${check}" == "--check" ]]; then
  echo "Preflight passed: recorder, calibrations, arms, and cameras are present. No hardware was opened."
  exit 0
fi

mkdir -p "$(dirname -- "${root}")" "${REPO_DIR}/logs/episode_recording"
lock="${REPO_DIR}/logs/episode_recording/${slug}.lock"
log="${REPO_DIR}/logs/episode_recording/${slug}_$(date +%Y%m%d_%H%M%S).log"
camera_json="{\"front\":{\"type\":\"opencv\",\"index_or_path\":\"${front_camera}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\",\"warmup_s\":8},\"up\":{\"type\":\"opencv\",\"index_or_path\":\"${up_camera}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"YUYV\",\"warmup_s\":8}}"

exec /usr/bin/flock --nonblock "${lock}" sg dialout -c "exec env PYTHONUNBUFFERED=1 LEROBOT_KEEP_POSE_MAX_STEP='${VIOLA_MAX_STEP:-3.0}' '${python_bin}' '${SCRIPT_DIR}/record_episodes_keep_pose.py' \
  --robot.type=lerobot_robot_viola --robot.port='${robot_port}' --robot.id='${robot_id}' --robot.disable_torque_on_disconnect=true --robot.use_degrees=false --robot.cameras='${camera_json}' \
  --teleop.type=lerobot_teleoperator_violin --teleop.port='${teleop_port}' --teleop.id='${teleop_id}' --teleop.use_degrees=false \
  --display_data=true --dataset.repo_id='${repo_id}' --dataset.root='${root}' --dataset.fps=30 --dataset.episode_time_s=60 --dataset.reset_time_s=60 \
  --dataset.num_episodes='${num_episodes}' --dataset.video=true --dataset.video_encoding_batch_size=1 --dataset.push_to_hub=false --dataset.single_task='${task}' --play_sounds=true --resume='${resume}' 2>&1 | tee '${log}'"
