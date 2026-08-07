#!/usr/bin/env bash
set -Eeuo pipefail

echo "Disabled: the right-to-left source dataset is frozen and must not be appended to." >&2
echo "Create a new dataset version and a new launcher if more demonstrations are needed." >&2
exit 64

# Historical command retained below for provenance only. This code is unreachable.

readonly PROJECT_DIR="/home/yz/lerobot"
readonly PYTHON_BIN="/home/yz/anaconda3/envs/lerobot/bin/python"
readonly RECORDER="${PROJECT_DIR}/scripts/record_episodes_keep_pose.py"
readonly DEVICE_USER_SCANNER="${PROJECT_DIR}/scripts/find_device_users.py"
readonly DATASET_ROOT="${PROJECT_DIR}/data/recordings/bourn117/viola_cubes_right_to_left_keep_pose_v3_clean"
readonly ROBOT_PORT="/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0"
readonly TELEOP_PORT="/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0"
readonly FRONT_CAMERA="/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0"
readonly UP_CAMERA="/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0"
readonly LOG_DIR="${PROJECT_DIR}/logs/episode_recording"
readonly LOGIN_GROUP="$(id -gn "$(id -un)")"

case "${1:-}" in
  ""|--check) ;;
  *)
    echo "Usage: $0 [--check]" >&2
    exit 2
    ;;
esac

for required_path in \
  "${PYTHON_BIN}" \
  "${RECORDER}" \
  "${DEVICE_USER_SCANNER}" \
  "${DATASET_ROOT}/meta/info.json" \
  "${ROBOT_PORT}" \
  "${TELEOP_PORT}" \
  "${FRONT_CAMERA}" \
  "${UP_CAMERA}"
do
  if [[ ! -e "${required_path}" ]]; then
    echo "Preflight failed: required path is missing: ${required_path}" >&2
    exit 2
  fi
done

for camera_path in "${FRONT_CAMERA}" "${UP_CAMERA}"
do
  resolved_camera="$(readlink -f -- "${camera_path}")"
  # `sg dialout` changes the effective group. On this host, /proc then denies
  # access to file descriptors of same-user desktop processes. Run only this
  # read-only scan under the login group; recording remains in dialout.
  if [[ "$(id -gn)" == "${LOGIN_GROUP}" ]]; then
    camera_users="$("${PYTHON_BIN}" "${DEVICE_USER_SCANNER}" "${resolved_camera}")"
  else
    camera_users="$(
      sg "${LOGIN_GROUP}" -c \
        "exec '${PYTHON_BIN}' '${DEVICE_USER_SCANNER}' '${resolved_camera}'" \
        2>/dev/null
    )"
  fi
  if [[ -n "${camera_users//[[:space:]]/}" ]]; then
    echo "Preflight failed: ${camera_path} is already in use by PID(s):${camera_users}" >&2
    echo "Close the camera viewer or other process using it, then run this launcher again." >&2
    exit 3
  fi
done

if [[ "${1:-}" == "--check" ]]; then
  "${PYTHON_BIN}" "${RECORDER}" --help >/dev/null
  echo "Preflight passed: recorder, dataset, arms, and cameras are present. No hardware was opened."
  exit 0
fi

mkdir -p "${LOG_DIR}"
readonly DATASET_LOCK_FILE="${LOG_DIR}/right_to_left_dataset.lock"
readonly LOG_FILE="${LOG_DIR}/right_to_left_$(date +%Y%m%d_%H%M%S_%N)_$$.log"
echo "Recording log: ${LOG_FILE}"
echo "This launcher records up to 10 episodes. Recording starts automatically after connection."
echo "Between accepted episodes, reset both cubes; tap Right Arrow once during reset when ready."
echo "Each accepted episode is then saved and encoded before the next episode starts."
echo "The saving/encoding pause (typically 20-35 seconds) cannot be skipped safely."

cd "${PROJECT_DIR}"
set +e
/usr/bin/flock \
  --nonblock \
  --close \
  --conflict-exit-code 73 \
  "${DATASET_LOCK_FILE}" \
  /usr/bin/env \
  PYTHONUNBUFFERED=1 \
  OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT=1 \
  LEROBOT_KEEP_POSE_MAX_STEP=3.0 \
  LEROBOT_CAMERA_STARTUP_TIMEOUT_S=12.0 \
  LEROBOT_CAMERA_ASYNC_TIMEOUT_MS=1500 \
  LEROBOT_CAMERA_MAX_FRAME_AGE_MS=250 \
  LEROBOT_STARAI_READ_TIMEOUT_S=1.0 \
  LEROBOT_MIN_EPISODE_TIME_S=5.0 \
  "${PYTHON_BIN}" \
  "${RECORDER}" \
  --robot.type=lerobot_robot_viola \
  --robot.port="${ROBOT_PORT}" \
  --robot.id=my_awesome_staraiviola_arm \
  --robot.disable_torque_on_disconnect=true \
  --robot.use_degrees=false \
  --robot.cameras="{\"front\":{\"type\":\"opencv\",\"index_or_path\":\"${FRONT_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\"},\"up\":{\"type\":\"opencv\",\"index_or_path\":\"${UP_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\"}}" \
  --teleop.type=lerobot_teleoperator_violin \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id=my_awesome_staraiviolin_arm \
  --teleop.use_degrees=false \
  --display_data=true \
  --dataset.repo_id=bourn117/viola_cubes_right_to_left_keep_pose_v3_clean \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=10 \
  --dataset.video=true \
  --dataset.video_encoding_batch_size=1 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Move the blue cube, then the red cube, from the white pad on the right to the gray platform on the left." \
  --play_sounds=true \
  --resume=true \
  2>&1 | tee "${LOG_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e

if [[ "${pipeline_status[0]}" -eq 73 ]]; then
  echo "Preflight failed: another right-to-left recording launcher is already running." >&2
  echo "Wait for it to finish; never run two writers against the same dataset root." >&2
  exit 4
fi
if [[ "${pipeline_status[0]}" -ne 0 ]]; then
  exit "${pipeline_status[0]}"
fi
if [[ "${pipeline_status[1]}" -ne 0 ]]; then
  exit "${pipeline_status[1]}"
fi
