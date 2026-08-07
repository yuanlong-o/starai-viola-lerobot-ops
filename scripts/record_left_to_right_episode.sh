#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="${LEROBOT_PROJECT_DIR:-/home/yz/lerobot}"
readonly PYTHON_BIN="${LEROBOT_PYTHON_BIN:-/home/yz/anaconda3/envs/lerobot/bin/python}"
readonly RECORDER="${SCRIPT_DIR}/record_episodes_keep_pose.py"
readonly DEVICE_USER_SCANNER="${SCRIPT_DIR}/find_device_users.py"
readonly DATASET_ROOT="${VIOLA_LTR_DATASET_ROOT:-${PROJECT_DIR}/data/recordings/bourn117/viola_cubes_left_to_right_keep_pose_v3}"
readonly EXPECTED_TASK="Move the blue cube, then the red cube, from the gray platform on the left to the white pad on the right."
readonly ROBOT_PORT="${VIOLA_ROBOT_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0}"
readonly TELEOP_PORT="${VIOLA_TELEOP_PORT:-/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0}"
readonly FRONT_CAMERA="${VIOLA_FRONT_CAMERA:-/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0}"
readonly UP_CAMERA="${VIOLA_UP_CAMERA:-/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0}"
readonly CALIBRATION_ROOT="${HF_LEROBOT_CALIBRATION:-/home/yz/.cache/huggingface/lerobot/calibration}"
readonly ROBOT_CALIBRATION="${CALIBRATION_ROOT}/robots/starai_viola/my_awesome_staraiviola_arm.json"
readonly TELEOP_CALIBRATION="${CALIBRATION_ROOT}/teleoperators/starai_violin/my_awesome_staraiviolin_arm.json"
readonly ROBOT_CALIBRATION_SHA256="7e580ce32f1f4b9a37367d42ff1563edccd4d9130ea4e122febd0558517d30ac"
readonly TELEOP_CALIBRATION_SHA256="31d9cbe5471219df2087f4bd104dada86d3279965cee0666837c9dd880764fe7"
readonly LOG_DIR="${VIOLA_RECORDING_LOG_DIR:-${PROJECT_DIR}/logs/episode_recording}"
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
  "$(dirname -- "${DATASET_ROOT}")" \
  "${ROBOT_PORT}" \
  "${TELEOP_PORT}" \
  "${FRONT_CAMERA}" \
  "${UP_CAMERA}" \
  "${ROBOT_CALIBRATION}" \
  "${TELEOP_CALIBRATION}"
do
  if [[ ! -e "${required_path}" ]]; then
    echo "Preflight failed: required path is missing: ${required_path}" >&2
    exit 2
  fi
done

for serial_path in "${ROBOT_PORT}" "${TELEOP_PORT}"
do
  if [[ ! -r "${serial_path}" || ! -w "${serial_path}" ]]; then
    echo "Preflight failed: this process cannot read and write ${serial_path}." >&2
    echo "Run the documented sg dialout command, or log out and back in after joining dialout." >&2
    exit 2
  fi
done

for camera_path in "${FRONT_CAMERA}" "${UP_CAMERA}"
do
  if [[ ! -r "${camera_path}" || ! -w "${camera_path}" ]]; then
    echo "Preflight failed: this process cannot read and write ${camera_path}." >&2
    echo "Check the desktop ACL or video-group membership; do not use chmod 777." >&2
    exit 2
  fi
done

check_calibration_hash() {
  local calibration_path="$1"
  local expected_hash="$2"
  local device_name="$3"
  local actual_hash
  actual_hash="$(sha256sum -- "${calibration_path}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "Preflight failed: ${device_name} calibration changed after this dataset version was defined." >&2
    echo "Do not mix calibrations. Create a new left-to-right dataset version before recording." >&2
    exit 5
  fi
}

check_calibration_hash "${ROBOT_CALIBRATION}" "${ROBOT_CALIBRATION_SHA256}" "Viola follower"
check_calibration_hash "${TELEOP_CALIBRATION}" "${TELEOP_CALIBRATION_SHA256}" "Violin teacher"

RESUME_VALUE="false"
DATASET_SUMMARY="new dataset"
if [[ -e "${DATASET_ROOT}" ]]; then
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Preflight failed: dataset root exists but is not a directory: ${DATASET_ROOT}" >&2
    exit 5
  fi
  if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Preflight failed: left-to-right root exists without meta/info.json: ${DATASET_ROOT}" >&2
    echo "Preserve it for diagnosis; do not record over an unknown or partial directory." >&2
    exit 5
  fi

  DATASET_SUMMARY="$(${PYTHON_BIN} - "${DATASET_ROOT}" "${EXPECTED_TASK}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root = Path(sys.argv[1])
expected_task = sys.argv[2]
errors: list[str] = []

with (root / "meta/info.json").open() as stream:
    info = json.load(stream)

total_episodes = int(info.get("total_episodes", -1))
total_frames = int(info.get("total_frames", -1))
if total_episodes <= 0 or total_frames <= 0:
    errors.append(
        "the existing root has no finalized episode; it is probably an interrupted first-time initialization"
    )
if info.get("fps") != 30:
    errors.append(f"fps is {info.get('fps')!r}, expected 30")
if info.get("robot_type") != "starai_viola":
    errors.append(f"robot_type is {info.get('robot_type')!r}, expected 'starai_viola'")

features = info.get("features", {})
for key in ("action", "observation.state"):
    if features.get(key, {}).get("shape") != [7]:
        errors.append(f"{key} is missing or does not have shape [7]")
for key in ("observation.images.front", "observation.images.up"):
    feature = features.get(key, {})
    if feature.get("shape") != [480, 640, 3]:
        errors.append(f"{key} is missing or does not have shape [480, 640, 3]")
    if feature.get("info", {}).get("video.fps") != 30:
        errors.append(f"{key} is not 30 fps")

task_path = root / "meta/tasks.parquet"
episode_paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
if not task_path.is_file():
    errors.append("meta/tasks.parquet is missing")
else:
    task_table = pq.read_table(task_path)
    task_column = "__index_level_0__"
    tasks = set(task_table[task_column].to_pylist()) if task_column in task_table.column_names else set()
    if tasks != {expected_task}:
        errors.append(f"task labels are {sorted(tasks)!r}, expected only {expected_task!r}")

if not episode_paths:
    errors.append("episode metadata parquet files are missing")
else:
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_paths])
    episode_indices = [int(value) for value in episodes["episode_index"].to_pylist()]
    lengths = [int(value) for value in episodes["length"].to_pylist()]
    if episode_indices != list(range(total_episodes)):
        errors.append("episode indices are not contiguous from 0 to total_episodes - 1")
    if sum(lengths) != total_frames:
        errors.append(f"episode lengths sum to {sum(lengths)}, but info.json reports {total_frames}")
    episode_tasks = episodes["tasks"].to_pylist()
    if any(tasks != [expected_task] for tasks in episode_tasks):
        errors.append("one or more episode task labels do not match the left-to-right task")

    referenced_files: set[Path] = set()
    for chunk_index, file_index in zip(
        episodes["data/chunk_index"].to_pylist(), episodes["data/file_index"].to_pylist(), strict=True
    ):
        referenced_files.add(root / f"data/chunk-{int(chunk_index):03d}/file-{int(file_index):03d}.parquet")
    for camera_key in ("observation.images.front", "observation.images.up"):
        for chunk_index, file_index in zip(
            episodes[f"videos/{camera_key}/chunk_index"].to_pylist(),
            episodes[f"videos/{camera_key}/file_index"].to_pylist(),
            strict=True,
        ):
            referenced_files.add(
                root / f"videos/{camera_key}/chunk-{int(chunk_index):03d}/file-{int(file_index):03d}.mp4"
            )
    missing_files = sorted(str(path.relative_to(root)) for path in referenced_files if not path.is_file())
    empty_files = sorted(
        str(path.relative_to(root)) for path in referenced_files if path.is_file() and path.stat().st_size == 0
    )
    if missing_files:
        errors.append(f"referenced data/video files are missing: {missing_files}")
    if empty_files:
        errors.append(f"referenced data/video files are empty: {empty_files}")

if not (root / "meta/stats.json").is_file():
    errors.append("meta/stats.json is missing")

unexpected_dirs = sorted(
    path.name for path in root.iterdir() if path.is_dir() and path.name not in {"data", "meta", "videos"}
)
if unexpected_dirs:
    errors.append(f"unexpected possible partial-data directories exist: {unexpected_dirs}")

if errors:
    for error in errors:
        print(f"Preflight failed: {error}", file=sys.stderr)
    raise SystemExit(5)

print(f"resume dataset with {total_episodes} finalized episode(s) / {total_frames} frame(s)")
PY
)"
  RESUME_VALUE="true"
fi
readonly RESUME_VALUE
readonly DATASET_SUMMARY
NUM_EPISODES="10"
if [[ "${RESUME_VALUE}" == "false" ]]; then
  NUM_EPISODES="1"
fi
readonly NUM_EPISODES

for device_path in "${ROBOT_PORT}" "${TELEOP_PORT}" "${FRONT_CAMERA}" "${UP_CAMERA}"
do
  resolved_device="$(readlink -f -- "${device_path}")"
  # `sg dialout` changes the effective group. On this host, /proc then denies
  # access to file descriptors of same-user desktop processes. Run only this
  # read-only scan under the login group; recording remains in dialout.
  if [[ "$(id -gn)" == "${LOGIN_GROUP}" ]]; then
    device_users="$("${PYTHON_BIN}" "${DEVICE_USER_SCANNER}" "${resolved_device}")"
  else
    device_users="$(
      sg "${LOGIN_GROUP}" -c \
        "exec '${PYTHON_BIN}' '${DEVICE_USER_SCANNER}' '${resolved_device}'" \
        2>/dev/null
    )"
  fi
  if [[ -n "${device_users//[[:space:]]/}" ]]; then
    echo "Preflight failed: ${device_path} is already in use by PID(s):${device_users}" >&2
    echo "Close the process using it, then run this launcher again." >&2
    exit 3
  fi
done

if [[ "${1:-}" == "--check" ]]; then
  "${PYTHON_BIN}" "${RECORDER}" --help >/dev/null
  echo "Preflight passed: recorder, calibrations, arms, and cameras are present; ${DATASET_SUMMARY}."
  echo "No hardware was opened and no dataset file was created."
  exit 0
fi

mkdir -p "${LOG_DIR}"
readonly DATASET_LOCK_FILE="${LOG_DIR}/left_to_right_dataset.lock"
readonly LOG_FILE="${LOG_DIR}/left_to_right_$(date +%Y%m%d_%H%M%S_%N)_$$.log"
echo "Recording log: ${LOG_FILE}"
echo "Dataset mode: ${DATASET_SUMMARY}."
if [[ "${RESUME_VALUE}" == "false" ]]; then
  echo "This first launch records 1 supervised LEFT-to-RIGHT pilot episode."
  echo "After inspecting that pilot, run this same command again to append batches of up to 10."
else
  echo "This launcher records up to 10 LEFT-to-RIGHT episodes. Recording starts automatically."
fi
echo "Move BLUE first, then RED, from the LEFT gray platform to the RIGHT white pad."
echo "Between episodes, reset both cubes to the LEFT gray platform; tap Right Arrow once when ready."
echo "Each episode is saved and encoded before the next starts; encoding cannot be skipped safely."

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
  --dataset.repo_id=bourn117/viola_cubes_left_to_right_keep_pose_v3 \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=60 \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.video=true \
  --dataset.video_encoding_batch_size=1 \
  --dataset.push_to_hub=false \
  --dataset.single_task="${EXPECTED_TASK}" \
  --play_sounds=true \
  --resume="${RESUME_VALUE}" \
  2>&1 | tee "${LOG_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e

if [[ "${pipeline_status[0]}" -eq 73 ]]; then
  echo "Preflight failed: another left-to-right recording launcher is already running." >&2
  echo "Wait for it to finish; never run two writers against the same dataset root." >&2
  exit 4
fi
if [[ "${pipeline_status[0]}" -ne 0 ]]; then
  exit "${pipeline_status[0]}"
fi
if [[ "${pipeline_status[1]}" -ne 0 ]]; then
  exit "${pipeline_status[1]}"
fi
