#!/usr/bin/env bash
set -u

readonly WORKSPACE="/home/yz/starai_ws"
readonly REPOSITORY="${WORKSPACE}/src/fashionstar-starai-arm-ros2"
readonly ENV_PREFIX="/home/yz/anaconda3/envs/starai_ros_humble"
readonly UPSTREAM_COMMIT="be498c0034f30bfbe1ceacafeb773ffd95309a69"
readonly REVIEWED_TREE="e5b01741b6be8f9cdb69d09c90ba12a20807b4a1"
readonly PATCH_FILE="${WORKSPACE}/patches/0001-harden-viola-ros-2-moveit-startup-and-control.patch"
readonly FOLLOWER_PORT="/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0"
readonly TEACHER_PORT="/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0"

failures=0
warnings=0

pass() {
  printf '[PASS] %s\n' "$*"
}

fail() {
  printf '[FAIL] %s\n' "$*"
  failures=$((failures + 1))
}

warn() {
  printf '[WARN] %s\n' "$*"
  warnings=$((warnings + 1))
}

for required_command in fuser pgrep timeout; do
  if command -v "${required_command}" >/dev/null 2>&1; then
    pass "Required safety command is available: ${required_command}"
  else
    fail "Required safety command is missing: ${required_command}"
  fi
done

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]]; then
    pass "Operating system: ${PRETTY_NAME}"
  else
    fail "Expected Ubuntu 22.04; found ${PRETTY_NAME:-unknown}"
  fi
else
  fail "Cannot read /etc/os-release"
fi

if [[ -r "${WORKSPACE}/starai_env.sh" ]]; then
  # shellcheck disable=SC1091
  if source "${WORKSPACE}/starai_env.sh"; then
    pass "Loaded the rootless StarAI ROS environment"
  else
    fail "Could not load ${WORKSPACE}/starai_env.sh"
  fi
else
  fail "Missing ${WORKSPACE}/starai_env.sh"
fi

if [[ "${CONDA_PREFIX:-}" == "${ENV_PREFIX}" ]]; then
  pass "Active environment: ${ENV_PREFIX}"
else
  fail "Expected CONDA_PREFIX=${ENV_PREFIX}; found ${CONDA_PREFIX:-unset}"
fi

python_executable="$(command -v python 2>/dev/null || true)"
ros2_executable="$(command -v ros2 2>/dev/null || true)"
if [[ "${python_executable}" == "${ENV_PREFIX}/bin/python" &&
      "${ros2_executable}" == "${ENV_PREFIX}/bin/ros2" ]]; then
  pass "python and ros2 resolve inside the dedicated Conda environment"
else
  fail "Expected dedicated python/ros2; found ${python_executable:-missing} and ${ros2_executable:-missing}"
fi

ros_environment_paths="${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:"\
"${COLCON_PREFIX_PATH:-}:${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}"
if [[ "${ros_environment_paths}" == *"/opt/ros/"* ]]; then
  fail "A system /opt/ros path leaked into the rootless environment"
else
  pass "No system /opt/ros path is present in ROS, library, or Python overlays"
fi

if [[ "${ROS_LOCALHOST_ONLY:-}" == "1" ]]; then
  pass "ROS discovery is restricted to this PC"
else
  fail "ROS_LOCALHOST_ONLY must be 1 before hardware control"
fi
if [[ "${ROS_DOMAIN_ID:-}" == "42" ]]; then
  pass "ROS uses the dedicated local StarAI domain 42"
else
  fail "ROS_DOMAIN_ID must be 42 before hardware control"
fi

if python - <<'PY' >/dev/null 2>&1
import sys
from importlib.metadata import version
import numpy
import rclpy
import serial
import fashionstar_uart_sdk

assert sys.version_info[:2] == (3, 11)
assert numpy.__version__ == "1.26.4"
assert version("setuptools") == "68.2.2"
assert version("pytest") == "7.4.4"
assert version("pyserial") == "3.5"
assert version("fashionstar-uart-sdk") == "1.3.8"
PY
then
  pass "Pinned Python, NumPy, setuptools, pytest, serial, and servo SDK versions"
else
  fail "One or more pinned Python dependencies are missing or changed"
fi

for package in moveit_ros_move_group controller_manager rviz2 robo_driver \
  viola_controller viola_description viola_moveit_config; do
  if ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    pass "ROS package available: ${package}"
  else
    fail "ROS package missing: ${package}"
  fi
done

if ros2 interface show robo_interfaces/srv/SetAngles >/dev/null 2>&1 &&
   ros2 interface show robo_interfaces/srv/StopServos >/dev/null 2>&1; then
  pass "Acknowledged motion and stop service interfaces are installed"
else
  fail "SetAngles or StopServos interface is missing"
fi

active_nodes=""
if active_nodes="$(timeout 4 ros2 node list 2>/dev/null)"; then
  node_conflict=false
  for conflict_node in /controller_manager /move_group /viola_controller \
    /viola_controller_node /viola_driver /robo_driver_node; do
    if [[ $'\n'"${active_nodes}"$'\n' == *$'\n'"${conflict_node}"$'\n'* ]]; then
      fail "Conflicting ROS node is already active: ${conflict_node}"
      node_conflict=true
    fi
  done
  if [[ "${node_conflict}" == false ]]; then
    pass "No fake-hardware or existing Viola control node was discovered"
  fi
else
  fail "Could not inspect active ROS nodes; refusing to assume the graph is clear"
fi

if [[ -d "${REPOSITORY}/.git" ]]; then
  if git -C "${REPOSITORY}" merge-base --is-ancestor \
    "${UPSTREAM_COMMIT}" HEAD 2>/dev/null; then
    pass "Source is based on pinned Seeed commit ${UPSTREAM_COMMIT:0:12}"
  else
    fail "Source is not based on the pinned Seeed commit"
  fi
  if [[ -z "$(git -C "${REPOSITORY}" status --porcelain)" ]]; then
    pass "Patched source checkout is clean"
  else
    fail "Patched source checkout has uncommitted changes"
  fi
  actual_tree="$(git -C "${REPOSITORY}" rev-parse 'HEAD^{tree}' 2>/dev/null || true)"
  if [[ "${actual_tree}" == "${REVIEWED_TREE}" ]]; then
    pass "Source tree exactly matches reviewed tree ${REVIEWED_TREE:0:12}"
  else
    fail "Source tree ${actual_tree:-unknown} does not match reviewed tree"
  fi
  if [[ -r "${PATCH_FILE}" ]] &&
     git -C "${REPOSITORY}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
    pass "Reviewed Viola safety patch is present"
  else
    fail "Cannot verify the reviewed Viola safety patch"
  fi
else
  fail "StarAI source repository is missing"
fi

if python - <<'PY' >/dev/null 2>&1
from robo_driver.robo_driver import (
    DEFAULT_SERIAL_PORT,
    MAX_SERVO_SPEED_DPS,
    SERVO_ANGLE_LIMITS,
)

assert DEFAULT_SERIAL_PORT.endswith("usb-0:10:1.0-port0")
assert SERVO_ANGLE_LIMITS[6] == (6.25, 100.0)
assert len(MAX_SERVO_SPEED_DPS) == 7
PY
then
  pass "Driver has the stable follower port, model limits, and speed guards"
else
  fail "Driver constants do not match the reviewed Viola safety configuration"
fi

driver_args="$(ros2 launch viola_moveit_config driver.launch.py --show-args 2>/dev/null || true)"
if [[ "${driver_args}" == *"hold_current"* &&
      "${driver_args}" == *"reset_multiturn"* &&
      "${driver_args}" == *"default: 'false'"* &&
      "${driver_args}" == *"feedback_fault_timeout_s"* &&
      "${driver_args}" == *"default: '0.5'"* &&
      "${driver_args}" == *"damping"* ]]; then
  pass "Driver defaults to hold-current, no reset, watchdog, and damping shutdown"
else
  fail "Driver launch arguments do not have the reviewed safety defaults"
fi

rviz_config="${WORKSPACE}/install/viola_moveit_config/share/viola_moveit_config/config/moveit.rviz"
if [[ -r "${rviz_config}" ]] &&
   [[ "$(<"${rviz_config}")" == *"Velocity_Scaling_Factor: 0.05"* ]] &&
   [[ "$(<"${rviz_config}")" == *"Acceleration_Scaling_Factor: 0.05"* ]]; then
  pass "RViz motion-request velocity and acceleration scaling are both 5%"
else
  fail "Installed RViz config does not default both motion scaling factors to 5%"
fi

moveit_args="$(ros2 launch viola_moveit_config actual_robot_demo.launch.py --show-args 2>/dev/null || true)"
if [[ "${moveit_args}" == *"allow_trajectory_execution"* &&
      "${moveit_args}" == *"default: 'false'"* ]]; then
  pass "Real-robot MoveIt defaults to planning-only"
else
  fail "Real-robot MoveIt does not default to planning-only"
fi

if [[ -e "${FOLLOWER_PORT}" ]]; then
  pass "Viola follower path exists: ${FOLLOWER_PORT} -> $(readlink -f "${FOLLOWER_PORT}")"
else
  fail "Viola follower path is not connected: ${FOLLOWER_PORT}"
fi
if [[ -e "${TEACHER_PORT}" ]]; then
  pass "Violin teacher path exists: ${TEACHER_PORT} -> $(readlink -f "${TEACHER_PORT}")"
else
  warn "Violin teacher path is not currently connected: ${TEACHER_PORT}"
fi
warn "USB by-path identifies the physical PC socket, not the arm; confirm the "\
  "follower cable is physically labelled on USB path 0:10 before startup"

if [[ -e "${FOLLOWER_PORT}" ]]; then
  if [[ -r "${FOLLOWER_PORT}" && -w "${FOLLOWER_PORT}" ]]; then
    pass "Current process can read and write the Viola follower path"
  else
    fail "No read/write access to the Viola follower; run preflight through sg dialout"
  fi
  if command -v fuser >/dev/null 2>&1; then
    holders="$(fuser "${FOLLOWER_PORT}" 2>/dev/null || true)"
    if [[ -z "${holders//[[:space:]]/}" ]]; then
      pass "No process currently owns the Viola follower port"
    else
      fail "Viola follower port is already used by PID(s): ${holders}"
    fi
  else
    warn "fuser is unavailable; port ownership was not inspected"
  fi
fi

if pgrep -af \
  'lerobot-(teleoperate|record)|teleoperate_keep_pose\.py|record(_episodes)?_keep_pose\.py|robo_driver.*driver' \
  >/dev/null 2>&1; then
  fail "A LeRobot or ROS hardware-control process appears to be active"
else
  pass "No LeRobot or ROS hardware-control process was detected"
fi

printf '\nSummary: %d failure(s), %d warning(s)\n' "${failures}" "${warnings}"
printf 'No serial device was opened and no motor command was sent.\n'
if ((failures)); then
  exit 1
fi
