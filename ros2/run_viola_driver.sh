#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKSPACE="/home/yz/starai_ws"
readonly FOLLOWER_PORT="/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

for required_command in fuser pgrep timeout; do
  command -v "${required_command}" >/dev/null 2>&1 ||
    die "Required safety command is missing: ${required_command}"
done

[[ -e "${FOLLOWER_PORT}" ]] || die "Viola follower is not connected at ${FOLLOWER_PORT}."
[[ -r "${FOLLOWER_PORT}" && -w "${FOLLOWER_PORT}" ]] ||
  die "No serial permission. Run this script with: sg dialout -c 'exec $0'"

holders="$(fuser "${FOLLOWER_PORT}" 2>/dev/null || true)"
[[ -z "${holders//[[:space:]]/}" ]] ||
  die "Follower port is already owned by PID(s): ${holders}"

if pgrep -af \
  'lerobot-(teleoperate|record)|teleoperate_keep_pose\.py|record(_episodes)?_keep_pose\.py' \
  >/dev/null 2>&1; then
  die "A LeRobot teleoperation or recording process is active. Stop it first."
fi

# shellcheck disable=SC1091
source "${WORKSPACE}/starai_env.sh"

if ! active_nodes="$(timeout 4 ros2 node list 2>/dev/null)"; then
  die "Could not inspect active ROS nodes; refusing hardware startup."
fi
for conflict_node in /controller_manager /move_group /viola_controller \
  /viola_controller_node /viola_driver /robo_driver_node; do
  if [[ $'\n'"${active_nodes}"$'\n' == *$'\n'"${conflict_node}"$'\n'* ]]; then
    die "Conflicting ROS node is already active: ${conflict_node}. Stop simulation/control first."
  fi
done

printf 'Starting Viola with fresh seven-servo read, hold-current startup, no reset, and damping shutdown.\n'
printf 'Keep the arm supported and the power cutoff within reach.\n'

exec ros2 launch viola_moveit_config driver.launch.py \
  serial_port:="${FOLLOWER_PORT}" \
  startup_mode:=hold_current \
  reset_multiturn:=false \
  feedback_fault_timeout_s:=0.5 \
  shutdown_mode:=damping
