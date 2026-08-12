#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKSPACE="/home/yz/starai_ws"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

for required_command in pgrep timeout; do
  command -v "${required_command}" >/dev/null 2>&1 ||
    die "Required safety command is missing: ${required_command}"
done

if pgrep -af \
  'lerobot-(teleoperate|record)|teleoperate_keep_pose\.py|record(_episodes)?_keep_pose\.py|robo_driver.*driver' \
  >/dev/null 2>&1; then
  die "A LeRobot or ROS hardware-control process appears to be active. Stop it first."
fi

# shellcheck disable=SC1091
source "${WORKSPACE}/starai_env.sh"

if ! active_nodes="$(timeout 4 ros2 node list 2>/dev/null)"; then
  die "Could not inspect active ROS nodes; refusing simulation startup."
fi
for conflict_node in /controller_manager /move_group /viola_controller \
  /viola_controller_node /viola_driver /robo_driver_node; do
  if [[ $'\n'"${active_nodes}"$'\n' == *$'\n'"${conflict_node}"$'\n'* ]]; then
    die "Conflicting ROS node is already active: ${conflict_node}. Stop it first."
  fi
done

printf 'Starting isolated fake-hardware MoveIt simulation. No hardware driver is active.\n'
exec ros2 launch viola_moveit_config demo.launch.py
