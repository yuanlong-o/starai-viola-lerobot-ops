#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Direct ROS 2 Viola driver startup is disabled.' \
  '' \
  'ROS motion is not supported by the live-session contract. This script cannot' \
  'inspect or connect a serial device, launch a driver, or arm a motor.'
exit 64
