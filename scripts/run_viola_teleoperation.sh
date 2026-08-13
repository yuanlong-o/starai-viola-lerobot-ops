#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Legacy Viola teleoperation is disabled.' \
  '' \
  'Direct teleoperation is not supported by the live-session contract. This script' \
  'cannot connect a leader, follower, camera, serial device, or motor.'
exit 64
