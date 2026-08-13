#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Archived right-to-left recording is disabled.' \
  '' \
  'The source dataset is frozen and must not be appended to. Legacy demonstration' \
  'recording is not supported by the live-session contract, and this script cannot' \
  'connect cameras, serial devices, a leader, or a follower.'
exit 64
