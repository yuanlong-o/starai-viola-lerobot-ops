#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Legacy Viola episode recording is disabled.' \
  '' \
  'Demonstration recording is not supported by the live-session contract. This' \
  'script cannot connect cameras, serial devices, a leader, or a follower.'
exit 64
