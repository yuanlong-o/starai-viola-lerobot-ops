#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Legacy Viola calibration is disabled.' \
  '' \
  'Calibration motion is not supported by the live-session contract. This script' \
  'cannot connect a serial device or motor. Use only a separately reviewed setup' \
  'procedure whose evidence is bound into session inputs.'
exit 64
