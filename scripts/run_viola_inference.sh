#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Legacy inference is disabled.' \
  '' \
  'Inference may run only through the unified live-session safety gate. It requires' \
  'an accepted, blocker-free rollout_session, reviewed setup, current E-stop' \
  'evidence, and explicit operator actions.' \
  '' \
  'Start with:' \
  '  viola-ops policy execute --help'
exit 64
