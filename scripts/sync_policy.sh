#!/usr/bin/env bash
set -Eeuo pipefail

printf '%s\n' \
  'Disabled: arbitrary ACT checkpoint copying is not a policy handoff.' \
  'Use viola-handoff accept on a Repo-B policy_candidate bundle.' \
  'Runtime code consumes only the verified receiver-local artifact copy.' >&2
exit 64
