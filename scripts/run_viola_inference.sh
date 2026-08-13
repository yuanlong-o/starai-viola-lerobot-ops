#!/usr/bin/env bash
set -Eeuo pipefail

printf >&2 '%s\n' \
  'Legacy inference is disabled.' \
  '' \
  'The exact previously deployed ACT model now has a Repo-A-local safety gate.' \
  'It requires reviewed local bytes, a clean revision, current E-stop evidence,' \
  'online W&B, and explicit operator actions, but no Repo-B handoff.' \
  '' \
  'Start with:' \
  '  viola-ops act run --help'
exit 64
