#!/usr/bin/env bash
set -Eeuo pipefail

printf '%s\n' \
  'Disabled: the historical ROS setup applied a patch to third-party source.' \
  'Repo-A production paths use public APIs and repo-owned adapters only.' \
  'No ROS hardware setup or driver was changed or started.' >&2
exit 64
