#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this file instead of executing it:\n  source /home/yz/starai_ws/starai_env.sh\n' >&2
  exit 2
fi

STARAI_CONDA_SH="/home/yz/anaconda3/etc/profile.d/conda.sh"
STARAI_ENV_PREFIX="/home/yz/anaconda3/envs/starai_ros_humble"
STARAI_WORKSPACE="/home/yz/starai_ws"

if [[ ! -r "${STARAI_CONDA_SH}" ]]; then
  printf 'Conda initialization is missing at %s\n' "${STARAI_CONDA_SH}" >&2
  return 1
fi
if [[ ! -r "${STARAI_ENV_PREFIX}/conda-meta/history" ]]; then
  printf 'The StarAI ROS environment is missing. Run %s/setup_starai.sh first.\n' \
    "${STARAI_WORKSPACE}" >&2
  return 1
fi
if [[ ! -r "${STARAI_WORKSPACE}/install/setup.bash" ]]; then
  printf 'The StarAI workspace is not built. Run %s/setup_starai.sh first.\n' \
    "${STARAI_WORKSPACE}" >&2
  return 1
fi

export PYTHONNOUSERSITE=1

# Do not let a previously sourced system ROS or unrelated overlay leak ABI or
# Python paths into this rootless workspace. install/setup.bash reconstructs
# the reviewed Conda parent prefix after these variables are cleared.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH PYTHONPATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ETC_DIR
unset RMW_IMPLEMENTATION

starai_nounset_was_on=false
case $- in
  *u*)
    starai_nounset_was_on=true
    set +u
    ;;
esac

starai_load_status=0
if [[ "${CONDA_PREFIX:-}" != "${STARAI_ENV_PREFIX}" ]]; then
  # shellcheck disable=SC1090
  if ! source "${STARAI_CONDA_SH}"; then
    printf 'Could not initialize Conda from %s\n' "${STARAI_CONDA_SH}" >&2
    starai_load_status=1
  elif ! conda activate "${STARAI_ENV_PREFIX}"; then
    printf 'Could not activate %s\n' "${STARAI_ENV_PREFIX}" >&2
    starai_load_status=1
  fi
fi

if ((starai_load_status == 0)); then
  # shellcheck disable=SC1091
  if ! source "${STARAI_WORKSPACE}/install/setup.bash"; then
    printf 'Could not load %s/install/setup.bash\n' "${STARAI_WORKSPACE}" >&2
    starai_load_status=1
  fi
fi

if [[ "${starai_nounset_was_on}" == true ]]; then
  set -u
fi
unset starai_nounset_was_on

if ((starai_load_status != 0)); then
  unset starai_load_status
  return 1
fi
unset starai_load_status

export RCUTILS_COLORIZED_OUTPUT=1
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
export ROS2CLI_DISABLE_DAEMON=1

printf 'Loaded ROS 2 %s from %s and workspace %s\n' \
  "${ROS_DISTRO:-unknown}" "${STARAI_ENV_PREFIX}" "${STARAI_WORKSPACE}"
