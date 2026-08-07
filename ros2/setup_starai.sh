#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKSPACE="/home/yz/starai_ws"
readonly CONDA_EXE="/home/yz/anaconda3/bin/conda"
readonly CONDA_SH="/home/yz/anaconda3/etc/profile.d/conda.sh"
readonly ENV_PREFIX="/home/yz/anaconda3/envs/starai_ros_humble"
readonly REPOSITORY_URL="https://github.com/Seeed-Projects/fashionstar-starai-arm-ros2.git"
readonly REPOSITORY_DIR="${WORKSPACE}/src/fashionstar-starai-arm-ros2"
readonly UPSTREAM_COMMIT="be498c0034f30bfbe1ceacafeb773ffd95309a69"
readonly REVIEWED_TREE="e5b01741b6be8f9cdb69d09c90ba12a20807b4a1"
readonly LOCAL_BRANCH="local/safe-viola-moveit"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ASSET_DIR="${STARAI_ASSET_DIR:-${SCRIPT_DIR}}"
readonly ENV_FILE="${ASSET_DIR}/environment.yml"
readonly PATCH_FILE="${ASSET_DIR}/patches/0001-harden-viola-ros-2-moveit-startup-and-control.patch"
readonly ENV_LOADER="${ASSET_DIR}/starai_env.sh"
readonly PREFLIGHT_SCRIPT="${ASSET_DIR}/preflight.sh"
readonly RUNNER_SCRIPT="${ASSET_DIR}/run_viola_driver.sh"
readonly SIMULATION_RUNNER="${ASSET_DIR}/run_viola_simulation.sh"

if [[ -r "${ASSET_DIR}/README.md" ]]; then
  readonly README_FILE="${ASSET_DIR}/README.md"
else
  readonly README_FILE="${ASSET_DIR}/../docs/ROS2_MOVEIT.md"
fi

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -r /etc/os-release ]] || die "Cannot identify the operating system."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] ||
  die "This implementation targets Ubuntu 22.04; found ${PRETTY_NAME:-unknown}."
[[ "$(uname -m)" == "x86_64" ]] || die "This environment file targets x86_64."
[[ -x "${CONDA_EXE}" && -r "${CONDA_SH}" ]] ||
  die "Expected Anaconda under /home/yz/anaconda3."
command -v git >/dev/null || die "git is required."
[[ -r "${ENV_FILE}" ]] || die "Missing environment file: ${ENV_FILE}"
[[ -r "${PATCH_FILE}" ]] || die "Missing safety patch: ${PATCH_FILE}"
[[ -r "${README_FILE}" ]] || die "Missing deployment README: ${README_FILE}"
[[ -r "${ENV_LOADER}" ]] || die "Missing environment loader: ${ENV_LOADER}"
[[ -r "${PREFLIGHT_SCRIPT}" ]] || die "Missing preflight script: ${PREFLIGHT_SCRIPT}"
[[ -r "${RUNNER_SCRIPT}" ]] || die "Missing hardware runner: ${RUNNER_SCRIPT}"
[[ -r "${SIMULATION_RUNNER}" ]] || die "Missing simulation runner: ${SIMULATION_RUNNER}"

# Remove inherited system ROS/overlay state before Conda is invoked. The
# dedicated environment and colcon workspace rebuild these paths explicitly.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH PYTHONPATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ETC_DIR
unset RMW_IMPLEMENTATION

printf '\n[1/6] Installing the tracked workspace entry points\n'
if [[ "${ASSET_DIR}" != "${WORKSPACE}" ]]; then
  mkdir -p "${WORKSPACE}/patches"
  install -m 0644 "${ENV_FILE}" "${WORKSPACE}/environment.yml"
  install -m 0644 "${PATCH_FILE}" "${WORKSPACE}/patches/$(basename "${PATCH_FILE}")"
  install -m 0644 "${README_FILE}" "${WORKSPACE}/README.md"
  install -m 0755 "${ASSET_DIR}/setup_starai.sh" "${WORKSPACE}/setup_starai.sh"
  install -m 0755 "${ENV_LOADER}" "${WORKSPACE}/starai_env.sh"
  install -m 0755 "${PREFLIGHT_SCRIPT}" "${WORKSPACE}/preflight.sh"
  install -m 0755 "${RUNNER_SCRIPT}" "${WORKSPACE}/run_viola_driver.sh"
  install -m 0755 "${SIMULATION_RUNNER}" "${WORKSPACE}/run_viola_simulation.sh"
fi

printf '\n[2/6] Creating or updating the rootless ROS environment\n'
if [[ -d "${ENV_PREFIX}/conda-meta" ]]; then
  "${CONDA_EXE}" env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}" --prune
else
  "${CONDA_EXE}" env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

export PYTHONNOUSERSITE=1
# shellcheck disable=SC1090
set +u
source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"
set -u

printf '\n[3/6] Fetching the pinned official source\n'
mkdir -p "${WORKSPACE}/src"
if [[ -e "${REPOSITORY_DIR}" && ! -d "${REPOSITORY_DIR}/.git" ]]; then
  die "${REPOSITORY_DIR} exists but is not a Git repository."
fi
cloned_repository=false
if [[ ! -d "${REPOSITORY_DIR}/.git" ]]; then
  git clone "${REPOSITORY_URL}" "${REPOSITORY_DIR}"
  cloned_repository=true
fi
if [[ -n "$(git -C "${REPOSITORY_DIR}" status --porcelain)" ]]; then
  die "The source checkout has uncommitted changes; refusing to overwrite them."
fi
if ! git -C "${REPOSITORY_DIR}" cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${REPOSITORY_DIR}" fetch origin "${UPSTREAM_COMMIT}"
fi
if [[ "${cloned_repository}" == true ]]; then
  git -C "${REPOSITORY_DIR}" switch --detach "${UPSTREAM_COMMIT}"
fi

printf '\n[4/6] Applying or verifying the local safety patch\n'
if git -C "${REPOSITORY_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  printf 'Safety patch is already present.\n'
elif [[ "$(git -C "${REPOSITORY_DIR}" rev-parse HEAD)" == "${UPSTREAM_COMMIT}" ]]; then
  if git -C "${REPOSITORY_DIR}" show-ref --verify --quiet "refs/heads/${LOCAL_BRANCH}"; then
    die "Branch ${LOCAL_BRANCH} already exists but the patch is not active."
  fi
  git -C "${REPOSITORY_DIR}" switch -c "${LOCAL_BRANCH}"
  git -C "${REPOSITORY_DIR}" \
    -c user.name="StarAI Workspace" \
    -c user.email="starai-workspace@localhost" \
    am "${PATCH_FILE}"
else
  die "Checkout is neither the pinned upstream commit nor the reviewed patched tree."
fi
actual_tree="$(git -C "${REPOSITORY_DIR}" rev-parse 'HEAD^{tree}')"
[[ "${actual_tree}" == "${REVIEWED_TREE}" ]] ||
  die "Patched source tree ${actual_tree} does not match reviewed tree ${REVIEWED_TREE}."

printf '\n[5/6] Building all StarAI ROS packages\n'
cd "${WORKSPACE}"
colcon build --symlink-install --event-handlers console_cohesion+
# shellcheck disable=SC1091
set +u
source "${WORKSPACE}/install/setup.bash"
set -u

printf '\n[6/6] Running safety tests and launch parsing checks\n'
colcon test \
  --packages-select robo_driver viola_controller \
  --event-handlers console_cohesion+ \
  --return-code-on-test-failure
colcon test-result --test-result-base build/robo_driver --verbose
colcon test-result --test-result-base build/viola_controller --verbose
ros2 launch viola_moveit_config driver.launch.py --show-args >/dev/null
ros2 launch viola_moveit_config actual_robot_demo.launch.py --show-args >/dev/null

printf '\nSetup completed without opening a serial device.\n'
printf 'Next: sg dialout -c '\''exec /home/yz/starai_ws/preflight.sh'\''\n'
