#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Anaconda, then rerun." >&2
  exit 2
fi

conda create --name "${ENV_NAME}" python=3.12 pip -y
conda run --no-capture-output --name "${ENV_NAME}" \
  python -m pip install --upgrade pip
conda run --no-capture-output --name "${ENV_NAME}" \
  python -m pip install --requirement "${REPO_DIR}/requirements-validated.txt"
conda run --no-capture-output --name "${ENV_NAME}" python -m pip check

if [[ ! -f "${REPO_DIR}/config/operation.env" ]]; then
  cp -- "${REPO_DIR}/config/operation.env.example" "${REPO_DIR}/config/operation.env"
  echo "Created config/operation.env; update device paths after USB discovery."
fi

echo
echo "Environment '${ENV_NAME}' is installed. Next:"
echo "  1. sudo usermod -aG dialout,video \"${USER}\"; log out and back in"
echo "  2. edit ${REPO_DIR}/config/operation.env"
echo "  3. ${SCRIPT_DIR}/install_calibrations.sh --install"
echo "  4. ${SCRIPT_DIR}/sync_policy.sh"
echo "  5. conda activate ${ENV_NAME}; ${SCRIPT_DIR}/preflight.sh"
