#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Anaconda, then rerun." >&2
  exit 2
fi

if conda run --name "${ENV_NAME}" python -c 'raise SystemExit(0)' >/dev/null 2>&1; then
  echo "Using existing Conda environment '${ENV_NAME}'."
else
  echo "Creating Conda environment '${ENV_NAME}' with Python 3.12."
  conda create --name "${ENV_NAME}" python=3.12 pip -y
fi

python_version="$(conda run --name "${ENV_NAME}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.12" ]]; then
  echo "Environment '${ENV_NAME}' uses Python ${python_version}; Python 3.12 is required." >&2
  exit 2
fi

conda run --no-capture-output --name "${ENV_NAME}" \
  python -m pip install --upgrade pip
conda run --no-capture-output --name "${ENV_NAME}" \
  python -m pip install --requirement "${REPO_DIR}/requirements-validated.txt"
conda run --no-capture-output --name "${ENV_NAME}" \
  python -m pip install --no-deps --editable "${REPO_DIR}"
conda run --no-capture-output --name "${ENV_NAME}" python -m pip check

echo
echo "Repo-A software is installed in '${ENV_NAME}'."
echo "The repository was installed editable and without resolving dependencies a second time."
echo
echo "Next software-only check:"
echo "  conda activate ${ENV_NAME}"
echo "  ${SCRIPT_DIR}/preflight.sh"
echo
echo "Bootstrap does not configure hardware, authorize motion, or prove readiness."
echo "Configure online W&B credentials separately before running a command that publishes evidence."
