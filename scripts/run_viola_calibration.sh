#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_FILE="${VIOLA_OPERATION_CONFIG:-${REPO_DIR}/config/operation.env}"
[[ -f "${CONFIG_FILE}" ]] && source "${CONFIG_FILE}"

case "${1:-}" in
  violin|viola) device="$1" ;;
  *) echo "Usage: $0 {violin|viola}" >&2; exit 2 ;;
esac

readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
env_prefix="$(conda env list --json | python -c 'import json,sys; d=json.load(sys.stdin); n=sys.argv[1]; print(next((p for p in d["envs"] if p.rsplit("/",1)[-1] == n), ""))' "${ENV_NAME}")"
calibrate_bin="${env_prefix}/bin/lerobot-calibrate"
[[ -x "${calibrate_bin}" ]] || { echo "lerobot-calibrate not found for Conda environment ${ENV_NAME}" >&2; exit 2; }

if [[ "${device}" == "violin" ]]; then
  port="${VIOLA_TELEOP_PORT:?Set VIOLA_TELEOP_PORT in config/operation.env}"
  id="${VIOLA_TELEOP_ID:-my_awesome_staraiviolin_arm}"
  exec sg dialout -c "exec '${calibrate_bin}' --teleop.type=lerobot_teleoperator_violin --teleop.port='${port}' --teleop.id='${id}' --teleop.use_degrees=false"
fi

port="${VIOLA_ROBOT_PORT:?Set VIOLA_ROBOT_PORT in config/operation.env}"
id="${VIOLA_ROBOT_ID:-my_awesome_staraiviola_arm}"
exec sg dialout -c "exec '${calibrate_bin}' --robot.type=lerobot_robot_viola --robot.port='${port}' --robot.id='${id}' --robot.disable_torque_on_disconnect=true --robot.use_degrees=false"
