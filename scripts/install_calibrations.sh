#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly SOURCE_ROOT="${REPO_DIR}/calibration"
readonly TARGET_ROOT="${HF_LEROBOT_CALIBRATION:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot/calibration}"
readonly ROBOT_RELATIVE="robots/starai_viola/my_awesome_staraiviola_arm.json"
readonly TELEOP_RELATIVE="teleoperators/starai_violin/my_awesome_staraiviolin_arm.json"

usage() {
  echo "Usage: $0 --check | --install | --replace" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

mode="$1"
case "${mode}" in
  --check|--install|--replace) ;;
  *) usage; exit 2 ;;
esac

differences=0

for relative in "${ROBOT_RELATIVE}" "${TELEOP_RELATIVE}"
do
  source_path="${SOURCE_ROOT}/${relative}"
  target_path="${TARGET_ROOT}/${relative}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing repository calibration: ${source_path}" >&2
    exit 3
  fi

  source_hash="$(sha256sum -- "${source_path}" | awk '{print $1}')"
  if [[ "${mode}" == "--check" ]]; then
    if [[ ! -f "${target_path}" ]]; then
      echo "MISSING: ${target_path}"
      differences=$((differences + 1))
      continue
    fi
    target_hash="$(sha256sum -- "${target_path}" | awk '{print $1}')"
    if [[ "${source_hash}" == "${target_hash}" ]]; then
      echo "MATCH: ${target_path} (${target_hash})"
    else
      echo "DIFFERENT: ${target_path} (${target_hash}; repository ${source_hash})"
      differences=$((differences + 1))
    fi
    continue
  fi

  mkdir -p -- "$(dirname -- "${target_path}")"
  if [[ -e "${target_path}" && "${mode}" == "--install" ]]; then
    echo "Refusing to overwrite existing calibration: ${target_path}" >&2
    echo "Use --check first. Use --replace only for this exact physical arm pair." >&2
    exit 4
  fi
  if [[ -e "${target_path}" && "${mode}" == "--replace" ]]; then
    backup_path="${target_path}.before-restore-$(date +%Y%m%d_%H%M%S)"
    cp --preserve=mode,timestamps -- "${target_path}" "${backup_path}"
    echo "Backed up existing calibration to ${backup_path}"
  fi
  install -m 0644 -- "${source_path}" "${target_path}"
  installed_hash="$(sha256sum -- "${target_path}" | awk '{print $1}')"
  if [[ "${installed_hash}" != "${source_hash}" ]]; then
    echo "Checksum mismatch after installing ${target_path}" >&2
    exit 5
  fi
  echo "Installed: ${target_path} (${installed_hash})"
done

if [[ "${mode}" == "--check" && "${differences}" -ne 0 ]]; then
  exit 1
fi
