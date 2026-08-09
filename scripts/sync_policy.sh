#!/usr/bin/env bash
set -Eeuo pipefail

readonly SOURCE_POLICY="${VIOLA_POLICY_SOURCE:-/mnt/nas02/yz/starai/outputs/act_viola_right_to_left_blue_then_red_val20_v1/checkpoints/080000/pretrained_model}"
readonly TARGET_POLICY="${VIOLA_POLICY_DIR:-${HOME}/models/act_viola_val20_step080000}"
readonly EXPECTED_MODEL_SHA256="1093aaeddfb902e7e596425d87676baba58cb8ab617a52c954ec11940726b886"
readonly REQUIRED_FILES=(
  config.json
  train_config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
  policy_preprocessor_step_3_normalizer_processor.safetensors
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
)

if [[ ! -d "${SOURCE_POLICY}" ]]; then
  echo "Policy source is unavailable: ${SOURCE_POLICY}" >&2
  echo "Mount the NAS or set VIOLA_POLICY_SOURCE to a complete checkpoint directory." >&2
  exit 2
fi

mkdir -p -- "${TARGET_POLICY}"
rsync -a --checksum -- "${SOURCE_POLICY}/" "${TARGET_POLICY}/"

for filename in "${REQUIRED_FILES[@]}"; do
  [[ -s "${TARGET_POLICY}/${filename}" ]] || {
    echo "Checkpoint bundle is incomplete: ${TARGET_POLICY}/${filename}" >&2
    exit 3
  }
done

actual_hash="$(sha256sum -- "${TARGET_POLICY}/model.safetensors" | awk '{print $1}')"
if [[ "${actual_hash}" != "${EXPECTED_MODEL_SHA256}" ]]; then
  echo "Model checksum mismatch: ${actual_hash}; expected ${EXPECTED_MODEL_SHA256}" >&2
  exit 4
fi

echo "Policy verified: ${TARGET_POLICY} (${actual_hash})"
