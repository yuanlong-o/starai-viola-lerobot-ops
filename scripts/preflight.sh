#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
readonly REQUIREMENTS_FILE="${REPO_DIR}/requirements-validated.txt"

errors=0
fail() { echo "FAIL: $*" >&2; errors=$((errors + 1)); }
pass() { echo "PASS: $*"; }

echo "StarAI Viola Repo-A software and contract preflight"
echo "Checks: repository identity, clean revision, Python, pinned packages, editable install, and shared schemas."
echo "Not checked: devices, cameras, motors, calibration, checkpoints, W&B connectivity, or physical behavior."
echo "IMPORTANT: passing this audit is not motion authorization."
echo

if command -v git >/dev/null 2>&1 && git -C "${REPO_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  revision="$(git -C "${REPO_DIR}" rev-parse HEAD)"
  branch="$(git -C "${REPO_DIR}" branch --show-current)"
  pass "repository ${REPO_DIR}"
  pass "branch ${branch:-detached HEAD}; revision ${revision}"

  mapfile -t worktree_changes < <(git -C "${REPO_DIR}" status --short --untracked-files=all)
  if (( ${#worktree_changes[@]} == 0 )); then
    pass "worktree is clean"
  else
    fail "worktree has ${#worktree_changes[@]} tracked or untracked change(s)"
    printf '      %s\n' "${worktree_changes[@]:0:10}" >&2
    if (( ${#worktree_changes[@]} > 10 )); then
      echo "      ... and $((${#worktree_changes[@]} - 10)) more" >&2
    fi
  fi
else
  fail "${REPO_DIR} is not an inspectable Git worktree"
fi

if [[ -n "${LEROBOT_PYTHON_BIN:-}" ]]; then
  python_command=("${LEROBOT_PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
  python_command=(conda run --no-capture-output --name "${ENV_NAME}" python)
else
  python_command=()
  fail "Conda is unavailable and LEROBOT_PYTHON_BIN was not provided"
fi

if (( ${#python_command[@]} > 0 )); then
  if "${python_command[@]}" - "${REPO_DIR}" "${REQUIREMENTS_FILE}" "${ENV_NAME}" <<'PY'
from __future__ import annotations

import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, distribution, entry_points, version
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


repo = Path(sys.argv[1]).resolve()
requirements_path = Path(sys.argv[2])
expected_environment = sys.argv[3]
problems: list[str] = []

if sys.version_info[:2] != (3, 12):
    problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} is active; expected 3.12")

active_environment = os.environ.get("CONDA_DEFAULT_ENV")
if active_environment != expected_environment:
    problems.append(
        f"Conda environment is {active_environment or 'unset'}; expected {expected_environment}"
    )

expected_packages: dict[str, str] = {}
try:
    requirement_lines = requirements_path.read_text(encoding="utf-8").splitlines()
except OSError as error:
    problems.append(f"cannot read {requirements_path}: {error}")
    requirement_lines = []
for line in requirement_lines:
    clean = line.strip()
    if not clean or clean.startswith("#"):
        continue
    if clean.count("==") != 1:
        problems.append(f"requirement is not exactly pinned: {clean}")
        continue
    package, wanted = clean.split("==", 1)
    expected_packages[package] = wanted

for package, wanted in expected_packages.items():
    try:
        actual = version(package)
    except PackageNotFoundError:
        problems.append(f"{package} is not installed")
        continue
    if actual != wanted:
        problems.append(f"{package}=={actual}; expected {wanted}")

try:
    repo_distribution = distribution("starai-viola-lerobot-ops")
except PackageNotFoundError:
    problems.append("starai-viola-lerobot-ops is not installed")
else:
    direct_url_text = repo_distribution.read_text("direct_url.json")
    if direct_url_text is None:
        problems.append("starai-viola-lerobot-ops is not an editable install")
    else:
        try:
            direct_url = json.loads(direct_url_text)
            installed_url = direct_url["url"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            problems.append(f"editable-install metadata is invalid: {error}")
        else:
            installed_path = Path(unquote(urlsplit(installed_url).path)).resolve()
            if direct_url.get("dir_info", {}).get("editable") is not True:
                problems.append("starai-viola-lerobot-ops install is not marked editable")
            if installed_path != repo:
                problems.append(f"editable install points to {installed_path}; expected {repo}")

console_scripts = {item.name: item.value for item in entry_points(group="console_scripts")}
expected_scripts = {
    "viola-handoff": "viola_handoff.cli:main",
    "viola-ops": "viola_ops.cli:main",
}
for name, target in expected_scripts.items():
    if console_scripts.get(name) != target:
        problems.append(f"console command {name} does not resolve to {target}")

for module_name in ("viola_handoff.cli", "viola_ops.cli"):
    try:
        importlib.import_module(module_name)
    except Exception as error:
        problems.append(f"cannot import {module_name}: {error}")

try:
    from viola_handoff.contract import CONTRACT_SHA256
except Exception as error:
    problems.append(f"cannot import the shared handoff contract: {error}")
else:
    expected_contract = "fbfef2f214ff320f03891f9694056a4377c1405e0362bb5e9cf226ef1b82e99e"
    if CONTRACT_SHA256 != expected_contract:
        problems.append(f"handoff contract hash is {CONTRACT_SHA256}; expected {expected_contract}")

expected_schemas = {
    "dataset_release_v2.schema.json": "64e1c1fdea32117c6b1a26b93f8cc2825c05a202de36e958b46c3198944df6db",
    "rollout_evidence.schema.json": "f39530be3a279c893941d0fb2dcf7721bc16fd1e6c9880db7375ca9ab577f9ba",
    "rollout_session.schema.json": "31c24505b03fcb75acdd649e0ed3492795cf8246267915879ce3e0708b6f80e5",
    "shadow_evidence.schema.json": "778a3cbb73355764626a2f1b2bf915d347951d118287cc846e7a30e9a09cfd90",
}
for name, wanted in expected_schemas.items():
    schema_path = repo / "contracts" / name
    try:
        raw_schema = schema_path.read_bytes()
        json.loads(raw_schema)
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"cannot read valid schema {name}: {error}")
        continue
    actual = hashlib.sha256(raw_schema).hexdigest()
    if actual != wanted:
        problems.append(f"schema {name} hash is {actual}; expected {wanted}")

if problems:
    for problem in problems:
        print(f"      {problem}", file=sys.stderr)
    raise SystemExit(1)

print(f"      Python {sys.version.split()[0]} in Conda environment {active_environment}")
print(f"      {len(expected_packages)} exact package pins match")
print("      Repo-A editable install and both console commands match")
print("      Shared handoff contract and four schema hashes match")
PY
  then
    pass "Python environment and shared contract"
  else
    fail "Python environment or shared contract mismatch"
  fi
fi

case "${WANDB_MODE:-online}" in
  online|ONLINE|Online)
    pass "W&B mode permits online evidence publishing; connectivity was not tested"
    ;;
  *)
    fail "WANDB_MODE=${WANDB_MODE} prevents required online evidence publishing"
    ;;
esac

echo
if (( errors > 0 )); then
  echo "Software/contract preflight failed with ${errors} problem(s)." >&2
  echo "No hardware was inspected or opened. This audit never authorizes motion." >&2
  exit 1
fi
echo "Software/contract preflight passed."
echo "No hardware was inspected or opened, and this result is not motion authorization."
