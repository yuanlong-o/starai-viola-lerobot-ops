"""Immutable, content-addressed PC-A/PC-B handoff bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import urlsplit

from .errors import BundleValidationError, EnvironmentValidationError, EvidenceError, HandoffError
from .evidence import EvidenceLogger, required_evidence_logger

SCHEMA_VERSION: Final = "1.0"
EXPECTED_PYTHON: Final = (3, 12)
EXPECTED_LEROBOT_VERSION: Final = "0.6.1"
EXPECTED_WANDB_VERSION: Final = "0.27.2"
EXPECTED_CONDA_ENV: Final = "lerobot"
DEFAULT_HANDOFF_ROOT: Final = Path("/mnt/nas02/yz/starai/handoffs/v1")
DEFAULT_ACCEPT_ROOT: Final = Path("~/.local/share/viola/handoffs/v1")
WANDB_ID_PREFIX_LENGTH: Final = 16
MAX_PAYLOAD_BYTES: Final = 64 * 1024 * 1024
MAX_JSON_BYTES: Final = 64 * 1024 * 1024

KIND_RULES: Final[dict[str, dict[str, Any]]] = {
    "dataset_release": {
        "producer": "pc_a",
        "consumers": ["pc_b"],
        "permissions": ["data_only"],
        "default_permission": "data_only",
    },
    "policy_candidate": {
        "producer": "pc_b",
        "consumers": ["pc_a"],
        "permissions": ["disconnected_only"],
        "default_permission": "disconnected_only",
    },
    "shadow_evidence": {
        "producer": "pc_a",
        "consumers": ["pc_b"],
        "permissions": ["evidence_only"],
        "default_permission": "evidence_only",
    },
    "session_inputs": {
        "producer": "pc_a",
        "consumers": ["pc_b"],
        "permissions": ["planning_only"],
        "default_permission": "planning_only",
    },
    "rollout_session": {
        "producer": "pc_b",
        "consumers": ["pc_a"],
        "permissions": ["blocked", "live_session"],
        "default_permission": "blocked",
    },
    "rollout_evidence": {
        "producer": "pc_a",
        "consumers": ["pc_b"],
        "permissions": ["evidence_only"],
        "default_permission": "evidence_only",
    },
    "report": {
        "producer": "pc_b",
        "consumers": ["pc_a", "notion"],
        "permissions": ["report_only"],
        "default_permission": "report_only",
    },
}
SUPPORTED_KINDS: Final = tuple(KIND_RULES)
RECEIPT_STATUSES: Final = ("accepted", "rejected", "revoked")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically, rejecting NaN/Infinity and non-JSON data."""

    def validate(item: Any, location: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not (float("-inf") < item < float("inf")):
                raise BundleValidationError(f"non-finite number at {location}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{location}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise BundleValidationError(f"non-string JSON key at {location}")
                validate(child, f"{location}.{key}")
            return
        raise BundleValidationError(f"non-JSON value at {location}: {type(item).__name__}")

    validate(value, "$")
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CONTRACT_DEFINITION: Final = {
    "schema_version": SCHEMA_VERSION,
    "content_id_fields": [
        "contract_sha256",
        "kind",
        "experiment",
        "subject",
        "lineage",
        "payload",
        "artifact_inventories",
    ],
    "bundle_id": "content_id",
    "canonical_json": "utf8/sorted-keys/compact/no-nan",
    "hash": "sha256",
    "kinds": KIND_RULES,
    "live_session_seal": "typed_validated_rollout_session_only",
    "safety_authorization_source_freshness": "canonical-active-manifest-ready-match",
    "ready_wandb_url": "https://wandb.ai/{entity}/{project}/runs/{manifest_run_id}",
    "receipt_fields": [
        "accepted_artifacts",
        "actor",
        "bundle_id",
        "content_id",
        "created_at",
        "note",
        "receipt_key",
        "receipt_sha256",
        "schema_version",
        "status",
        "wandb_run_id",
    ],
    "receipt_statuses": list(RECEIPT_STATUSES),
}
CONTRACT_SHA256: Final = _sha256_bytes(canonical_json_bytes(CONTRACT_DEFINITION))

_HEX_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def wandb_run_id(content_id: str) -> str:
    """Return the stable W&B run ID shared by all events for one bundle."""

    if not _HEX_SHA_RE.fullmatch(content_id):
        raise BundleValidationError("content_id must be a lowercase SHA-256 digest")
    return f"ho-{content_id[:WANDB_ID_PREFIX_LENGTH]}"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise BundleValidationError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise BundleValidationError(f"{label} must be an RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo != UTC:
        raise BundleValidationError(f"{label} must use UTC")
    return value


def _reject_json_constant(token: str) -> None:
    raise BundleValidationError(f"non-finite JSON constant is forbidden: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_canonical_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise BundleValidationError(f"JSON file exceeds {MAX_JSON_BYTES} bytes: {path}")
        raw = path.read_bytes()
    except OSError as exc:
        raise BundleValidationError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except BundleValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"invalid JSON in {path}: {exc}") from exc
    if raw != canonical_json_bytes(value):
        raise BundleValidationError(f"JSON is not canonical: {path}")
    return value


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BundleValidationError(f"{label} must be a JSON object with string keys")
    # Round-tripping validates every nested value and protects the caller from
    # mutating a request after its content ID is calculated.
    return json.loads(canonical_json_bytes(value))


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BundleValidationError(f"{label} keys differ; missing={missing}, extra={extra}")


def _safe_relative_path(value: str, *, label: str = "inventory path") -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BundleValidationError(f"unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.endswith("/")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BundleValidationError(f"unsafe {label}: {value!r}")
    if path.as_posix() != value:
        raise BundleValidationError(f"non-canonical {label}: {value!r}")
    return path


def _absolute_without_symlinks(path: str | Path, *, must_exist: bool = True) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if must_exist:
                raise BundleValidationError(f"path does not exist: {candidate}") from None
            break
        if stat.S_ISLNK(mode):
            raise BundleValidationError(f"symlink path component is forbidden: {current}")
    if must_exist and not candidate.exists():
        raise BundleValidationError(f"path does not exist: {candidate}")
    return candidate


def _hash_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BundleValidationError(f"cannot safely open file {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise BundleValidationError(f"non-regular file is forbidden: {path}")
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise BundleValidationError(f"file changed while hashing: {path}")
        return digest.hexdigest(), after.st_size
    finally:
        os.close(fd)


def _inventory_tree(root: str | Path) -> dict[str, Any]:
    root_path = _absolute_without_symlinks(root)
    try:
        root_mode = root_path.lstat().st_mode
    except OSError as exc:
        raise BundleValidationError(f"cannot inspect inventory root {root_path}: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise BundleValidationError(f"inventory root must be a directory: {root_path}")

    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for current_text, dir_names, file_names in os.walk(root_path, topdown=True, followlinks=False):
        current = Path(current_text)
        dir_names.sort()
        file_names.sort()
        for name in dir_names:
            child = current / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise BundleValidationError(f"symlink is forbidden in inventory: {child}")
            if not stat.S_ISDIR(mode):
                raise BundleValidationError(f"non-directory entry is forbidden: {child}")
            relative = child.relative_to(root_path).as_posix()
            _safe_relative_path(relative)
            directories.append(relative)
        for name in file_names:
            child = current / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise BundleValidationError(f"symlink is forbidden in inventory: {child}")
            if not stat.S_ISREG(mode):
                raise BundleValidationError(f"non-regular file is forbidden: {child}")
            relative = child.relative_to(root_path).as_posix()
            _safe_relative_path(relative)
            digest, size = _hash_regular_file(child)
            files.append({"path": relative, "sha256": digest, "size_bytes": size})

    directories.sort()
    files.sort(key=lambda entry: entry["path"])
    inventory_body = {"directories": directories, "files": files}
    return {
        "directories": directories,
        "files": files,
        "file_count": len(files),
        "byte_count": sum(entry["size_bytes"] for entry in files),
        "inventory_sha256": _sha256_bytes(canonical_json_bytes(inventory_body)),
    }


def inventory_root(root: str | Path) -> dict[str, Any]:
    """Return the exact strict inventory shape used by seal and acceptance.

    This is a read-only filesystem operation. It rejects symlink path
    components, symlink entries, and non-regular files, and hashes every file
    using safe no-follow opens. Producers can bind the inventory SHA-256 in a
    payload before asking seal_bundle to independently inventory the same
    artifact again.
    """

    return _inventory_tree(root)


def _validate_inventory_shape(inventory: Any, label: str) -> dict[str, Any]:
    value = _json_object(inventory, label)
    _require_keys(
        value,
        {"directories", "files", "file_count", "byte_count", "inventory_sha256"},
        label,
    )
    if not isinstance(value["directories"], list) or not isinstance(value["files"], list):
        raise BundleValidationError(f"{label} directories/files must be lists")
    directories: list[str] = []
    previous = ""
    for item in value["directories"]:
        if not isinstance(item, str):
            raise BundleValidationError(f"{label} contains a non-string directory")
        _safe_relative_path(item)
        if item <= previous:
            raise BundleValidationError(f"{label} directories must be unique and sorted")
        previous = item
        directories.append(item)

    files: list[dict[str, Any]] = []
    previous = ""
    for index, raw_entry in enumerate(value["files"]):
        entry = _json_object(raw_entry, f"{label}.files[{index}]")
        _require_keys(entry, {"path", "sha256", "size_bytes"}, f"{label}.files[{index}]")
        path = entry["path"]
        if not isinstance(path, str):
            raise BundleValidationError(f"{label} contains a non-string file path")
        _safe_relative_path(path)
        if path <= previous:
            raise BundleValidationError(f"{label} file paths must be unique and sorted")
        previous = path
        if (
            not isinstance(entry["size_bytes"], int)
            or isinstance(entry["size_bytes"], bool)
            or entry["size_bytes"] < 0
        ):
            raise BundleValidationError(f"{label} contains an invalid file size")
        if not isinstance(entry["sha256"], str) or not _HEX_SHA_RE.fullmatch(entry["sha256"]):
            raise BundleValidationError(f"{label} contains an invalid file digest")
        files.append(entry)

    if value["file_count"] != len(files):
        raise BundleValidationError(f"{label} file_count does not match files")
    if value["byte_count"] != sum(entry["size_bytes"] for entry in files):
        raise BundleValidationError(f"{label} byte_count does not match files")
    expected_digest = _sha256_bytes(
        canonical_json_bytes({"directories": directories, "files": files})
    )
    if value["inventory_sha256"] != expected_digest:
        raise BundleValidationError(f"{label} inventory digest does not match entries")
    return value


def _identity_dict(identity: RuntimeIdentity) -> dict[str, Any]:
    value = {
        "role": identity.role,
        "repository_commit": identity.repository_commit,
        "repository_clean": identity.repository_clean,
        "hostname": identity.hostname,
        "python_version": identity.python_version,
        "lerobot_version": identity.lerobot_version,
        "conda_environment": identity.conda_environment,
    }
    _validate_identity(value)
    return value


def _validate_identity(value: Any, label: str = "runtime identity") -> dict[str, Any]:
    identity = _json_object(value, label)
    _require_keys(
        identity,
        {
            "role",
            "repository_commit",
            "repository_clean",
            "hostname",
            "python_version",
            "lerobot_version",
            "conda_environment",
        },
        label,
    )
    for key in ("role", "hostname", "python_version", "lerobot_version", "conda_environment"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise BundleValidationError(f"{label}.{key} must be a nonempty string")
    if not isinstance(identity["repository_commit"], str) or not _GIT_SHA_RE.fullmatch(
        identity["repository_commit"]
    ):
        raise BundleValidationError(f"{label}.repository_commit must be a full Git SHA")
    if identity["repository_clean"] is not True:
        raise BundleValidationError(f"{label} must bind a clean repository")
    version_parts = identity["python_version"].split(".")
    if version_parts[:2] != [str(EXPECTED_PYTHON[0]), str(EXPECTED_PYTHON[1])]:
        raise BundleValidationError(f"{label} must use Python 3.12")
    if identity["lerobot_version"] != EXPECTED_LEROBOT_VERSION:
        raise BundleValidationError(f"{label} must use LeRobot {EXPECTED_LEROBOT_VERSION}")
    if identity["conda_environment"] != EXPECTED_CONDA_ENV:
        raise BundleValidationError(f"{label} must use the lerobot Conda environment")
    return identity


@dataclass(frozen=True)
class RuntimeIdentity:
    role: str
    repository_commit: str
    repository_clean: bool
    hostname: str
    python_version: str
    lerobot_version: str
    conda_environment: str

    @classmethod
    def capture(cls, *, role: str, repo_root: str | Path) -> RuntimeIdentity:
        """Capture and enforce the clean PC runtime used for a write operation."""

        if sys.version_info[:2] != EXPECTED_PYTHON:
            found_python = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise EnvironmentValidationError(
                f"handoff writes require Python 3.12, found {found_python}"
            )
        conda_environment = os.environ.get("CONDA_DEFAULT_ENV", "")
        if conda_environment != EXPECTED_CONDA_ENV:
            raise EnvironmentValidationError(
                f"handoff writes require Conda environment {EXPECTED_CONDA_ENV!r}"
            )
        try:
            lerobot_version = importlib.metadata.version("lerobot")
        except importlib.metadata.PackageNotFoundError as exc:
            raise EnvironmentValidationError("LeRobot is not installed") from exc
        if lerobot_version != EXPECTED_LEROBOT_VERSION:
            message = (
                f"handoff writes require LeRobot {EXPECTED_LEROBOT_VERSION}, "
                f"found {lerobot_version}"
            )
            raise EnvironmentValidationError(message)
        try:
            wandb_version = importlib.metadata.version("wandb")
        except importlib.metadata.PackageNotFoundError as exc:
            raise EnvironmentValidationError("W&B is not installed") from exc
        if wandb_version != EXPECTED_WANDB_VERSION:
            raise EnvironmentValidationError(
                f"handoff writes require W&B {EXPECTED_WANDB_VERSION}, found {wandb_version}"
            )

        repository = _absolute_without_symlinks(repo_root)
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EnvironmentValidationError(
                f"cannot inspect Git repository {repository}: {exc}"
            ) from exc
        if dirty:
            raise EnvironmentValidationError("handoff writes require a clean Git repository")
        identity = cls(
            role=role,
            repository_commit=commit,
            repository_clean=True,
            hostname=socket.gethostname(),
            python_version=".".join(str(part) for part in sys.version_info[:3]),
            lerobot_version=lerobot_version,
            conda_environment=conda_environment,
        )
        _identity_dict(identity)
        return identity


@dataclass(frozen=True)
class SealRequest:
    root: str | Path
    kind: str
    experiment: str
    subject: str
    producer: RuntimeIdentity
    lineage: Mapping[str, Any]
    wandb_project: str
    payload_dir: str | Path | None = None
    artifact_roots: Mapping[str, str | Path] | None = None
    consumer_role: str | None = None
    permission: str | None = None
    created_at: str | None = None


_ROLLOUT_POLICIES: Final = frozenset(
    {"act", "diffusion", "vqbet", "smolvla", "pi0", "pi0_fast", "pi05", "groot"}
)
_ROLLOUT_PAYLOAD_KEYS: Final = {
    "schema_version",
    "session_id",
    "policy_bundle_id",
    "policy_content_id",
    "setup_hashes",
    "phase_permissions",
    "blockers",
    "operator",
    "task",
    "session_inputs_binding",
    "source_session",
    "executor",
    "estop",
    "act_infrastructure_clearance",
    "trial_protocol",
}
_ROLLOUT_LINEAGE_KEYS: Final = {
    "session_id",
    "policy",
    "policy_bundle_id",
    "policy_content_id",
    "session_manifest_sha256",
    "source_session_sha256",
    "session_inputs_bundle_id",
    "session_inputs_content_id",
    "session_inputs_manifest_sha256",
    "session_inputs_payload_sha256",
    "session_inputs_setup_record_inventory_sha256",
    "benchmark_lineage_sha256",
    "policy_bindings_sha256",
    "policy_binding_sha256",
    "physical_setup_binding_sha256",
    "executor_binding_sha256",
    "executor_attestation_sha256",
    "estop_attestation_sha256",
    "blockers",
    "act_infrastructure_clearance",
}
_ROLLOUT_SESSION_INPUT_KEYS: Final = {
    "bundle_id",
    "content_id",
    "manifest_sha256",
    "payload_sha256",
    "setup_record_inventory_sha256",
}
_ROLLOUT_SOURCE_KEYS: Final = {
    "sha256",
    "created_at_utc",
    "benchmark_lineage_sha256",
    "policy_bindings_sha256",
    "policy_binding_sha256",
    "physical_setup_binding_sha256",
    "executor_binding_sha256",
}
_ROLLOUT_EXECUTOR_KEYS: Final = {
    "repository",
    "repository_commit",
    "entrypoint_sha256",
    "attestation_sha256",
    "reviewed_at_utc",
}
_ROLLOUT_ESTOP_KEYS: Final = {"operator", "tested_at_utc", "attestation_sha256"}
_ROLLOUT_CLEARANCE_KEYS: Final = {
    "mode",
    "act_outcome_sha256",
    "act_terminal_status",
    "act_rollout_evidence_bundle_id",
    "attestation_sha256",
    "blocker_stage",
    "blocker_code",
    "reviewer",
    "reviewed_at",
}
_ROLLOUT_PROTOCOL_KEYS: Final = {
    "seed",
    "hold_required",
    "shakedown_trials",
    "shakedown_speed_scale",
    "scored_trials",
    "nominal_trials",
    "perturbation_trials",
    "trial_duration_s",
    "stable_success_s",
    "target_hz",
    "action_dimensions",
    "replan_actions",
    "ordered_conditions",
    "schedule_order",
    "execution_order",
}
_ROLLOUT_CONDITION_KEYS: Final = {
    "condition_id",
    "stratum",
    "blue_axis",
    "blue_offset_mm",
    "red_axis",
    "red_offset_mm",
}
_CANONICAL_ROLLOUT_TASK: Final = (
    "Move the blue cube, then the red cube, from the white pad on the right "
    "to the gray platform on the left."
)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA_RE.fullmatch(value):
        raise BundleValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_utc_instant(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"{label} must be a nonempty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleValidationError(f"{label} must be an RFC 3339 timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise BundleValidationError(f"{label} must represent UTC")
    return value


def _validate_rollout_protocol(value: Any) -> None:
    protocol = _json_object(value, "rollout_session.trial_protocol")
    _require_keys(protocol, _ROLLOUT_PROTOCOL_KEYS, "rollout_session.trial_protocol")
    expected_scalars = {
        "seed": 1000,
        "hold_required": True,
        "shakedown_trials": 2,
        "shakedown_speed_scale": 0.25,
        "scored_trials": 10,
        "nominal_trials": 6,
        "perturbation_trials": 4,
        "trial_duration_s": 60,
        "stable_success_s": 3.0,
        "target_hz": 30.0,
        "action_dimensions": 7,
        "replan_actions": 10,
    }
    for field, expected in expected_scalars.items():
        if protocol[field] != expected or isinstance(protocol[field], bool) != isinstance(
            expected, bool
        ):
            raise BundleValidationError(
                f"rollout_session.trial_protocol.{field} must equal {expected!r}"
            )
    conditions = protocol["ordered_conditions"]
    if not isinstance(conditions, list) or len(conditions) != 10:
        raise BundleValidationError("rollout_session requires exactly ten ordered conditions")
    expected_conditions = [
        {
            "condition_id": f"nominal_{index:02d}",
            "stratum": "nominal",
            "blue_axis": None,
            "blue_offset_mm": 0.0,
            "red_axis": None,
            "red_offset_mm": 0.0,
        }
        for index in range(1, 7)
    ] + [
        {
            "condition_id": f"robustness_{index:02d}",
            "stratum": "robustness",
            "blue_axis": "pad_x",
            "blue_offset_mm": blue,
            "red_axis": "pad_y",
            "red_offset_mm": red,
        }
        for index, (blue, red) in enumerate(
            ((-25.0, -25.0), (-25.0, 25.0), (25.0, -25.0), (25.0, 25.0)),
            start=1,
        )
    ]
    for index, raw_condition in enumerate(conditions):
        condition = _json_object(raw_condition, f"rollout condition {index}")
        _require_keys(condition, _ROLLOUT_CONDITION_KEYS, f"rollout condition {index}")
        if condition != expected_conditions[index]:
            raise BundleValidationError(
                f"rollout condition {index} differs from the seed-1000 protocol"
            )
    expected_order = [condition["condition_id"] for condition in expected_conditions]
    if protocol["schedule_order"] != expected_order:
        raise BundleValidationError("rollout_session schedule_order is not canonical")
    if protocol["execution_order"] != expected_order:
        raise BundleValidationError("rollout_session execution_order is not canonical")


def _validate_rollout_clearance(
    value: Any,
    *,
    policy: str,
    payload_root: Path,
) -> set[str]:
    if policy == "act":
        if value is not None:
            raise BundleValidationError("ACT cannot authorize its own live session")
        return {"rollout_session.json"}
    clearance = _json_object(value, "rollout_session.act_infrastructure_clearance")
    _require_keys(clearance, _ROLLOUT_CLEARANCE_KEYS, "rollout_session clearance")
    _require_sha256(clearance["act_outcome_sha256"], "ACT outcome hash")
    outcome = payload_root / "act_infrastructure_outcome.json"
    if _hash_regular_file(outcome)[0] != clearance["act_outcome_sha256"]:
        raise BundleValidationError("ACT outcome attachment hash mismatch")
    if clearance["mode"] == "shared_infrastructure_proven":
        if (
            clearance["act_terminal_status"] != "scored"
            or not isinstance(clearance["act_rollout_evidence_bundle_id"], str)
            or not _HEX_SHA_RE.fullmatch(clearance["act_rollout_evidence_bundle_id"])
            or any(
                clearance[field] is not None
                for field in (
                    "attestation_sha256",
                    "blocker_stage",
                    "blocker_code",
                    "reviewer",
                    "reviewed_at",
                )
            )
        ):
            raise BundleValidationError("invalid scored ACT infrastructure clearance")
        return {"rollout_session.json", "act_infrastructure_outcome.json"}
    if clearance["mode"] != "policy_specific_act_blocker":
        raise BundleValidationError("unsupported ACT infrastructure clearance mode")
    if clearance["act_terminal_status"] not in {
        "ineligible_training",
        "ineligible_offline",
        "ineligible_pc_runtime",
        "unsafe_shadow",
        "unsafe_shakedown",
    }:
        raise BundleValidationError("invalid ACT policy-specific terminal status")
    for field in ("blocker_stage", "blocker_code", "reviewer"):
        _validate_nonempty_string(clearance[field], f"ACT clearance {field}")
    _require_utc_instant(clearance["reviewed_at"], "ACT clearance reviewed_at")
    attestation_sha256 = _require_sha256(
        clearance["attestation_sha256"], "ACT clearance attestation hash"
    )
    if clearance["act_rollout_evidence_bundle_id"] is not None:
        raise BundleValidationError("policy-specific ACT clearance cannot cite rollout evidence")
    attestation = payload_root / "act_infrastructure_attestation.json"
    if _hash_regular_file(attestation)[0] != attestation_sha256:
        raise BundleValidationError("ACT blocker attestation attachment hash mismatch")
    return {
        "rollout_session.json",
        "act_infrastructure_outcome.json",
        "act_infrastructure_attestation.json",
    }


def _validate_validated_rollout_session_request(request: SealRequest) -> None:
    if request.kind != "rollout_session" or request.permission != "live_session":
        raise BundleValidationError(
            "typed rollout-session sealing requires kind=rollout_session "
            "and permission=live_session"
        )
    if request.artifact_roots:
        raise BundleValidationError("rollout_session bundles cannot declare artifact roots")
    if request.payload_dir is None:
        raise BundleValidationError("rollout_session requires its typed payload")
    payload_root = _absolute_without_symlinks(request.payload_dir)
    inventory = _inventory_tree(payload_root)
    if inventory["directories"]:
        raise BundleValidationError("rollout_session payload cannot contain directories")
    payload_path = payload_root / "rollout_session.json"
    payload = _json_object(_load_canonical_json(payload_path), "rollout_session payload")
    _require_keys(payload, _ROLLOUT_PAYLOAD_KEYS, "rollout_session payload")
    if payload["schema_version"] != 1:
        raise BundleValidationError("rollout_session payload schema_version must be 1")
    for field in ("policy_bundle_id", "policy_content_id"):
        _require_sha256(payload[field], f"rollout_session.{field}")
    if payload["policy_bundle_id"] != payload["policy_content_id"]:
        raise BundleValidationError("policy bundle/content IDs differ")
    if payload["phase_permissions"] != ["hold", "shakedown", "scored"]:
        raise BundleValidationError("rollout_session phase permissions are not canonical")
    if payload["blockers"] != []:
        raise BundleValidationError("live rollout_session must be blocker-free")
    session_id = _validate_nonempty_string(payload["session_id"], "rollout_session session_id")
    operator = _validate_nonempty_string(payload["operator"], "rollout_session operator")
    if payload["task"] != _CANONICAL_ROLLOUT_TASK:
        raise BundleValidationError("rollout_session task differs from the fixed task")

    setup_hashes = _json_object(payload["setup_hashes"], "rollout_session.setup_hashes")
    _require_keys(setup_hashes, {"calibration", "camera", "robot", "reset"}, "setup hashes")
    for name, value in setup_hashes.items():
        _require_sha256(value, f"setup hash {name}")

    session_inputs = _json_object(
        payload["session_inputs_binding"], "rollout_session.session_inputs_binding"
    )
    _require_keys(session_inputs, _ROLLOUT_SESSION_INPUT_KEYS, "session_inputs binding")
    for name, value in session_inputs.items():
        _require_sha256(value, f"session_inputs {name}")
    if session_inputs["bundle_id"] != session_inputs["content_id"]:
        raise BundleValidationError("session_inputs bundle/content IDs differ")

    source = _json_object(payload["source_session"], "rollout_session.source_session")
    _require_keys(source, _ROLLOUT_SOURCE_KEYS, "rollout_session source_session")
    for name in _ROLLOUT_SOURCE_KEYS - {"created_at_utc"}:
        _require_sha256(source[name], f"source_session {name}")
    _require_utc_instant(source["created_at_utc"], "source_session created_at_utc")

    executor = _json_object(payload["executor"], "rollout_session.executor")
    _require_keys(executor, _ROLLOUT_EXECUTOR_KEYS, "rollout_session executor")
    if executor["repository"] != "starai-viola-lerobot-ops":
        raise BundleValidationError("rollout_session executor repository is not Repo A")
    if not isinstance(executor["repository_commit"], str) or not _GIT_SHA_RE.fullmatch(
        executor["repository_commit"]
    ):
        raise BundleValidationError("rollout_session executor commit is invalid")
    for name in ("entrypoint_sha256", "attestation_sha256"):
        _require_sha256(executor[name], f"executor {name}")
    _require_utc_instant(executor["reviewed_at_utc"], "executor reviewed_at_utc")

    estop = _json_object(payload["estop"], "rollout_session.estop")
    _require_keys(estop, _ROLLOUT_ESTOP_KEYS, "rollout_session estop")
    if estop["operator"] != operator:
        raise BundleValidationError("rollout operator differs from tested E-stop owner")
    _require_utc_instant(estop["tested_at_utc"], "E-stop tested_at_utc")
    _require_sha256(estop["attestation_sha256"], "E-stop attestation hash")
    _validate_rollout_protocol(payload["trial_protocol"])

    lineage = _json_object(request.lineage, "lineage")
    _require_keys(lineage, _ROLLOUT_LINEAGE_KEYS, "rollout_session lineage")
    policy = lineage["policy"]
    if policy not in _ROLLOUT_POLICIES:
        raise BundleValidationError("rollout_session lineage policy is unsupported")
    expected_files = _validate_rollout_clearance(
        payload["act_infrastructure_clearance"], policy=policy, payload_root=payload_root
    )
    actual_files = {entry["path"] for entry in inventory["files"]}
    if actual_files != expected_files:
        raise BundleValidationError(
            f"rollout_session payload files differ; expected={sorted(expected_files)}, "
            f"actual={sorted(actual_files)}"
        )
    if request.subject != session_id:
        raise BundleValidationError("rollout_session subject must equal its session_id")
    expected_pairs = {
        "session_id": session_id,
        "policy_bundle_id": payload["policy_bundle_id"],
        "policy_content_id": payload["policy_content_id"],
        "session_manifest_sha256": source["sha256"],
        "source_session_sha256": source["sha256"],
        "session_inputs_bundle_id": session_inputs["bundle_id"],
        "session_inputs_content_id": session_inputs["content_id"],
        "session_inputs_manifest_sha256": session_inputs["manifest_sha256"],
        "session_inputs_payload_sha256": session_inputs["payload_sha256"],
        "session_inputs_setup_record_inventory_sha256": session_inputs[
            "setup_record_inventory_sha256"
        ],
        "benchmark_lineage_sha256": source["benchmark_lineage_sha256"],
        "policy_bindings_sha256": source["policy_bindings_sha256"],
        "policy_binding_sha256": source["policy_binding_sha256"],
        "physical_setup_binding_sha256": source["physical_setup_binding_sha256"],
        "executor_binding_sha256": source["executor_binding_sha256"],
        "executor_attestation_sha256": executor["attestation_sha256"],
        "estop_attestation_sha256": estop["attestation_sha256"],
        "blockers": [],
        "act_infrastructure_clearance": payload["act_infrastructure_clearance"],
    }
    for field, expected in expected_pairs.items():
        if lineage[field] != expected:
            raise BundleValidationError(f"rollout_session lineage {field} differs from payload")


@dataclass(frozen=True)
class VerifiedBundle:
    path: Path
    manifest: dict[str, Any]
    ready: dict[str, Any]
    artifacts_verified: bool
    receipts: tuple[dict[str, Any], ...]

    @property
    def content_id(self) -> str:
        return self.manifest["content_id"]

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def kind(self) -> str:
        return self.manifest["kind"]

    @property
    def permission(self) -> str:
        return self.manifest["permission"]

    @property
    def is_active(self) -> bool:
        return not any(receipt["status"] in {"rejected", "revoked"} for receipt in self.receipts)

    def payload_file(self, relative: str) -> Path:
        safe = _safe_relative_path(relative, label="payload path").as_posix()
        declared = {entry["path"] for entry in self.manifest["payload"]["files"]}
        if safe not in declared:
            raise BundleValidationError(f"payload file is not declared by the bundle: {safe}")
        return self.path / "payload" / Path(*PurePosixPath(safe).parts)

    def artifact_root(self, name: str) -> Path:
        if not self.artifacts_verified:
            raise BundleValidationError("artifact roots are unavailable until fully verified")
        for artifact in self.manifest["artifacts"]:
            if artifact["name"] == name:
                return Path(artifact["root"])
        raise BundleValidationError(f"bundle has no artifact named {name!r}")

    def accepted_artifact_root(self, name: str) -> Path:
        """Return a verified receiver-local copy of a declared artifact.

        Accepted bundles use a deterministic sibling layout:
        ``<destination_root>/artifacts/<bundle_id>/<name>``.  This method never
        falls back to the source NAS path, so deployment cannot accidentally
        consume mutable/remote bytes after acceptance.
        """

        declared: dict[str, Any] | None = None
        for artifact in self.manifest["artifacts"]:
            if artifact["name"] == name:
                declared = artifact
                break
        if declared is None:
            raise BundleValidationError(f"bundle has no artifact named {name!r}")
        local = self.path.parent.parent / "artifacts" / self.bundle_id / name
        try:
            actual = _inventory_tree(local)
        except BundleValidationError as exc:
            raise BundleValidationError(
                f"bundle has no verified receiver-local artifact copy {name!r}; "
                f"run viola-handoff accept first ({exc})"
            ) from exc
        expected = {key: value for key, value in declared.items() if key not in {"name", "root"}}
        if actual != expected:
            raise BundleValidationError(
                f"receiver-local artifact copy differs from the signed inventory: {name!r}"
            )
        return local


def _validate_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise BundleValidationError(f"{label} must be a nonempty string of at most 512 characters")
    return value


def _resolve_rule(
    *,
    kind: str,
    producer_role: str,
    consumer_role: str | None,
    permission: str | None,
    lineage: Mapping[str, Any],
) -> tuple[str, str]:
    if kind not in KIND_RULES:
        raise BundleValidationError(f"unsupported handoff kind: {kind!r}")
    rule = KIND_RULES[kind]
    if producer_role != rule["producer"]:
        raise BundleValidationError(
            f"{kind} must be produced by {rule['producer']}, not {producer_role}"
        )
    chosen_consumer = consumer_role or rule["consumers"][0]
    if chosen_consumer not in rule["consumers"]:
        raise BundleValidationError(f"{kind} cannot be consumed by {chosen_consumer}")
    chosen_permission = permission or rule["default_permission"]
    if chosen_permission not in rule["permissions"]:
        raise BundleValidationError(f"invalid permission {chosen_permission!r} for {kind}")
    if kind == "rollout_session":
        blockers = lineage.get("blockers")
        if not isinstance(blockers, list) or any(
            not isinstance(item, str) or not item for item in blockers
        ):
            raise BundleValidationError(
                "rollout_session lineage.blockers must be a list of strings"
            )
        if chosen_permission == "live_session" and blockers:
            raise BundleValidationError("live_session permission requires an empty blocker list")
        if chosen_permission == "blocked" and not blockers:
            raise BundleValidationError(
                "blocked rollout_session permission requires at least one blocker"
            )
    return chosen_consumer, chosen_permission


def _artifact_inventories(artifact_roots: Mapping[str, str | Path] | None) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for name, raw_root in sorted((artifact_roots or {}).items()):
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise BundleValidationError(f"unsafe artifact name: {name!r}")
        root = _absolute_without_symlinks(raw_root)
        inventory = _inventory_tree(root)
        artifacts.append({"name": name, "root": str(root), **inventory})
    return artifacts


def _content_material(
    *,
    kind: str,
    experiment: str,
    subject: str,
    lineage: Mapping[str, Any],
    payload: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact_inventories = [
        {key: value for key, value in artifact.items() if key != "root"} for artifact in artifacts
    ]
    return {
        "contract_sha256": CONTRACT_SHA256,
        "kind": kind,
        "experiment": experiment,
        "subject": subject,
        "lineage": lineage,
        "payload": payload,
        "artifact_inventories": artifact_inventories,
    }


def _aggregate_inventory_sha256(manifest: Mapping[str, Any]) -> str:
    value = {
        "payload": manifest["payload"]["inventory_sha256"],
        "artifacts": [
            {"name": artifact["name"], "inventory_sha256": artifact["inventory_sha256"]}
            for artifact in manifest["artifacts"]
        ],
    }
    return _sha256_bytes(canonical_json_bytes(value))


def _assert_retry_compatible(existing: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    if existing["content_id"] != requested["content_id"]:
        raise BundleValidationError("published bundle conflicts with requested content")
    for field in ("kind", "consumer", "permission", "wandb"):
        if existing[field] != requested[field]:
            raise BundleValidationError(f"idempotent retry cannot change manifest field {field!r}")
    if existing["producer"]["role"] != requested["producer"]["role"]:
        raise BundleValidationError("idempotent retry cannot change producer role")


def _validate_manifest(value: Any) -> dict[str, Any]:
    manifest = _json_object(value, "manifest")
    _require_keys(
        manifest,
        {
            "schema_version",
            "contract_sha256",
            "kind",
            "bundle_id",
            "content_id",
            "experiment",
            "subject",
            "producer",
            "consumer",
            "permission",
            "lineage",
            "payload",
            "artifacts",
            "wandb",
            "created_at",
        },
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise BundleValidationError(f"unsupported schema version: {manifest['schema_version']!r}")
    if manifest["contract_sha256"] != CONTRACT_SHA256:
        raise BundleValidationError("bundle uses a different handoff contract")
    kind = manifest["kind"]
    if not isinstance(kind, str):
        raise BundleValidationError("manifest.kind must be a string")
    producer = _validate_identity(manifest["producer"], "manifest.producer")
    consumer = _json_object(manifest["consumer"], "manifest.consumer")
    _require_keys(consumer, {"role"}, "manifest.consumer")
    lineage = _json_object(manifest["lineage"], "manifest.lineage")
    consumer_role, permission = _resolve_rule(
        kind=kind,
        producer_role=producer["role"],
        consumer_role=consumer["role"],
        permission=manifest["permission"],
        lineage=lineage,
    )
    if consumer["role"] != consumer_role or manifest["permission"] != permission:
        raise BundleValidationError("manifest roles/permission do not match the contract")
    experiment = _validate_nonempty_string(manifest["experiment"], "manifest.experiment")
    subject = _validate_nonempty_string(manifest["subject"], "manifest.subject")
    _validate_utc_timestamp(manifest["created_at"], "manifest.created_at")
    payload = _validate_inventory_shape(manifest["payload"], "manifest.payload")
    if payload["byte_count"] > MAX_PAYLOAD_BYTES:
        raise BundleValidationError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")

    if not isinstance(manifest["artifacts"], list):
        raise BundleValidationError("manifest.artifacts must be a list")
    artifacts: list[dict[str, Any]] = []
    previous = ""
    for index, raw_artifact in enumerate(manifest["artifacts"]):
        artifact = _json_object(raw_artifact, f"manifest.artifacts[{index}]")
        _require_keys(
            artifact,
            {
                "name",
                "root",
                "directories",
                "files",
                "file_count",
                "byte_count",
                "inventory_sha256",
            },
            f"manifest.artifacts[{index}]",
        )
        name = artifact["name"]
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or name <= previous:
            raise BundleValidationError("artifact names must be safe, unique, and sorted")
        previous = name
        if not isinstance(artifact["root"], str) or not Path(artifact["root"]).is_absolute():
            raise BundleValidationError("artifact roots must be absolute paths")
        inventory = _validate_inventory_shape(
            {key: value for key, value in artifact.items() if key not in {"name", "root"}},
            f"manifest.artifacts[{index}]",
        )
        artifacts.append({"name": name, "root": artifact["root"], **inventory})

    wandb = _json_object(manifest["wandb"], "manifest.wandb")
    _require_keys(wandb, {"project", "run_id"}, "manifest.wandb")
    _validate_nonempty_string(wandb["project"], "manifest.wandb.project")

    material = _content_material(
        kind=kind,
        experiment=experiment,
        subject=subject,
        lineage=lineage,
        payload=payload,
        artifacts=artifacts,
    )
    content_id = _sha256_bytes(canonical_json_bytes(material))
    if manifest["content_id"] != content_id or manifest["bundle_id"] != content_id:
        raise BundleValidationError("content_id/bundle_id does not match bundle content")
    if wandb["run_id"] != wandb_run_id(content_id):
        raise BundleValidationError("manifest has a non-deterministic W&B run ID")
    return manifest


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as exc:
        raise HandoffError(f"cannot create immutable file {path}: {exc}") from exc
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_bytes_exclusive(path, canonical_json_bytes(value))


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_ready_atomic(bundle: Path, ready: Mapping[str, Any]) -> None:
    target = bundle / "READY.json"
    if target.exists():
        existing = _load_canonical_json(target)
        if existing != ready:
            # A retry can have a later ready_at or a newly resolved URL; the
            # immutable existing marker remains authoritative if its binding is
            # otherwise valid.
            for key in (
                "schema_version",
                "bundle_id",
                "content_id",
                "manifest_sha256",
                "inventory_sha256",
            ):
                if existing.get(key) != ready.get(key):
                    raise BundleValidationError("existing READY marker conflicts with bundle")
        return
    temporary = bundle / f".READY.{uuid.uuid4().hex}.partial"
    _write_json_exclusive(temporary, ready)
    try:
        os.replace(temporary, target)
        _fsync_directory(bundle)
    except Exception:
        raise


def _copy_regular_file(source: Path, destination: Path) -> None:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, source_flags)
        destination_fd = os.open(destination, destination_flags, 0o644)
    except OSError as exc:
        if "source_fd" in locals():
            os.close(source_fd)
        raise HandoffError(f"cannot copy immutable file {source}: {exc}") from exc
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise BundleValidationError(f"copy source is not a regular file: {source}")
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                view = view[os.write(destination_fd, view) :]
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)


def _copy_inventory(source: Path, destination: Path, inventory: Mapping[str, Any]) -> None:
    destination.mkdir(mode=0o755)
    for relative in inventory["directories"]:
        safe = _safe_relative_path(relative)
        (destination / Path(*safe.parts)).mkdir(mode=0o755)
    for entry in inventory["files"]:
        safe = _safe_relative_path(entry["path"])
        target = destination / Path(*safe.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_regular_file(source / Path(*safe.parts), target)
    copied = _inventory_tree(destination)
    if copied != inventory:
        raise BundleValidationError("source changed or copy failed inventory verification")


def _validate_top_level(bundle: Path, *, ready_required: bool) -> None:
    expected = {"manifest.json", "payload", "receipts"}
    if ready_required:
        expected.add("READY.json")
    try:
        entries = list(os.scandir(bundle))
    except OSError as exc:
        raise BundleValidationError(f"cannot enumerate bundle {bundle}: {exc}") from exc
    actual = {entry.name for entry in entries}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BundleValidationError(f"bundle files differ; missing={missing}, extra={extra}")
    for entry in entries:
        if entry.is_symlink():
            raise BundleValidationError(f"symlink is forbidden in bundle: {entry.path}")
        if entry.name in {"payload", "receipts"} and not entry.is_dir(follow_symlinks=False):
            raise BundleValidationError(f"bundle entry must be a directory: {entry.path}")
        if entry.name not in {"payload", "receipts"} and not entry.is_file(follow_symlinks=False):
            raise BundleValidationError(f"bundle entry must be a regular file: {entry.path}")


def _validate_receipt(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    filename: str,
) -> dict[str, Any]:
    receipt = _json_object(value, "receipt")
    _require_keys(
        receipt,
        {
            "schema_version",
            "receipt_key",
            "receipt_sha256",
            "bundle_id",
            "content_id",
            "status",
            "actor",
            "accepted_artifacts",
            "note",
            "created_at",
            "wandb_run_id",
        },
        "receipt",
    )
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise BundleValidationError("receipt schema version does not match")
    bundle_id = manifest["bundle_id"]
    if receipt["bundle_id"] != bundle_id or receipt["content_id"] != bundle_id:
        raise BundleValidationError("receipt is bound to another bundle")
    if receipt["status"] not in RECEIPT_STATUSES:
        raise BundleValidationError("receipt has an invalid status")
    actor = _validate_identity(receipt["actor"], "receipt.actor")
    producer_role = manifest["producer"]["role"]
    consumer_role = manifest["consumer"]["role"]
    if receipt["status"] in {"accepted", "rejected"} and actor["role"] != consumer_role:
        raise BundleValidationError(
            f"{receipt['status']} receipt must be authored by consumer {consumer_role}"
        )
    if receipt["status"] == "revoked" and actor["role"] not in {
        producer_role,
        consumer_role,
    }:
        raise BundleValidationError("revocation must be authored by the producer or consumer")
    accepted_artifacts = _json_object(receipt["accepted_artifacts"], "receipt.accepted_artifacts")
    if receipt["status"] != "accepted" and accepted_artifacts:
        raise BundleValidationError("only an accepted receipt may bind receiver-local artifacts")
    if receipt["status"] == "accepted":
        expected_names = {artifact["name"] for artifact in manifest["artifacts"]}
        if set(accepted_artifacts) != expected_names:
            raise BundleValidationError(
                "accepted receipt artifact names differ from the signed manifest"
            )
        declared = {artifact["name"]: artifact for artifact in manifest["artifacts"]}
        for name, raw_details in accepted_artifacts.items():
            details = _json_object(raw_details, f"receipt.accepted_artifacts.{name}")
            _require_keys(
                details,
                {"root", "inventory_sha256", "file_count", "byte_count"},
                f"receipt.accepted_artifacts.{name}",
            )
            if not isinstance(details["root"], str) or not Path(details["root"]).is_absolute():
                raise BundleValidationError("accepted artifact roots must be absolute")
            signed = declared[name]
            for field in ("inventory_sha256", "file_count", "byte_count"):
                if details[field] != signed[field]:
                    raise BundleValidationError(
                        f"accepted receipt {name}.{field} differs from signed inventory"
                    )
    if not isinstance(receipt["note"], str) or len(receipt["note"]) > 2000:
        raise BundleValidationError("receipt note must be a string of at most 2000 characters")
    _validate_utc_timestamp(receipt["created_at"], "receipt.created_at")
    if receipt["wandb_run_id"] != wandb_run_id(bundle_id):
        raise BundleValidationError("receipt W&B run ID does not match")
    semantic = {
        "schema_version": receipt["schema_version"],
        "bundle_id": receipt["bundle_id"],
        "content_id": receipt["content_id"],
        "status": receipt["status"],
        "actor": receipt["actor"],
        "accepted_artifacts": receipt["accepted_artifacts"],
        "note": receipt["note"],
        "wandb_run_id": receipt["wandb_run_id"],
    }
    key = _sha256_bytes(canonical_json_bytes(semantic))
    if receipt["receipt_key"] != key or filename != f"{key}.json":
        raise BundleValidationError("receipt key/filename does not match its semantics")
    without_digest = {key_: value_ for key_, value_ in receipt.items() if key_ != "receipt_sha256"}
    if receipt["receipt_sha256"] != _sha256_bytes(canonical_json_bytes(without_digest)):
        raise BundleValidationError("receipt digest does not match its content")
    return receipt


def _inspect_receipts(
    receipts_dir: Path, *, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    receipts: list[dict[str, Any]] = []
    for entry in sorted(os.scandir(receipts_dir), key=lambda item: item.name):
        if (
            entry.is_symlink()
            or not entry.is_file(follow_symlinks=False)
            or not entry.name.endswith(".json")
        ):
            raise BundleValidationError(f"invalid receipt entry: {entry.path}")
        receipts.append(
            _validate_receipt(
                _load_canonical_json(Path(entry.path)),
                manifest=manifest,
                filename=entry.name,
            )
        )
    return tuple(receipts)


def _inspect_bundle(
    bundle: str | Path,
    *,
    verify_artifacts: bool,
    ready_required: bool,
    enforce_directory_name: bool,
) -> VerifiedBundle:
    bundle_path = _absolute_without_symlinks(bundle)
    if not bundle_path.is_dir():
        raise BundleValidationError(f"bundle path is not a directory: {bundle_path}")
    _validate_top_level(bundle_path, ready_required=ready_required)
    manifest_path = bundle_path / "manifest.json"
    manifest = _validate_manifest(_load_canonical_json(manifest_path))
    manifest_sha256 = _sha256_bytes(manifest_path.read_bytes())
    if enforce_directory_name and bundle_path.name != manifest["bundle_id"]:
        raise BundleValidationError("bundle directory name does not match bundle_id")
    if enforce_directory_name and bundle_path.parent.name != manifest["kind"]:
        raise BundleValidationError("bundle parent directory does not match kind")

    payload_actual = _inventory_tree(bundle_path / "payload")
    if payload_actual != manifest["payload"]:
        raise BundleValidationError("payload inventory differs from manifest")

    if verify_artifacts:
        local_root = bundle_path.parent.parent / "artifacts" / manifest["bundle_id"]
        if local_root.exists():
            _verify_local_artifacts(destination_root=bundle_path.parent.parent, manifest=manifest)
        else:
            for artifact in manifest["artifacts"]:
                root = _absolute_without_symlinks(artifact["root"])
                actual = _inventory_tree(root)
                expected = {
                    key: value for key, value in artifact.items() if key not in {"name", "root"}
                }
                if actual != expected:
                    raise BundleValidationError(f"artifact inventory differs: {artifact['name']}")

    receipts = _inspect_receipts(bundle_path / "receipts", manifest=manifest)
    if ready_required:
        ready = _json_object(_load_canonical_json(bundle_path / "READY.json"), "READY")
        _require_keys(
            ready,
            {
                "schema_version",
                "bundle_id",
                "content_id",
                "manifest_sha256",
                "inventory_sha256",
                "wandb_url",
                "ready_at",
            },
            "READY",
        )
        if ready["schema_version"] != SCHEMA_VERSION:
            raise BundleValidationError("READY schema version does not match")
        if (
            ready["bundle_id"] != manifest["bundle_id"]
            or ready["content_id"] != manifest["content_id"]
        ):
            raise BundleValidationError("READY is bound to another bundle")
        if ready["manifest_sha256"] != manifest_sha256:
            raise BundleValidationError("manifest digest differs from READY")
        if ready["inventory_sha256"] != _aggregate_inventory_sha256(manifest):
            raise BundleValidationError("aggregate inventory digest differs from READY")
        _validate_ready_wandb_url(ready["wandb_url"], manifest=manifest)
        _validate_nonempty_string(ready["ready_at"], "READY.ready_at")
    else:
        ready = {}
    return VerifiedBundle(
        path=bundle_path,
        manifest=manifest,
        ready=ready,
        artifacts_verified=verify_artifacts,
        receipts=receipts,
    )


def _validate_ready_wandb_url(value: Any, *, manifest: Mapping[str, Any]) -> None:
    """Bind mutable READY metadata to the manifest's deterministic W&B run."""

    if not isinstance(value, str):
        raise BundleValidationError("READY.wandb_url must be the canonical W&B run URL")
    parsed = urlsplit(value)
    parts = parsed.path.split("/")
    wandb = manifest["wandb"]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "wandb.ai"
        or parsed.query
        or parsed.fragment
        or len(parts) != 5
        or parts[0] != ""
        or not parts[1]
        or parts[2] != wandb["project"]
        or parts[3] != "runs"
        or parts[4] != wandb["run_id"]
    ):
        raise BundleValidationError(
            "READY.wandb_url differs from the manifest W&B project/run identity"
        )


def inspect_bundle(
    bundle: str | Path,
    *,
    verify_artifacts: bool = True,
    expected_kind: str | None = None,
    required_permission: str | None = None,
    allow_inactive: bool = False,
) -> VerifiedBundle:
    """Fully verify a READY bundle before consuming any of its paths."""

    verified = _inspect_bundle(
        bundle,
        verify_artifacts=verify_artifacts,
        ready_required=True,
        enforce_directory_name=True,
    )
    if expected_kind is not None and verified.kind != expected_kind:
        raise BundleValidationError(
            f"expected bundle kind {expected_kind!r}, found {verified.kind!r}"
        )
    if required_permission is not None and verified.permission != required_permission:
        raise BundleValidationError(
            f"expected permission {required_permission!r}, found {verified.permission!r}"
        )
    if not allow_inactive:
        inactive = [
            receipt["status"]
            for receipt in verified.receipts
            if receipt["status"] in {"rejected", "revoked"}
        ]
        if inactive:
            raise BundleValidationError(
                f"bundle is inactive due to append-only receipt status: {inactive[0]}"
            )
    return verified


verify_bundle = inspect_bundle


def require_active_canonical_source(
    accepted: VerifiedBundle,
    *,
    handoff_root: str | Path = DEFAULT_HANDOFF_ROOT,
    expected_kind: str | None = None,
    required_permission: str | None = None,
) -> VerifiedBundle:
    """Prove that an accepted snapshot's canonical source is still active.

    Acceptance deliberately preserves an immutable receiver-local snapshot and
    receiver-local artifact copies.  A later rejected or revoked receipt is,
    however, appended to the canonical NAS bundle and cannot retroactively
    appear in that snapshot.  Safety-authorizing consumers must call this gate
    immediately before granting motion.

    Both the local metadata and the canonical small bundle are reopened.  The
    canonical source must be available, active, and byte-semantically identical
    in its manifest and READY record.  Artifact bytes are *not* resolved from
    the canonical source: callers continue using the already verified
    receiver-local paths held by ``accepted``.

    Non-motion inspection may continue to use :func:`inspect_bundle` directly
    when offline operation is intentional.
    """

    if not isinstance(accepted, VerifiedBundle):
        raise BundleValidationError(
            "canonical source freshness requires a previously verified bundle"
        )
    kind = expected_kind or accepted.kind
    permission = required_permission or accepted.permission
    try:
        local = inspect_bundle(
            accepted.path,
            verify_artifacts=False,
            expected_kind=kind,
            required_permission=permission,
        )
    except (HandoffError, OSError) as exc:
        raise BundleValidationError(
            f"accepted bundle metadata is unavailable or invalid: {exc}"
        ) from exc
    if (
        local.bundle_id != accepted.bundle_id
        or local.content_id != accepted.content_id
        or local.manifest != accepted.manifest
        or local.ready != accepted.ready
    ):
        raise BundleValidationError("accepted bundle changed after its original verification")

    canonical_path = resolve_bundle(handoff_root, kind, local.bundle_id)
    try:
        canonical = inspect_bundle(
            canonical_path,
            verify_artifacts=False,
            expected_kind=kind,
            required_permission=permission,
        )
    except (HandoffError, OSError) as exc:
        raise BundleValidationError(
            f"canonical handoff source is unavailable, inactive, or invalid: {exc}"
        ) from exc
    if (
        canonical.bundle_id != local.bundle_id
        or canonical.content_id != local.content_id
        or canonical.manifest != local.manifest
        or canonical.ready != local.ready
    ):
        raise BundleValidationError("canonical handoff source differs from the accepted snapshot")
    return accepted


def resolve_bundle(root: str | Path, kind: str, bundle_id: str) -> Path:
    if kind not in KIND_RULES:
        raise BundleValidationError(f"unsupported handoff kind: {kind!r}")
    if not _HEX_SHA_RE.fullmatch(bundle_id):
        raise BundleValidationError("bundle_id must be a lowercase SHA-256 digest")
    return Path(os.path.abspath(os.path.expanduser(os.fspath(root)))) / kind / bundle_id


def _event_metadata(
    verified: VerifiedBundle,
    *,
    actor: Mapping[str, Any] | None = None,
    accepted_artifact_roots: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "bundle_id": verified.bundle_id,
        "content_id": verified.content_id,
        "kind": verified.kind,
        "permission": verified.permission,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(verified.manifest)),
        "inventory_sha256": _aggregate_inventory_sha256(verified.manifest),
        "contract_sha256": CONTRACT_SHA256,
    }
    if actor is not None:
        metadata["actor_role"] = actor["role"]
        metadata["actor_commit"] = actor["repository_commit"]
    if accepted_artifact_roots is not None:
        metadata["accepted_artifact_count"] = len(accepted_artifact_roots)
        metadata["accepted_artifact_layout"] = "artifacts/<bundle_id>/<name>"
    return metadata


def _make_ready(verified: VerifiedBundle, *, wandb_url: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": verified.bundle_id,
        "content_id": verified.content_id,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(verified.manifest)),
        "inventory_sha256": _aggregate_inventory_sha256(verified.manifest),
        "wandb_url": wandb_url,
        "ready_at": _utc_now(),
    }


def _required_evidence_url(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise EvidenceError("online W&B evidence did not return an HTTPS run URL")
    return value


def seal_bundle(
    request: SealRequest,
    *,
    evidence_logger: EvidenceLogger | None = None,
) -> VerifiedBundle:
    """Seal a generic bundle without granting live robot authority.

    ``rollout_session/live_session`` is intentionally excluded.  Its only
    public construction path is :func:`seal_validated_rollout_session_bundle`,
    which validates the complete portable authorization envelope first.
    """

    return _seal_bundle(request, evidence_logger=evidence_logger, validated_live=False)


def seal_validated_rollout_session_bundle(
    request: SealRequest,
    *,
    evidence_logger: EvidenceLogger | None = None,
) -> VerifiedBundle:
    """Seal a live rollout authorization after exact typed-envelope validation."""

    _validate_validated_rollout_session_request(request)
    return _seal_bundle(request, evidence_logger=evidence_logger, validated_live=True)


def _seal_bundle(
    request: SealRequest,
    *,
    evidence_logger: EvidenceLogger | None,
    validated_live: bool,
) -> VerifiedBundle:
    """Seal one content-addressed bundle and publish READY only after W&B succeeds."""

    producer = _identity_dict(request.producer)
    lineage = _json_object(request.lineage, "lineage")
    consumer_role, permission = _resolve_rule(
        kind=request.kind,
        producer_role=producer["role"],
        consumer_role=request.consumer_role,
        permission=request.permission,
        lineage=lineage,
    )
    if request.kind == "rollout_session" and permission == "live_session" and not validated_live:
        raise BundleValidationError(
            "generic sealing cannot grant live_session authority; use the typed validated "
            "rollout-session sealer"
        )
    experiment = _validate_nonempty_string(request.experiment, "experiment")
    subject = _validate_nonempty_string(request.subject, "subject")
    project = _validate_nonempty_string(request.wandb_project, "wandb_project")

    if request.payload_dir is None:
        payload: dict[str, Any] = {
            "directories": [],
            "files": [],
            "file_count": 0,
            "byte_count": 0,
            "inventory_sha256": _sha256_bytes(
                canonical_json_bytes({"directories": [], "files": []})
            ),
        }
        payload_source = None
    else:
        payload_source = _absolute_without_symlinks(request.payload_dir)
        payload = _inventory_tree(payload_source)
    if payload["byte_count"] > MAX_PAYLOAD_BYTES:
        raise BundleValidationError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    artifacts = _artifact_inventories(request.artifact_roots)
    material = _content_material(
        kind=request.kind,
        experiment=experiment,
        subject=subject,
        lineage=lineage,
        payload=payload,
        artifacts=artifacts,
    )
    content_id = _sha256_bytes(canonical_json_bytes(material))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "kind": request.kind,
        "bundle_id": content_id,
        "content_id": content_id,
        "experiment": experiment,
        "subject": subject,
        "producer": producer,
        "consumer": {"role": consumer_role},
        "permission": permission,
        "lineage": lineage,
        "payload": payload,
        "artifacts": artifacts,
        "wandb": {"project": project, "run_id": wandb_run_id(content_id)},
        "created_at": request.created_at or _utc_now(),
    }
    _validate_manifest(manifest)

    root = _absolute_without_symlinks(request.root, must_exist=False)
    root.mkdir(parents=True, exist_ok=True)
    root = _absolute_without_symlinks(root)
    kind_root = root / request.kind
    kind_root.mkdir(mode=0o755, exist_ok=True)
    _absolute_without_symlinks(kind_root)
    target = kind_root / content_id

    def finish_promoted(promoted: Path) -> VerifiedBundle:
        promoted = _absolute_without_symlinks(promoted)
        # Recover only our own interrupted READY temp files.  They never make a
        # bundle consumable and cannot carry arbitrary path names.
        if not (promoted / "READY.json").exists():
            ready_temp_re = re.compile(r"^\.READY\.[0-9a-f]{32}\.partial$")
            for entry in os.scandir(promoted):
                if ready_temp_re.fullmatch(entry.name):
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise BundleValidationError(f"unsafe interrupted READY file: {entry.path}")
                    Path(entry.path).unlink()
        unready = _inspect_bundle(
            promoted,
            verify_artifacts=True,
            ready_required=False,
            enforce_directory_name=True,
        )
        # created_at and producer machine/commit metadata may differ on a later
        # retry.  Security-relevant routing and W&B identity may not.
        _assert_retry_compatible(unready.manifest, manifest)
        logger = required_evidence_logger(evidence_logger)
        url = _required_evidence_url(
            logger.record(
                project=unready.manifest["wandb"]["project"],
                run_id=unready.manifest["wandb"]["run_id"],
                event="sealed",
                metadata=_event_metadata(unready, actor=unready.manifest["producer"]),
            )
        )
        _write_ready_atomic(promoted, _make_ready(unready, wandb_url=url))
        return inspect_bundle(promoted, verify_artifacts=True)

    if target.exists():
        try:
            existing = inspect_bundle(target, verify_artifacts=True)
        except BundleValidationError:
            # A process can be interrupted after atomic promotion but before its
            # W&B-backed READY marker.  Only that exact, valid unready layout is
            # recoverable; corrupted targets are never overwritten.
            return finish_promoted(target)
        _assert_retry_compatible(existing.manifest, manifest)
        return existing

    staging = kind_root / f".{content_id}.{uuid.uuid4().hex}.partial"
    staging.mkdir(mode=0o755)
    (staging / "receipts").mkdir(mode=0o755)
    if payload_source is None:
        (staging / "payload").mkdir(mode=0o755)
    else:
        _copy_inventory(payload_source, staging / "payload", payload)
    _write_json_exclusive(staging / "manifest.json", manifest)
    _inspect_bundle(
        staging,
        verify_artifacts=True,
        ready_required=False,
        enforce_directory_name=False,
    )
    _fsync_directory(staging)
    try:
        os.rename(staging, target)
        _fsync_directory(kind_root)
    except FileExistsError:
        return finish_promoted(target)
    return finish_promoted(target)


def _build_receipt(
    verified: VerifiedBundle,
    *,
    status: str,
    actor: RuntimeIdentity,
    note: str,
    accepted_artifact_roots: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if status not in RECEIPT_STATUSES:
        raise BundleValidationError(f"invalid receipt status: {status!r}")
    if not isinstance(note, str) or len(note) > 2000:
        raise BundleValidationError("receipt note must be at most 2000 characters")
    actor_value = _identity_dict(actor)
    producer_role = verified.manifest["producer"]["role"]
    consumer_role = verified.manifest["consumer"]["role"]
    if status in {"accepted", "rejected"} and actor_value["role"] != consumer_role:
        raise BundleValidationError(
            f"{status} receipt must be authored by consumer {consumer_role}"
        )
    if status == "revoked" and actor_value["role"] not in {producer_role, consumer_role}:
        raise BundleValidationError("revocation must be authored by the producer or consumer")
    if status == "accepted":
        roots = dict(accepted_artifact_roots or {})
        expected_names = {artifact["name"] for artifact in verified.manifest["artifacts"]}
        if set(roots) != expected_names:
            raise BundleValidationError("accepted receipt must bind every receiver-local artifact")
        accepted_artifacts = {
            artifact["name"]: {
                "root": roots[artifact["name"]],
                "inventory_sha256": artifact["inventory_sha256"],
                "file_count": artifact["file_count"],
                "byte_count": artifact["byte_count"],
            }
            for artifact in verified.manifest["artifacts"]
        }
    else:
        if accepted_artifact_roots:
            raise BundleValidationError(
                "only an accepted receipt may bind receiver-local artifacts"
            )
        accepted_artifacts = {}
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": verified.bundle_id,
        "content_id": verified.content_id,
        "status": status,
        "actor": actor_value,
        "accepted_artifacts": accepted_artifacts,
        "note": note,
        "wandb_run_id": verified.manifest["wandb"]["run_id"],
    }
    receipt = {
        **semantic,
        "receipt_key": _sha256_bytes(canonical_json_bytes(semantic)),
        "created_at": _utc_now(),
    }
    receipt["receipt_sha256"] = _sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def _append_receipt(
    bundle: Path, receipt: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> Path:
    destination = bundle / "receipts" / f"{receipt['receipt_key']}.json"
    if destination.exists():
        existing = _validate_receipt(
            _load_canonical_json(destination),
            manifest=manifest,
            filename=destination.name,
        )
        semantic_keys = {
            "schema_version",
            "bundle_id",
            "content_id",
            "status",
            "actor",
            "accepted_artifacts",
            "note",
            "wandb_run_id",
            "receipt_key",
        }
        if any(existing[key] != receipt[key] for key in semantic_keys):
            raise BundleValidationError("existing receipt conflicts with acknowledgement")
        return destination
    try:
        _write_json_exclusive(destination, receipt)
    except HandoffError:
        if destination.exists():
            return _append_receipt(bundle, receipt, manifest=manifest)
        raise
    _fsync_directory(destination.parent)
    return destination


def _copy_bundle_snapshot(source: VerifiedBundle, staging: Path) -> None:
    staging.mkdir(mode=0o755)
    (staging / "receipts").mkdir(mode=0o755)
    _copy_regular_file(source.path / "manifest.json", staging / "manifest.json")
    _copy_regular_file(source.path / "READY.json", staging / "READY.json")
    _copy_inventory(source.path / "payload", staging / "payload", source.manifest["payload"])
    for entry in sorted(os.scandir(source.path / "receipts"), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise BundleValidationError(f"invalid receipt while copying: {entry.path}")
        _copy_regular_file(Path(entry.path), staging / "receipts" / entry.name)


def _artifact_inventory_only(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key not in {"name", "root"}}


def _artifact_target_root(destination_root: Path, bundle_id: str) -> Path:
    return destination_root / "artifacts" / bundle_id


def _verify_local_artifacts(
    *, destination_root: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    local_roots: dict[str, str] = {}
    root = _artifact_target_root(destination_root, manifest["bundle_id"])
    if not manifest["artifacts"]:
        if root.exists():
            try:
                entries = list(os.scandir(_absolute_without_symlinks(root)))
            except OSError as exc:
                raise BundleValidationError(
                    f"cannot inspect accepted artifact root: {exc}"
                ) from exc
            if entries:
                raise BundleValidationError("accepted artifact root contains undeclared entries")
        return local_roots
    root = _absolute_without_symlinks(root)
    actual_names: set[str] = set()
    for entry in os.scandir(root):
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise BundleValidationError(f"invalid receiver-local artifact entry: {entry.path}")
        actual_names.add(entry.name)
    expected_names = {artifact["name"] for artifact in manifest["artifacts"]}
    if actual_names != expected_names:
        raise BundleValidationError(
            "receiver-local artifact names differ; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for artifact in manifest["artifacts"]:
        local = root / artifact["name"]
        actual = _inventory_tree(local)
        if actual != _artifact_inventory_only(artifact):
            raise BundleValidationError(
                f"receiver-local artifact inventory differs: {artifact['name']}"
            )
        local_roots[artifact["name"]] = str(local)
    return local_roots


def _stage_local_artifacts(
    *, incoming: VerifiedBundle, destination_root: Path
) -> tuple[Path | None, dict[str, str]]:
    final_root = _artifact_target_root(destination_root, incoming.bundle_id)
    if final_root.exists():
        return None, _verify_local_artifacts(
            destination_root=destination_root, manifest=incoming.manifest
        )
    if not incoming.manifest["artifacts"]:
        return None, {}

    artifacts_parent = destination_root / "artifacts"
    artifacts_parent.mkdir(mode=0o755, exist_ok=True)
    artifacts_parent = _absolute_without_symlinks(artifacts_parent)
    staging = artifacts_parent / f".{incoming.bundle_id}.{uuid.uuid4().hex}.partial"
    staging.mkdir(mode=0o755)
    try:
        for artifact in incoming.manifest["artifacts"]:
            try:
                source = incoming.accepted_artifact_root(artifact["name"])
            except BundleValidationError:
                source = _absolute_without_symlinks(artifact["root"])
            _copy_inventory(
                source,
                staging / artifact["name"],
                _artifact_inventory_only(artifact),
            )
        # Verify the staging root with the same inventory loop used after
        # promotion, but without relying on the final path existing yet.
        for artifact in incoming.manifest["artifacts"]:
            actual = _inventory_tree(staging / artifact["name"])
            if actual != _artifact_inventory_only(artifact):
                raise BundleValidationError(
                    f"staged receiver-local artifact differs: {artifact['name']}"
                )
        _fsync_directory(staging)
    except Exception:
        # Deliberately preserve `.partial` for forensic inspection and safe retry.
        # It is never resolved by accepted_artifact_root or deployment code.
        raise
    local_roots = {
        artifact["name"]: str(final_root / artifact["name"])
        for artifact in incoming.manifest["artifacts"]
    }
    return staging, local_roots


def _promote_local_artifacts(
    *, staging: Path | None, destination_root: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    final_root = _artifact_target_root(destination_root, manifest["bundle_id"])
    if staging is not None:
        try:
            os.rename(staging, final_root)
            _fsync_directory(final_root.parent)
        except FileExistsError:
            pass
    return _verify_local_artifacts(destination_root=destination_root, manifest=manifest)


def _accepted_bundle_matches(
    *, target: Path, incoming: VerifiedBundle, destination_root: Path
) -> VerifiedBundle:
    accepted = inspect_bundle(target, verify_artifacts=True)
    if accepted.manifest != incoming.manifest or accepted.ready != incoming.ready:
        raise BundleValidationError("accepted target differs from the incoming signed bundle")
    if accepted.manifest["payload"] != incoming.manifest["payload"]:
        raise BundleValidationError("accepted target payload declaration differs from source")
    _verify_local_artifacts(destination_root=destination_root, manifest=accepted.manifest)
    return accepted


def accept_bundle(
    source: str | Path,
    destination_root: str | Path,
    *,
    receiver: RuntimeIdentity,
    evidence_logger: EvidenceLogger | None = None,
    note: str = "",
) -> VerifiedBundle:
    """Verify, checksum-copy, atomically promote, and acknowledge a bundle."""

    incoming_path = _absolute_without_symlinks(source)
    candidate = inspect_bundle(
        incoming_path,
        verify_artifacts=False,
    )
    receiver_value = _identity_dict(receiver)
    if receiver_value["role"] != candidate.manifest["consumer"]["role"]:
        addressed_role = candidate.manifest["consumer"]["role"]
        raise BundleValidationError(
            f"bundle is addressed to {addressed_role}, not {receiver_value['role']}"
        )
    destination_root_path = _absolute_without_symlinks(destination_root, must_exist=False)
    destination_root_path.mkdir(parents=True, exist_ok=True)
    destination_root_path = _absolute_without_symlinks(destination_root_path)
    kind_root = destination_root_path / candidate.kind
    kind_root.mkdir(mode=0o755, exist_ok=True)
    target = kind_root / candidate.bundle_id
    if os.path.abspath(source) == os.path.abspath(target):
        raise BundleValidationError("acceptance destination must differ from the source bundle")

    if target.exists():
        # An idempotent retry is proven by the signed small source bundle plus
        # the independently verified, already-promoted receiver bundle and its
        # local artifacts. The original NAS artifact may now be offline.
        incoming = candidate
    else:
        incoming = inspect_bundle(incoming_path, verify_artifacts=True)

    logger = required_evidence_logger(evidence_logger)
    artifact_staging, local_artifact_roots = _stage_local_artifacts(
        incoming=incoming, destination_root=destination_root_path
    )
    if target.exists():
        accepted = _accepted_bundle_matches(
            target=target, incoming=incoming, destination_root=destination_root_path
        )
        if accepted.content_id != incoming.content_id:
            raise BundleValidationError("accepted target conflicts with incoming bundle")
        receipt = _build_receipt(
            accepted,
            status="accepted",
            actor=receiver,
            note=note,
            accepted_artifact_roots=local_artifact_roots,
        )
        receipt_target = accepted.path / "receipts" / f"{receipt['receipt_key']}.json"
        if not receipt_target.exists():
            _required_evidence_url(
                logger.record(
                    project=incoming.manifest["wandb"]["project"],
                    run_id=incoming.manifest["wandb"]["run_id"],
                    event="accepted",
                    metadata=_event_metadata(
                        accepted,
                        actor=receiver_value,
                        accepted_artifact_roots=local_artifact_roots,
                    ),
                )
            )
        local_artifact_roots = _promote_local_artifacts(
            staging=artifact_staging,
            destination_root=destination_root_path,
            manifest=incoming.manifest,
        )
        if not receipt_target.exists():
            _append_receipt(accepted.path, receipt, manifest=accepted.manifest)
    else:
        staging = kind_root / f".{incoming.bundle_id}.{uuid.uuid4().hex}.partial"
        _copy_bundle_snapshot(incoming, staging)
        staged = _inspect_bundle(
            staging,
            # Incoming artifact bytes and the receiver staging copy were both
            # checksum-verified above. This temporary bundle has no deterministic
            # local-artifact sibling until it is promoted.
            verify_artifacts=False,
            ready_required=True,
            enforce_directory_name=False,
        )
        # Full artifacts are copied and re-hashed before the W&B acceptance
        # receipt.  They are still hidden `.partial` data at this point.
        _required_evidence_url(
            logger.record(
                project=incoming.manifest["wandb"]["project"],
                run_id=incoming.manifest["wandb"]["run_id"],
                event="accepted",
                metadata=_event_metadata(
                    staged,
                    actor=receiver_value,
                    accepted_artifact_roots=local_artifact_roots,
                ),
            )
        )
        receipt = _build_receipt(
            staged,
            status="accepted",
            actor=receiver,
            note=note,
            accepted_artifact_roots=local_artifact_roots,
        )
        _append_receipt(staging, receipt, manifest=staged.manifest)
        _inspect_bundle(
            staging,
            verify_artifacts=False,
            ready_required=True,
            enforce_directory_name=False,
        )
        try:
            local_artifact_roots = _promote_local_artifacts(
                staging=artifact_staging,
                destination_root=destination_root_path,
                manifest=incoming.manifest,
            )
            os.rename(staging, target)
            _fsync_directory(kind_root)
        except FileExistsError:
            pass
        except Exception:
            final_artifact_root = _artifact_target_root(destination_root_path, incoming.bundle_id)
            if (
                artifact_staging is not None
                and final_artifact_root.exists()
                and not artifact_staging.exists()
            ):
                os.rename(final_artifact_root, artifact_staging)
                _fsync_directory(final_artifact_root.parent)
            raise
        accepted = _accepted_bundle_matches(
            target=target, incoming=incoming, destination_root=destination_root_path
        )
        if accepted.content_id != incoming.content_id:
            raise BundleValidationError("concurrent accepted target conflicts with incoming bundle")

    receipt = _build_receipt(
        accepted,
        status="accepted",
        actor=receiver,
        note=note,
        accepted_artifact_roots=local_artifact_roots,
    )
    target_receipt = _append_receipt(accepted.path, receipt, manifest=accepted.manifest)
    # Promotion makes the receiver's receipt authoritative.  Propagate that
    # exact object back to the source, so an interruption can only leave a
    # missing source receipt, never an accepted target without its receipt.
    authoritative_receipt = _load_canonical_json(target_receipt)
    _append_receipt(incoming.path, authoritative_receipt, manifest=incoming.manifest)
    result = _accepted_bundle_matches(
        target=accepted.path, incoming=incoming, destination_root=destination_root_path
    )
    for name in local_artifact_roots:
        result.accepted_artifact_root(name)
    return result


def ack_bundle(
    bundle: str | Path,
    *,
    status: str,
    actor: RuntimeIdentity,
    note: str = "",
    evidence_logger: EvidenceLogger | None = None,
) -> Path:
    """Append an idempotent accepted/rejected/revoked receipt after W&B evidence."""

    if status not in RECEIPT_STATUSES:
        raise BundleValidationError(f"invalid receipt status: {status!r}")
    if status == "accepted":
        raise BundleValidationError(
            "accepted receipts are created only by accept_bundle; use viola-handoff accept"
        )
    verified = _inspect_bundle(
        bundle,
        verify_artifacts=True,
        ready_required=True,
        enforce_directory_name=True,
    )
    actor_value = _identity_dict(actor)
    receipt = _build_receipt(verified, status=status, actor=actor, note=note)
    destination = verified.path / "receipts" / f"{receipt['receipt_key']}.json"
    if destination.exists():
        return _append_receipt(verified.path, receipt, manifest=verified.manifest)
    logger = required_evidence_logger(evidence_logger)
    _required_evidence_url(
        logger.record(
            project=verified.manifest["wandb"]["project"],
            run_id=verified.manifest["wandb"]["run_id"],
            event=status,
            metadata=_event_metadata(verified, actor=actor_value),
        )
    )
    result = _append_receipt(verified.path, receipt, manifest=verified.manifest)
    _inspect_bundle(
        verified.path,
        verify_artifacts=True,
        ready_required=True,
        enforce_directory_name=True,
    )
    return result


def copy_paste_message(
    bundle: VerifiedBundle, *, destination_root: str | Path = DEFAULT_ACCEPT_ROOT
) -> str:
    """Render the intentionally small human handoff block."""

    destination = os.path.expanduser(os.fspath(destination_root))
    receiver = bundle.manifest["consumer"]["role"]
    artifact_hashes = (
        ",".join(
            f"{item['name']}={item['inventory_sha256']}" for item in bundle.manifest["artifacts"]
        )
        or "none"
    )
    return "\n".join(
        [
            f"kind: {bundle.kind}",
            f"bundle_id: {bundle.bundle_id}",
            f"nas_path: {bundle.path}",
            f"manifest_sha256: {bundle.ready['manifest_sha256']}",
            f"inventory_sha256: {bundle.ready['inventory_sha256']} ({artifact_hashes})",
            f"permission: {bundle.permission}",
            f"receiver: {receiver}",
            f"wandb: {bundle.ready['wandb_url'] or bundle.manifest['wandb']['run_id']}",
            "accept: "
            f"viola-handoff accept {shlex.quote(str(bundle.path))} "
            f"--destination-root {shlex.quote(destination)}",
        ]
    )
