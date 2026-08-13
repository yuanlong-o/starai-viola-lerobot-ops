"""Produce the reviewed, hardware-inert inputs used for Repo-B planning.

This module reads files and seals evidence.  It deliberately imports no
camera, serial, motor, robot, or policy implementation.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import viola_handoff

from .errors import ValidationError
from .jsonutil import (
    copy_regular_file,
    read_json_object,
    require_exact_keys,
    sha256_file,
    sha256_json,
    write_canonical_json,
)
from .schemas import EXECUTOR_CAPABILITIES, ReviewedSetup, VIOLA_CAMERAS, VIOLA_JOINTS

ESTOP_MAX_AGE = timedelta(hours=24)
DEFAULT_WANDB_PROJECT = "starai-viola-policy-benchmark"
DEFAULT_MATERIAL_ROOT = Path("/mnt/nas02/yz/starai/producer-materials/v1")

_SETUP_FIELDS = {
    "schema_version",
    "setup_id",
    "robot_port",
    "cameras",
    "calibration_path",
    "calibration_sha256",
    "reset_protocol_path",
    "camera_config_sha256",
    "robot_config_sha256",
    "reset_protocol_sha256",
    "joint_limits",
    "max_step_deltas",
    "speed_scale",
    "executor_attestation",
    "estop",
}
_EXECUTOR_FIELDS = {
    "reviewed",
    "clean_commit",
    "commit",
    "reviewer",
    "reviewed_at",
    "repository",
    "python_version",
    "lerobot_version",
    "conda_environment",
    "execution_backend",
    "entrypoint_path",
    "entrypoint_sha256",
    "capabilities",
}
_ESTOP_FIELDS = {"tested_at", "operator", "passed"}
_CAMERA_FIELDS = {"type", "index_or_path", "width", "height", "fps"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HEX_SHA = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_FULL_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PYTHON_312 = re.compile(r"^3\.12(?:\.\d+)?$")


@dataclass(frozen=True, slots=True)
class SessionInputsRelease:
    """The sealed bundle and deterministic producer material behind it."""

    bundle: viola_handoff.VerifiedBundle
    material_root: Path
    payload_root: Path
    setup_record: Path
    receiver_destination: Path


def seal_session_inputs(
    setup_path: str | Path,
    *,
    experiment: str,
    subject: str,
    handoff_root: str | Path = viola_handoff.DEFAULT_HANDOFF_ROOT,
    material_root: str | Path = DEFAULT_MATERIAL_ROOT,
    destination_root: str | Path = viola_handoff.DEFAULT_ACCEPT_ROOT,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    repo_root: str | Path,
    producer_identity: viola_handoff.RuntimeIdentity | None = None,
    evidence_logger: viola_handoff.EvidenceLogger | None = None,
    now: datetime | None = None,
) -> SessionInputsRelease:
    """Validate reviewed setup evidence and seal a planning-only bundle.

    ``destination_root`` is returned as operator guidance; Repo A never
    self-accepts the bundle on Repo B's behalf.
    """

    current_time = _utc_now(now)
    setup = load_reviewed_setup(setup_path, now=current_time)
    if setup.setup_id != subject:
        raise ValidationError(
            f"subject {subject!r} must exactly match setup_id {setup.setup_id!r}"
        )

    repository = _existing_directory(repo_root, label="Repo-A worktree")
    material_base = _external_output_root(
        material_root, repository=repository, label="producer material root"
    )
    handoff_base = _external_output_root(
        handoff_root, repository=repository, label="handoff root"
    )
    identity_was_captured = producer_identity is None
    identity = _producer_identity(
        producer_identity
        or viola_handoff.RuntimeIdentity.capture(role="pc_a", repo_root=repository)
    )
    if identity.repository_commit != setup.executor["commit"]:
        raise ValidationError(
            "reviewed executor commit differs from the clean Repo-A producer revision"
        )

    source_hashes = _setup_source_hashes(setup)
    material_key = sha256_json(
        {
            "setup_id": setup.setup_id,
            "calibration_sha256": source_hashes["calibration"],
            "reset_protocol_sha256": source_hashes["reset"],
            "entrypoint_sha256": source_hashes["entrypoint"],
            "camera_config_sha256": sha256_json(setup.cameras),
            "robot_config_sha256": sha256_json(_robot_binding(setup)),
            "estop": setup.estop,
        }
    )
    material = _safe_output_directory(
        material_base / setup.setup_id / material_key,
        repository=repository,
        label="producer material directory",
    )
    payload = material / "payload"
    artifact = material / "setup_record"

    calibration = copy_regular_file(setup.calibration_path, artifact / "calibration.json")
    reset = copy_regular_file(setup.reset_protocol_path, artifact / "reset_protocol.json")
    entrypoint = copy_regular_file(
        setup.executor_entrypoint, artifact / "executor_entrypoint.py"
    )

    setup_hashes = {
        "calibration": sha256_file(calibration),
        "camera": sha256_json(setup.cameras),
        "robot": sha256_json(_robot_binding(setup)),
        "reset": sha256_file(reset),
    }
    hardware_setup = _hardware_payload(setup, setup_hashes)
    executor = _executor_payload(setup.executor, entrypoint)
    setup_record_inventory = viola_handoff.inventory_root(artifact)
    session_inputs = {
        "schema_version": 1,
        "setup_id": setup.setup_id,
        "setup_hashes": setup_hashes,
        "calibration_artifact": "setup_record",
        "calibration_relative_path": "calibration.json",
        "reset_relative_path": "reset_protocol.json",
        "setup_record_inventory_sha256": setup_record_inventory["inventory_sha256"],
        "executor": executor,
        "estop_operator": setup.estop["operator"],
        "estop_tested_at": setup.estop["tested_at"],
    }
    write_canonical_json(payload / "hardware_setup.json", hardware_setup)
    write_canonical_json(payload / "session_inputs.json", session_inputs)
    payload_inventory = viola_handoff.inventory_root(payload)

    revalidation_time = current_time if now is not None else _utc_now(None)
    current_setup = load_reviewed_setup(setup.source_path, now=revalidation_time)
    if current_setup != setup or _setup_source_hashes(current_setup) != source_hashes:
        raise ValidationError("reviewed setup changed immediately before bundle sealing")
    if viola_handoff.inventory_root(artifact) != setup_record_inventory:
        raise ValidationError("setup-record inventory changed before bundle sealing")
    if viola_handoff.inventory_root(payload) != payload_inventory:
        raise ValidationError("session-input payload changed before bundle sealing")
    if identity_was_captured:
        _require_same_runtime(identity, repo_root=repository)

    request = viola_handoff.SealRequest(
        root=handoff_base,
        kind="session_inputs",
        experiment=experiment,
        subject=subject,
        producer=identity,
        lineage={"setup_id": setup.setup_id},
        wandb_project=wandb_project,
        payload_dir=payload,
        artifact_roots={"setup_record": artifact},
    )
    # The no-keyword form intentionally remains compatible with Repo B's
    # filesystem-only cross-repository probe, which replaces this public call.
    if evidence_logger is None:
        bundle = viola_handoff.seal_bundle(request)
    else:
        bundle = viola_handoff.seal_bundle(request, evidence_logger=evidence_logger)
    return SessionInputsRelease(
        bundle=bundle,
        material_root=material,
        payload_root=payload,
        setup_record=artifact,
        receiver_destination=Path(destination_root).expanduser(),
    )


def load_reviewed_setup(
    setup_path: str | Path,
    *,
    now: datetime | None = None,
) -> ReviewedSetup:
    """Parse and validate reviewed setup evidence without touching hardware."""

    current_time = _utc_now(now)
    raw = read_json_object(setup_path, label="reviewed hardware setup")
    require_exact_keys(raw, _SETUP_FIELDS, label="hardware_setup")
    if raw["schema_version"] != 1:
        raise ValidationError("hardware_setup.schema_version must be 1")

    setup_id = _nonempty(raw["setup_id"], "setup_id")
    if _SAFE_NAME.fullmatch(setup_id) is None:
        raise ValidationError("setup_id must be a safe 1-64 character name")
    robot_port = _nonempty(raw["robot_port"], "robot_port")
    if not robot_port.startswith("/dev/"):
        raise ValidationError("robot_port must be an explicit absolute /dev path")

    cameras = _cameras(raw["cameras"])
    joint_limits = _joint_limits(raw["joint_limits"])
    step_limits = _step_limits(raw["max_step_deltas"])
    speed_scale = _finite_number(raw["speed_scale"], "speed_scale")
    if speed_scale != 1.0:
        raise ValidationError("speed_scale must be exactly 1.0")

    calibration_path = Path(_nonempty(raw["calibration_path"], "calibration_path"))
    reset_path = Path(_nonempty(raw["reset_protocol_path"], "reset_protocol_path"))
    executor = _executor(raw["executor_attestation"], current_time)
    entrypoint = Path(executor["entrypoint_path"])

    expected_hashes = {
        "calibration_sha256": sha256_file(calibration_path),
        "camera_config_sha256": sha256_json(cameras),
        "robot_config_sha256": sha256_json(
            {
                "robot_port": robot_port,
                "joint_limits": joint_limits,
                "max_step_deltas": step_limits,
                "speed_scale": speed_scale,
            }
        ),
        "reset_protocol_sha256": sha256_file(reset_path),
    }
    for field, actual in expected_hashes.items():
        if raw[field] != actual:
            raise ValidationError(f"{field} does not match the reviewed source bytes")
    if executor["entrypoint_sha256"] != sha256_file(entrypoint):
        raise ValidationError("executor entrypoint_sha256 does not match its reviewed file")

    estop = _estop(raw["estop"], current_time)
    return ReviewedSetup(
        source_path=Path(setup_path).expanduser().absolute(),
        setup_id=setup_id,
        robot_port=robot_port,
        cameras=cameras,
        joint_limits=joint_limits,
        max_step_deltas=step_limits,
        speed_scale=speed_scale,
        calibration_path=calibration_path,
        reset_protocol_path=reset_path,
        executor_entrypoint=entrypoint,
        executor=executor,
        estop=estop,
    )


def _cameras(value: Any) -> dict[str, dict[str, Any]]:
    require_exact_keys(value, set(VIOLA_CAMERAS), label="cameras")
    result: dict[str, dict[str, Any]] = {}
    device_paths: list[str] = []
    for name in VIOLA_CAMERAS:
        camera = dict(require_exact_keys(value[name], _CAMERA_FIELDS, label=f"camera {name}"))
        if camera["type"] != "opencv":
            raise ValidationError(f"camera {name} type must be 'opencv'")
        device = _nonempty(camera["index_or_path"], f"camera {name} index_or_path")
        if not device.startswith("/dev/"):
            raise ValidationError(f"camera {name} must use an explicit absolute /dev path")
        if (camera["width"], camera["height"], camera["fps"]) != (640, 480, 30):
            raise ValidationError(f"camera {name} must be configured for 640x480 at 30 fps")
        device_paths.append(device)
        result[name] = camera
    if len(set(device_paths)) != len(VIOLA_CAMERAS):
        raise ValidationError("front and up cameras must use different device paths")
    return result


def _joint_limits(value: Any) -> dict[str, list[float]]:
    require_exact_keys(value, set(VIOLA_JOINTS), label="joint_limits")
    result: dict[str, list[float]] = {}
    for name in VIOLA_JOINTS:
        limits = value[name]
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValidationError(f"joint_limits.{name} must contain [lower, upper]")
        lower = _finite_number(limits[0], f"joint_limits.{name}.lower")
        upper = _finite_number(limits[1], f"joint_limits.{name}.upper")
        if lower >= upper:
            raise ValidationError(f"joint_limits.{name} lower must be less than upper")
        result[name] = [lower, upper]
    return result


def _step_limits(value: Any) -> dict[str, float]:
    require_exact_keys(value, set(VIOLA_JOINTS), label="max_step_deltas")
    result: dict[str, float] = {}
    for name in VIOLA_JOINTS:
        delta = _finite_number(value[name], f"max_step_deltas.{name}")
        if delta <= 0:
            raise ValidationError(f"max_step_deltas.{name} must be greater than zero")
        result[name] = delta
    return result


def _executor(value: Any, now: datetime) -> dict[str, Any]:
    executor = dict(require_exact_keys(value, _EXECUTOR_FIELDS, label="executor_attestation"))
    if executor["reviewed"] is not True or executor["clean_commit"] is not True:
        raise ValidationError("executor must be reviewed at a clean commit")
    if executor["repository"] != "starai-viola-lerobot-ops":
        raise ValidationError("executor repository must be starai-viola-lerobot-ops")
    if not isinstance(executor["commit"], str) or _GIT_SHA.fullmatch(executor["commit"]) is None:
        raise ValidationError("executor commit must be a full lowercase Git revision")
    if (
        not isinstance(executor["python_version"], str)
        or not executor["python_version"].startswith("3.12.")
        or executor["lerobot_version"] != "0.6.1"
        or executor["conda_environment"] != "lerobot"
    ):
        raise ValidationError("executor must bind Python 3.12, LeRobot 0.6.1, and lerobot Conda")
    if executor["execution_backend"] != "direct_lerobot_fashionstar":
        raise ValidationError("executor must use direct_lerobot_fashionstar")
    if executor["capabilities"] != list(EXECUTOR_CAPABILITIES):
        raise ValidationError("executor capabilities differ from the reviewed v1 contract")
    _nonempty(executor["reviewer"], "executor reviewer")
    _nonempty(executor["entrypoint_path"], "executor entrypoint_path")
    _sha256(executor["entrypoint_sha256"], "executor entrypoint_sha256")
    reviewed_at = _parse_utc(executor["reviewed_at"], "executor reviewed_at")
    if reviewed_at > now:
        raise ValidationError("executor reviewed_at cannot be in the future")
    executor["reviewed_at"] = _format_utc(reviewed_at)
    return executor


def _estop(value: Any, now: datetime) -> dict[str, Any]:
    estop = dict(require_exact_keys(value, _ESTOP_FIELDS, label="estop"))
    if estop["passed"] is not True:
        raise ValidationError("E-stop function test must have passed")
    _nonempty(estop["operator"], "E-stop operator")
    tested_at = _parse_utc(estop["tested_at"], "E-stop tested_at")
    if tested_at > now:
        raise ValidationError("E-stop evidence cannot be future-dated")
    if now - tested_at >= ESTOP_MAX_AGE:
        raise ValidationError("E-stop evidence must be less than 24 hours old")
    estop["tested_at"] = _format_utc(tested_at)
    return estop


def _hardware_payload(setup: ReviewedSetup, hashes: Mapping[str, str]) -> dict[str, Any]:
    executor = setup.executor
    return {
        "schema_version": 1,
        "setup_id": setup.setup_id,
        **_robot_binding(setup),
        "cameras": setup.cameras,
        "calibration_path": "/producer/calibration.json",
        "calibration_sha256": hashes["calibration"],
        "reset_protocol_path": "/producer/reset_protocol.json",
        "camera_config_sha256": hashes["camera"],
        "robot_config_sha256": hashes["robot"],
        "reset_protocol_sha256": hashes["reset"],
        "executor_attestation": {
            "reviewed": True,
            "clean_commit": True,
            "commit": executor["commit"],
            "reviewer": executor["reviewer"],
            "reviewed_at": executor["reviewed_at"],
            "repository": executor["repository"],
            "python_version": executor["python_version"],
            "lerobot_version": executor["lerobot_version"],
            "conda_environment": executor["conda_environment"],
            "execution_backend": executor["execution_backend"],
            "entrypoint_path": "/producer/executor_entrypoint.py",
            "entrypoint_sha256": executor["entrypoint_sha256"],
            "capabilities": list(EXECUTOR_CAPABILITIES),
        },
        "estop": setup.estop,
    }


def _executor_payload(executor: Mapping[str, Any], entrypoint: Path) -> dict[str, Any]:
    return {
        "repository": executor["repository"],
        "repository_commit": executor["commit"],
        "repository_clean": True,
        "python_version": executor["python_version"],
        "lerobot_version": executor["lerobot_version"],
        "conda_environment": executor["conda_environment"],
        "execution_backend": executor["execution_backend"],
        "entrypoint_artifact": "setup_record",
        "entrypoint_relative_path": "executor_entrypoint.py",
        "entrypoint_sha256": sha256_file(entrypoint),
        "reviewer": executor["reviewer"],
        "reviewed_at": executor["reviewed_at"],
        "capabilities": list(EXECUTOR_CAPABILITIES),
    }


def _robot_binding(setup: ReviewedSetup) -> dict[str, Any]:
    return {
        "robot_port": setup.robot_port,
        "joint_limits": setup.joint_limits,
        "max_step_deltas": setup.max_step_deltas,
        "speed_scale": setup.speed_scale,
    }


def _setup_source_hashes(setup: ReviewedSetup) -> dict[str, str]:
    return {
        "reviewed_setup": sha256_file(setup.source_path),
        "calibration": sha256_file(setup.calibration_path),
        "reset": sha256_file(setup.reset_protocol_path),
        "entrypoint": sha256_file(setup.executor_entrypoint),
    }


def _existing_directory(path: str | Path, *, label: str) -> Path:
    candidate = _path_without_symlinks(path, label=label, must_exist=True)
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ValidationError(f"{label} is not a directory: {candidate}")
    return candidate


def _external_output_root(
    path: str | Path,
    *,
    repository: Path,
    label: str,
) -> Path:
    candidate = _path_without_symlinks(path, label=label, must_exist=False)
    if candidate == repository or repository in candidate.parents:
        raise ValidationError(f"{label} must be outside the Repo-A worktree")
    return candidate


def _safe_output_directory(
    path: str | Path,
    *,
    repository: Path,
    label: str,
) -> Path:
    candidate = _external_output_root(path, repository=repository, label=label)
    candidate.mkdir(parents=True, exist_ok=True)
    return _existing_directory(candidate, label=label)


def _path_without_symlinks(
    path: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if must_exist:
                raise ValidationError(f"{label} does not exist: {candidate}") from None
            break
        except OSError as exc:
            raise ValidationError(f"cannot inspect {label} {candidate}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink path is forbidden for {label}: {current}")
        if not stat.S_ISDIR(mode):
            raise ValidationError(f"{label} path component is not a directory: {current}")
    return candidate


def _producer_identity(
    identity: viola_handoff.RuntimeIdentity,
) -> viola_handoff.RuntimeIdentity:
    """Validate injected identities before any producer evidence is written."""

    if not isinstance(identity, viola_handoff.RuntimeIdentity):
        raise ValidationError("producer identity must be a RuntimeIdentity")
    if identity.role != "pc_a":
        raise ValidationError("session inputs must be produced with a pc_a identity")
    if identity.repository_clean is not True:
        raise ValidationError("session-input production requires a clean Repo-A identity")
    if _FULL_GIT_SHA.fullmatch(identity.repository_commit) is None:
        raise ValidationError("producer repository commit must be a full lowercase Git SHA")
    if not identity.hostname:
        raise ValidationError("producer hostname must be nonempty")
    if _PYTHON_312.fullmatch(identity.python_version) is None:
        raise ValidationError("session-input production requires Python 3.12")
    if identity.lerobot_version != "0.6.1":
        raise ValidationError("session-input production requires LeRobot 0.6.1")
    if identity.conda_environment != "lerobot":
        raise ValidationError("session-input production requires the lerobot Conda environment")
    return identity


def _require_same_runtime(
    original: viola_handoff.RuntimeIdentity,
    *,
    repo_root: Path,
) -> None:
    current = _producer_identity(
        viola_handoff.RuntimeIdentity.capture(role="pc_a", repo_root=repo_root)
    )
    if current != original:
        raise ValidationError(
            "Repo-A commit or runtime identity changed before session-input sealing"
        )


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() != timedelta(0):
        raise ValidationError("current time must be timezone-aware UTC")
    return current.astimezone(UTC)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValidationError(f"{label} must represent UTC")
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a nonempty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "DEFAULT_MATERIAL_ROOT",
    "DEFAULT_WANDB_PROJECT",
    "ESTOP_MAX_AGE",
    "SessionInputsRelease",
    "load_reviewed_setup",
    "seal_session_inputs",
]
