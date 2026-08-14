"""Fail-closed authorization checks for every physical policy action.

The functions in this module deliberately read like an operator checklist.  A
bundle is not a motion permit merely because transport-level checksums pass.
The reviewed executor, current checkout, E-stop evidence, session protocol,
and explicit operator action must all agree immediately before hardware exists.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, TextIO

from viola_handoff import (
    RuntimeIdentity,
    VerifiedBundle,
    canonical_json_bytes,
    inspect_bundle,
    inventory_root,
    require_active_canonical_source,
    resolve_bundle,
)

from .errors import SafetyGateError
from .schemas import EXECUTOR_CAPABILITIES

POLICIES: Final = (
    "act",
    "diffusion",
    "vqbet",
    "smolvla",
    "pi0",
    "pi0_fast",
    "pi05",
    "groot",
)
PHASES: Final = ("hold", "shakedown", "scored")
JOINTS: Final = (*tuple(f"Motor_{index}" for index in range(6)), "gripper")
CANONICAL_TASK: Final = (
    "Move the blue cube, then the red cube, from the white pad on the right "
    "to the gray platform on the left."
)
ESTOP_MAX_AGE: Final = timedelta(hours=24)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PERMIT_AUTHORITY = object()


@dataclass(slots=True)
class _PermitUse:
    """Mutable lifecycle hidden inside an otherwise immutable permit."""

    state: str = "issued"
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class MotionPermit:
    """One-use, in-process authorization for exactly one session phase/trial."""

    session_id: str
    session_bundle_id: str
    candidate_bundle_id: str
    policy: str
    phase: str
    trial: str
    operator: str
    setup_hashes: Mapping[str, str]
    absolute_limits: Mapping[str, tuple[float, float]]
    max_step_deltas: Mapping[str, float]
    speed_scale: float
    issued_at: datetime
    estop_tested_at: datetime
    _nonce: str
    _authority: object
    setup_id: str = ""
    robot_port: str = ""
    calibration_path: Path = Path()
    reset_protocol_path: Path = Path()
    executor_entrypoint: Path = Path()
    camera_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    local_act_duration_s: float = 0.0
    _use: _PermitUse = field(default_factory=_PermitUse, repr=False, compare=False)

    def allows(self, *, session_id: str, phase: str, trial: str) -> bool:
        """Return whether this permit names the exact requested operation."""

        return (
            self.session_id == session_id
            and self.phase == phase
            and self.trial == trial
            and bool(self._nonce)
            and self._authority is _PERMIT_AUTHORITY
        )


@dataclass(frozen=True, slots=True)
class GateRequest:
    session_bundle: Path
    candidate_bundle: Path
    phase: str
    trial: str
    repository_root: Path
    handoff_root: Path
    prior_hold_bundle: Path | None = None
    prior_shakedown_bundle: Path | None = None
    now: datetime | None = None


@dataclass(frozen=True, slots=True)
class LocalActCheckpointBinding:
    """Exact local ACT bytes approved for one Repo-A-only inference path.

    This is deliberately not a ``policy_candidate`` handoff.  Its identity is
    derived from the complete checkpoint inventory, frozen dataset identity,
    and explicit ten-action deployment overlay, so it cannot be confused with
    Repo-B acceptance.
    """

    root: Path
    inventory: Mapping[str, Any]
    model_sha256: str
    dataset_release_id: str
    dataset_inventory_sha256: str
    dataset_metadata_inventory_sha256: str
    deployment_action_steps: int = 10


@dataclass(frozen=True, slots=True)
class LocalActGateRequest:
    """Reviewed local inputs needed to issue an ACT commissioning permit."""

    setup: ResolvedSetup
    checkpoint: LocalActCheckpointBinding
    operator: str
    estop_tested_at: datetime
    estop_attestation_sha256: str
    estop_passed: bool
    trial: str
    duration_s: float
    repository_root: Path
    now: datetime | None = None


@dataclass(frozen=True, slots=True)
class ResolvedSetup:
    """Reviewed setup bytes recovered from the signed session-input lineage."""

    setup_id: str
    robot_port: str
    camera_configs: Mapping[str, Mapping[str, Any]]
    calibration_path: Path
    reset_protocol_path: Path
    executor_entrypoint: Path
    setup_hashes: Mapping[str, str]
    absolute_limits: Mapping[str, tuple[float, float]]
    max_step_deltas: Mapping[str, float]
    executor_identity: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _ValidatedGate:
    """Immutable result of the non-interactive motion checks."""

    phase: str
    checked_at: datetime
    session: VerifiedBundle
    candidate: VerifiedBundle
    session_payload: Mapping[str, Any]
    setup: ResolvedSetup
    operator: str
    policy: str
    speed_scale: float


@dataclass(frozen=True, slots=True)
class _ValidatedLocalActGate:
    """Immutable projection of one fully rechecked local ACT authority."""

    checked_at: datetime
    session_id: str
    session_binding_id: str
    candidate_id: str
    setup: ResolvedSetup
    operator: str
    estop_tested_at: datetime
    duration_s: float


def authorize_motion(
    request: GateRequest,
    *,
    input_stream: TextIO | None = None,
    terminal_check: Callable[[TextIO], bool] | None = None,
) -> MotionPermit:
    """Validate every gate and ask for an exact interactive operator action.

    This is the only production constructor for :class:`MotionPermit`.  It does
    not import a robot, motor bus, camera, policy, or serial implementation.
    """

    checked = _validate_gate(request)
    phase = checked.phase
    current_time = checked.checked_at
    session = checked.session
    candidate = checked.candidate
    session_payload = checked.session_payload
    setup = checked.setup

    operator = checked.operator
    owns_stream = input_stream is None
    stream = input_stream if input_stream is not None else _open_operator_terminal()
    checker = terminal_check or _is_terminal
    try:
        if not checker(stream):
            raise SafetyGateError("operator confirmation requires an interactive terminal")
        challenge = operator_challenge(session_payload["session_id"], phase, request.trial)
        stream.write(f"Type exactly: {challenge}\n> ")
        stream.flush()
        response = stream.readline()
    except (OSError, ValueError) as exc:
        raise SafetyGateError(f"could not read operator confirmation: {exc}") from exc
    finally:
        if owns_stream:
            stream.close()
    if response.rstrip("\r\n") != challenge:
        raise SafetyGateError("operator confirmation did not match the session challenge")

    nonce_material = (
        f"{session.bundle_id}:{phase}:{request.trial}:{operator}:"
        f"{current_time.isoformat()}:{os.getpid()}:{socket.gethostname()}"
    )
    return MotionPermit(
        session_id=session_payload["session_id"],
        session_bundle_id=session.bundle_id,
        candidate_bundle_id=candidate.bundle_id,
        policy=checked.policy,
        phase=phase,
        trial=request.trial,
        operator=operator,
        setup_hashes=dict(setup.setup_hashes),
        absolute_limits=dict(setup.absolute_limits),
        max_step_deltas=dict(setup.max_step_deltas),
        speed_scale=checked.speed_scale,
        issued_at=current_time,
        estop_tested_at=_timestamp(
            session_payload["estop"]["tested_at_utc"], "E-stop tested_at_utc"
        ),
        _nonce=hashlib.sha256(nonce_material.encode()).hexdigest(),
        _authority=_PERMIT_AUTHORITY,
        setup_id=setup.setup_id,
        robot_port=setup.robot_port,
        calibration_path=setup.calibration_path,
        reset_protocol_path=setup.reset_protocol_path,
        executor_entrypoint=setup.executor_entrypoint,
        camera_configs=dict(setup.camera_configs),
    )


def authorize_local_act_motion(
    request: LocalActGateRequest,
    *,
    identity: RuntimeIdentity,
    input_stream: TextIO | None = None,
    terminal_check: Callable[[TextIO], bool] | None = None,
) -> MotionPermit:
    """Issue one local ACT commissioning permit after an explicit E-stop phrase.

    Repo A owns this narrow authorization.  It does not create or impersonate
    a Repo-B candidate, rollout session, receipt, or READY handoff.  The
    resulting permit enters the same one-use execution state machine as the
    two-PC path.
    """

    checked = _validate_local_act_gate(request, identity=identity)
    owns_stream = input_stream is None
    stream = input_stream if input_stream is not None else _open_operator_terminal()
    checker = terminal_check or _is_terminal
    try:
        if not checker(stream):
            raise SafetyGateError("operator confirmation requires an interactive terminal")
        challenge = local_act_operator_challenge(
            checked.session_id,
            request.trial,
            checked.duration_s,
        )
        stream.write(f"Type exactly: {challenge}\n> ")
        stream.flush()
        response = stream.readline()
    except (OSError, ValueError) as exc:
        raise SafetyGateError(f"could not read operator confirmation: {exc}") from exc
    finally:
        if owns_stream:
            stream.close()
    if response.rstrip("\r\n") != challenge:
        raise SafetyGateError("operator confirmation did not match the local ACT challenge")

    setup = checked.setup
    nonce_material = (
        f"{checked.session_binding_id}:{checked.candidate_id}:local_act:"
        f"{request.trial}:{checked.operator}:{checked.checked_at.isoformat()}:"
        f"{os.getpid()}:{socket.gethostname()}"
    )
    return MotionPermit(
        session_id=checked.session_id,
        session_bundle_id=checked.session_binding_id,
        candidate_bundle_id=checked.candidate_id,
        policy="act",
        phase="local_act",
        trial=request.trial,
        operator=checked.operator,
        setup_hashes=dict(setup.setup_hashes),
        absolute_limits=dict(setup.absolute_limits),
        max_step_deltas=dict(setup.max_step_deltas),
        speed_scale=0.25,
        issued_at=checked.checked_at,
        estop_tested_at=checked.estop_tested_at,
        _nonce=hashlib.sha256(nonce_material.encode()).hexdigest(),
        _authority=_PERMIT_AUTHORITY,
        setup_id=setup.setup_id,
        robot_port=setup.robot_port,
        calibration_path=setup.calibration_path,
        reset_protocol_path=setup.reset_protocol_path,
        executor_entrypoint=setup.executor_entrypoint,
        camera_configs={name: dict(value) for name, value in setup.camera_configs.items()},
        local_act_duration_s=checked.duration_s,
    )


def revalidate_local_act_motion(
    request: LocalActGateRequest,
    permit: MotionPermit,
    *,
    identity: RuntimeIdentity,
    allow_consumed: bool = False,
) -> tuple[str, str]:
    """Recheck local ACT authority without asking the operator to arm again.

    Call this at every boundary where the unified executor currently calls
    :func:`revalidate_motion`.  It re-inventories the checkpoint, reproduces
    the setup hashes, verifies the current clean runtime, and checks E-stop
    freshness before returning the local session and candidate identities.
    """

    if allow_consumed:
        _assert_consumed_local_act_permit(permit)
    else:
        assert_permit_current(permit)
    checked = _validate_local_act_gate(request, identity=identity)
    _check_local_act_permit_binding(permit, request, checked)
    # Hashing a checkpoint can take long enough for a nearly-expired E-stop to
    # cross its boundary.  Keep freshness as the final validation as well.
    if allow_consumed:
        _assert_consumed_local_act_permit(permit)
    else:
        assert_permit_current(permit)
    return checked.session_binding_id, checked.candidate_id


def _assert_consumed_local_act_permit(permit: MotionPermit) -> None:
    """Require a genuine local permit whose one execution has torn down."""

    if not isinstance(permit, MotionPermit) or permit._authority is not _PERMIT_AUTHORITY:
        raise SafetyGateError("post-execution evidence requires an issued motion permit")
    if permit.phase != "local_act":
        raise SafetyGateError("post-execution evidence requires a local ACT permit")
    with permit._use.lock:
        if permit._use.state != "consumed":
            raise SafetyGateError("local ACT execution has not completed permit teardown")


def revalidate_motion(
    request: GateRequest,
    permit: MotionPermit,
    *,
    identity: RuntimeIdentity,
    allow_consumed: bool = False,
) -> tuple[VerifiedBundle, VerifiedBundle]:
    """Recheck live authority without asking the operator to arm a second time.

    Callers use this immediately before importing hardware-capable modules.  It
    reopens every accepted and canonical bundle, rechecks predecessors and
    E-stop age, and proves that the current clean runtime is the exact executor
    signed into the reviewed setup.
    """

    if allow_consumed:
        _assert_consumed_shared_permit(permit)
    else:
        assert_permit_current(permit)
    checked = _validate_gate(request)
    _check_permit_binding(permit, request, checked)
    _check_runtime_identity(identity, checked.setup)
    # The validation above can take time while hashing setup artifacts and
    # reopening canonical sources.  Make E-stop/permit freshness the last gate.
    if allow_consumed:
        _assert_consumed_shared_permit(permit)
    else:
        assert_permit_current(permit)
    return checked.session, checked.candidate


def _assert_consumed_shared_permit(permit: MotionPermit) -> None:
    """Require a genuine shared permit whose one execution has torn down."""

    if not isinstance(permit, MotionPermit) or permit._authority is not _PERMIT_AUTHORITY:
        raise SafetyGateError("post-execution evidence requires an issued motion permit")
    if permit.phase not in PHASES:
        raise SafetyGateError("post-execution evidence requires a shared execution permit")
    with permit._use.lock:
        if permit._use.state != "consumed":
            raise SafetyGateError("shared execution has not completed permit teardown")


def _validate_gate(request: GateRequest) -> _ValidatedGate:
    """Run every non-interactive authorization check against current bytes."""

    phase = _require_phase(request.phase)
    current_time = _utc_now(request.now)
    session = _accepted_bundle(
        request.session_bundle,
        "rollout_session",
        "live_session",
        receiver_role="pc_a",
    )
    candidate = _accepted_bundle(
        request.candidate_bundle,
        "policy_candidate",
        "disconnected_only",
        receiver_role="pc_a",
    )
    if {entry["path"] for entry in session.manifest["payload"]["files"]} not in (
        {"rollout_session.json"},
        {"rollout_session.json", "act_infrastructure_outcome.json"},
        {
            "rollout_session.json",
            "act_infrastructure_outcome.json",
            "act_infrastructure_attestation.json",
        },
    ):
        raise SafetyGateError("live rollout session contains unexpected payload files")
    if session.manifest["artifacts"]:
        raise SafetyGateError("live rollout session cannot contain external artifacts")
    session_payload = _canonical_object(session.payload_file("rollout_session.json"))
    candidate_payload = _canonical_object(candidate.payload_file("policy_candidate.json"))

    check_session_shape(session_payload)
    check_session_manifest_lineage(session_payload, session)
    _check_act_clearance_attachment(session_payload, session)
    check_candidate_binding(session_payload, session, candidate_payload, candidate)
    setup = resolve_reviewed_setup(
        session_payload,
        handoff_root=request.handoff_root,
        repository_root=request.repository_root,
    )
    check_phase_predecessors(
        phase,
        request.prior_hold_bundle,
        request.prior_shakedown_bundle,
        session=session,
        candidate=candidate,
        handoff_root=request.handoff_root,
    )

    # Revocations appended to canonical NAS bundles after local acceptance must
    # take effect immediately.  Artifact bytes still come only from local copies.
    require_active_canonical_source(
        session,
        handoff_root=request.handoff_root,
        expected_kind="rollout_session",
        required_permission="live_session",
    )
    require_active_canonical_source(
        candidate,
        handoff_root=request.handoff_root,
        expected_kind="policy_candidate",
        required_permission="disconnected_only",
    )
    # These time- and checkout-sensitive checks intentionally come last.
    check_estop_freshness(session_payload, current_time)
    check_current_checkout(session_payload, request.repository_root)
    operator = _nonempty(session_payload["operator"], "session operator")
    policy = _nonempty(session.manifest["lineage"]["policy"], "session policy")
    if policy not in POLICIES:
        raise SafetyGateError(f"unsupported rollout policy: {policy!r}")
    speed_scale = 0.25 if phase == "shakedown" else 1.0
    return _ValidatedGate(
        phase=phase,
        checked_at=current_time,
        session=session,
        candidate=candidate,
        session_payload=session_payload,
        setup=setup,
        operator=operator,
        policy=policy,
        speed_scale=speed_scale,
    )


def _validate_local_act_gate(
    request: LocalActGateRequest,
    *,
    identity: RuntimeIdentity,
) -> _ValidatedLocalActGate:
    """Reproduce every local binding before issuing or reusing a permit."""

    if not isinstance(request, LocalActGateRequest):
        raise SafetyGateError("local ACT authorization requires a local gate request")
    current_time = _utc_now(request.now)
    operator = _nonempty(request.operator, "local ACT operator")
    if request.estop_passed is not True:
        raise SafetyGateError("local ACT requires a passed physical E-stop test")
    estop_tested_at = _utc_now(request.estop_tested_at)
    if estop_tested_at > current_time:
        raise SafetyGateError("local ACT E-stop evidence is future-dated")
    if current_time - estop_tested_at >= ESTOP_MAX_AGE:
        raise SafetyGateError("local ACT E-stop evidence is 24 hours old or older")
    estop_digest = _digest(
        request.estop_attestation_sha256,
        "local ACT E-stop attestation",
    )
    _safe_local_component(request.trial, "local ACT trial")
    duration_s = _local_act_duration(request.duration_s)

    setup = _validate_local_act_setup(
        request.setup,
        repository_root=request.repository_root,
    )
    _check_runtime_identity(identity, setup)
    check_current_checkout(
        {"executor": {"repository_commit": identity.repository_commit}},
        request.repository_root,
    )
    candidate_id, checkpoint_inventory = _validate_local_act_checkpoint(
        request.checkpoint
    )
    entrypoint_sha256 = _sha256_file(setup.executor_entrypoint)
    session_material = {
        "schema_version": 1,
        "kind": "repo_a_local_act_authorization",
        "candidate_id": candidate_id,
        "checkpoint_inventory_sha256": checkpoint_inventory["inventory_sha256"],
        "deployment_action_steps": request.checkpoint.deployment_action_steps,
        "setup_id": setup.setup_id,
        "setup_hashes": dict(setup.setup_hashes),
        "executor_entrypoint_sha256": entrypoint_sha256,
        "repository_commit": identity.repository_commit,
        "operator": operator,
        "estop_tested_at": estop_tested_at.isoformat(),
        "estop_attestation_sha256": estop_digest,
        "policy": "act",
        "phase": "local_act",
        "duration_s": duration_s,
        "task": CANONICAL_TASK,
    }
    session_binding_id = _sha256_json(session_material)
    return _ValidatedLocalActGate(
        checked_at=current_time,
        session_id=f"local-act-{session_binding_id[:24]}",
        session_binding_id=session_binding_id,
        candidate_id=candidate_id,
        setup=setup,
        operator=operator,
        estop_tested_at=estop_tested_at,
        duration_s=duration_s,
    )


def _validate_local_act_checkpoint(
    binding: LocalActCheckpointBinding,
) -> tuple[str, dict[str, Any]]:
    """Re-inventory and identify a local ACT checkpoint without loading it."""

    if not isinstance(binding, LocalActCheckpointBinding):
        raise SafetyGateError("local ACT requires a validated checkpoint binding")
    if binding.deployment_action_steps != 10:
        raise SafetyGateError("local ACT deployment must use exactly ten queued actions")
    try:
        expected = json.loads(canonical_json_bytes(dict(binding.inventory)))
        actual = inventory_root(binding.root)
    except Exception as exc:
        raise SafetyGateError(f"cannot verify local ACT checkpoint: {exc}") from exc
    if actual != expected:
        raise SafetyGateError("local ACT checkpoint differs from its approved inventory")

    model_sha256 = _digest(binding.model_sha256, "local ACT model")
    dataset_release_id = _safe_local_component(
        binding.dataset_release_id, "local ACT dataset release"
    )
    dataset_inventory_sha256 = _digest(
        binding.dataset_inventory_sha256, "local ACT dataset inventory"
    )
    dataset_metadata_inventory_sha256 = _digest(
        binding.dataset_metadata_inventory_sha256,
        "local ACT dataset metadata inventory",
    )

    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    files = {entry["path"] for entry in actual["files"]}
    missing = sorted(required - files)
    if missing:
        raise SafetyGateError(f"local ACT checkpoint is missing required files: {missing}")
    if _sha256_file(Path(binding.root) / "model.safetensors") != model_sha256:
        raise SafetyGateError("local ACT model differs from its approved digest")
    config = _read_local_json_object(Path(binding.root) / "config.json", "ACT config")
    if config.get("type") != "act":
        raise SafetyGateError("local checkpoint policy type is not ACT")
    source_steps = config.get("n_action_steps")
    chunk_size = config.get("chunk_size")
    if (
        isinstance(source_steps, bool)
        or not isinstance(source_steps, int)
        or source_steps <= 0
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < binding.deployment_action_steps
    ):
        raise SafetyGateError("local ACT checkpoint has an invalid action queue or chunk size")

    candidate_material = {
        "policy": "act",
        "model_sha256": model_sha256,
        "checkpoint_inventory_sha256": expected["inventory_sha256"],
        "dataset_release_id": dataset_release_id,
        "dataset_inventory_sha256": dataset_inventory_sha256,
        "dataset_metadata_inventory_sha256": dataset_metadata_inventory_sha256,
        "deployment_queue_actions": binding.deployment_action_steps,
    }
    return _sha256_json(candidate_material), actual


def _validate_local_act_setup(
    setup: ResolvedSetup,
    *,
    repository_root: Path,
) -> ResolvedSetup:
    """Reproduce a validated setup directly from its reviewed local bytes."""

    if not isinstance(setup, ResolvedSetup):
        raise SafetyGateError("local ACT requires a validated reviewed setup")
    _safe_local_component(setup.setup_id, "local ACT setup_id")
    if not isinstance(setup.robot_port, str) or not setup.robot_port.startswith("/dev/"):
        raise SafetyGateError("local ACT setup must name an explicit /dev robot port")
    if set(setup.camera_configs) != {"front", "up"}:
        raise SafetyGateError("local ACT setup must name front and up cameras")
    cameras: dict[str, dict[str, Any]] = {}
    camera_paths: list[str] = []
    for name in ("front", "up"):
        camera = setup.camera_configs[name]
        local_camera_fields = {
            "type",
            "index_or_path",
            "width",
            "height",
            "fps",
            "fourcc",
            "warmup_s",
        }
        camera_fields = set(camera)
        if camera_fields != local_camera_fields:
            raise SafetyGateError(f"local ACT {name} camera fields differ")
        device = camera.get("index_or_path")
        if (
            camera.get("type") != "opencv"
            or (camera.get("width"), camera.get("height"), camera.get("fps"))
            != (640, 480, 30)
            or not isinstance(device, str)
            or not device.startswith("/dev/")
        ):
            raise SafetyGateError(
                f"local ACT {name} camera must be OpenCV 640x480 at 30 fps"
            )
        expected_fourcc = "MJPG" if name == "front" else "YUYV"
        if camera.get("fourcc") != expected_fourcc or camera.get("warmup_s") != 8:
            raise SafetyGateError(
                f"local ACT {name} camera format/warmup differs from the reviewed rig"
            )
        cameras[name] = dict(camera)
        camera_paths.append(device)
    if len(set(camera_paths)) != 2:
        raise SafetyGateError("local ACT front and up cameras must be distinct")

    if set(setup.absolute_limits) != set(JOINTS) or set(setup.max_step_deltas) != set(
        JOINTS
    ):
        raise SafetyGateError("local ACT setup must bind all seven joint safety limits")
    limits: dict[str, list[float]] = {}
    deltas: dict[str, float] = {}
    for joint in JOINTS:
        pair = setup.absolute_limits[joint]
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise SafetyGateError(f"local ACT {joint} limit must have lower/upper values")
        lower = _finite(pair[0], f"local ACT {joint} lower limit")
        upper = _finite(pair[1], f"local ACT {joint} upper limit")
        if lower >= upper:
            raise SafetyGateError(f"local ACT {joint} limits are not increasing")
        delta = _finite(
            setup.max_step_deltas[joint], f"local ACT {joint} maximum step"
        )
        if delta <= 0:
            raise SafetyGateError(f"local ACT {joint} maximum step must be positive")
        limits[joint] = [lower, upper]
        deltas[joint] = delta

    expected_hashes = {
        "calibration": _sha256_file(setup.calibration_path),
        "camera": _sha256_json(cameras),
        "robot": _sha256_json(
            {
                "robot_port": setup.robot_port,
                "joint_limits": limits,
                "max_step_deltas": deltas,
                "speed_scale": 1.0,
            }
        ),
        "reset": _sha256_file(setup.reset_protocol_path),
    }
    if dict(setup.setup_hashes) != expected_hashes:
        raise SafetyGateError("local ACT setup hashes do not reproduce from reviewed bytes")

    reviewed_entrypoint = _sha256_file(setup.executor_entrypoint)
    current_entrypoint = (
        Path(repository_root).resolve() / "src/viola_ops/execution.py"
    )
    if _sha256_file(current_entrypoint) != reviewed_entrypoint:
        raise SafetyGateError("local ACT executor differs from the reviewed setup snapshot")
    return setup


def _check_local_act_permit_binding(
    permit: MotionPermit,
    request: LocalActGateRequest,
    checked: _ValidatedLocalActGate,
) -> None:
    """Require a local permit to match the freshly reproduced authority."""

    setup = checked.setup
    if not permit.allows(
        session_id=checked.session_id,
        phase="local_act",
        trial=request.trial,
    ):
        raise SafetyGateError("local ACT permit no longer matches this execution")
    if (
        permit.session_bundle_id != checked.session_binding_id
        or permit.candidate_bundle_id != checked.candidate_id
        or permit.policy != "act"
        or permit.operator != checked.operator
        or permit.estop_tested_at != checked.estop_tested_at
        or permit.speed_scale != 0.25
        or permit.local_act_duration_s != checked.duration_s
        or permit.setup_id != setup.setup_id
        or permit.robot_port != setup.robot_port
        or permit.calibration_path != setup.calibration_path
        or permit.reset_protocol_path != setup.reset_protocol_path
        or permit.executor_entrypoint != setup.executor_entrypoint
        or dict(permit.setup_hashes) != dict(setup.setup_hashes)
        or dict(permit.absolute_limits) != dict(setup.absolute_limits)
        or dict(permit.max_step_deltas) != dict(setup.max_step_deltas)
        or dict(permit.camera_configs)
        != {name: dict(value) for name, value in setup.camera_configs.items()}
    ):
        raise SafetyGateError("local ACT permit differs from revalidated reviewed inputs")


def _check_permit_binding(
    permit: MotionPermit,
    request: GateRequest,
    checked: _ValidatedGate,
) -> None:
    """Require the original operator permit to name the revalidated authority."""

    setup = checked.setup
    session = checked.session
    candidate = checked.candidate
    expected_estop = _timestamp(
        checked.session_payload["estop"]["tested_at_utc"], "E-stop tested_at_utc"
    )
    if not permit.allows(
        session_id=checked.session_payload["session_id"],
        phase=checked.phase,
        trial=request.trial,
    ):
        raise SafetyGateError("motion permit no longer matches the requested execution")
    if (
        permit.session_bundle_id != session.bundle_id
        or permit.candidate_bundle_id != candidate.bundle_id
        or permit.policy != checked.policy
        or permit.operator != checked.operator
        or permit.speed_scale != checked.speed_scale
        or permit.estop_tested_at != expected_estop
        or permit.setup_id != setup.setup_id
        or permit.robot_port != setup.robot_port
        or permit.calibration_path != setup.calibration_path
        or permit.reset_protocol_path != setup.reset_protocol_path
        or permit.executor_entrypoint != setup.executor_entrypoint
        or dict(permit.setup_hashes) != dict(setup.setup_hashes)
        or dict(permit.absolute_limits) != dict(setup.absolute_limits)
        or dict(permit.max_step_deltas) != dict(setup.max_step_deltas)
        or dict(permit.camera_configs) != dict(setup.camera_configs)
    ):
        raise SafetyGateError("motion permit differs from the revalidated signed inputs")


def _check_runtime_identity(identity: RuntimeIdentity, setup: ResolvedSetup) -> None:
    """Bind the fresh clean runtime to the exact reviewed executor environment."""

    if not isinstance(identity, RuntimeIdentity):
        raise SafetyGateError("final motion validation requires a captured runtime identity")
    actual = {
        "repository_commit": identity.repository_commit,
        "python_version": identity.python_version,
        "lerobot_version": identity.lerobot_version,
        "conda_environment": identity.conda_environment,
    }
    if identity.role != "pc_a" or identity.repository_clean is not True:
        raise SafetyGateError("final motion runtime must be a clean pc_a identity")
    if actual != dict(setup.executor_identity):
        raise SafetyGateError("current runtime differs from the signed reviewed executor")


def operator_challenge(session_id: str, phase: str, trial: str) -> str:
    """Return the phrase an operator must type on a real terminal."""

    for label, value in (("session_id", session_id), ("phase", phase), ("trial", trial)):
        if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
            raise SafetyGateError(
                f"operator challenge {label} must be one path-safe component"
            )
    return f"ARM {session_id} {phase} {trial}"


def local_act_operator_challenge(
    session_id: str,
    trial: str,
    duration_s: float,
) -> str:
    """Return the explicit local phrase that records the E-stop assertion."""

    _safe_local_component(session_id, "local ACT session_id")
    _safe_local_component(trial, "local ACT trial")
    duration = _local_act_duration(duration_s)
    return f"ARM {session_id} local_act {trial} {duration:g}s ESTOP TESTED"


def assert_permit_current(
    permit: MotionPermit,
    *,
    now: datetime | None = None,
    require_active: bool = False,
) -> None:
    """Reject forged permits and permits whose E-stop evidence has expired."""

    if not isinstance(permit, MotionPermit) or permit._authority is not _PERMIT_AUTHORITY:
        raise SafetyGateError("hardware requires a permit issued by the motion gate")
    current = _utc_now(now)
    if permit.estop_tested_at > current:
        raise SafetyGateError("motion permit contains future-dated E-stop evidence")
    if current - permit.estop_tested_at >= ESTOP_MAX_AGE:
        raise SafetyGateError("motion permit E-stop evidence has expired")
    with permit._use.lock:
        if permit._use.state == "consumed":
            raise SafetyGateError("motion permit was already consumed")
        if require_active and permit._use.state != "active":
            raise SafetyGateError("hardware requires the active execution of its motion permit")


def begin_permit_execution(permit: MotionPermit) -> None:
    """Consume the permit's one opportunity to start its bound execution."""

    assert_permit_current(permit)
    with permit._use.lock:
        if permit._use.state != "issued":
            raise SafetyGateError("motion permit can start exactly one execution")
        permit._use.state = "active"


def finish_permit_execution(permit: MotionPermit) -> None:
    """Permanently retire a permit after its execution attempt ends."""

    if not isinstance(permit, MotionPermit) or permit._authority is not _PERMIT_AUTHORITY:
        raise SafetyGateError("cannot finish a permit not issued by the motion gate")
    with permit._use.lock:
        if permit._use.state != "active":
            raise SafetyGateError("motion permit has no active execution to finish")
        permit._use.state = "consumed"


def check_session_shape(session: Mapping[str, Any]) -> None:
    """Apply the live-session invariants generic transport inspection omits."""

    expected = {
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
    _exact_keys(session, expected, "rollout_session")
    if session["schema_version"] != 1:
        raise SafetyGateError("rollout session schema_version must be 1")
    if not isinstance(session["session_id"], str) or not _SAFE_COMPONENT.fullmatch(
        session["session_id"]
    ):
        raise SafetyGateError("rollout session_id must be one path-safe component")
    if session["blockers"] != []:
        raise SafetyGateError("rollout session contains blockers")
    if session["phase_permissions"] != list(PHASES):
        raise SafetyGateError("rollout session phases are not hold/shakedown/scored")
    if session["task"] != CANONICAL_TASK:
        raise SafetyGateError("rollout session task is not the reviewed task")
    if session["policy_bundle_id"] != session["policy_content_id"]:
        raise SafetyGateError("session policy bundle/content IDs differ")
    binding = session["session_inputs_binding"]
    _exact_keys(
        binding,
        {
            "bundle_id",
            "content_id",
            "manifest_sha256",
            "payload_sha256",
            "setup_record_inventory_sha256",
        },
        "session_inputs binding",
    )
    for name, digest in binding.items():
        _digest(digest, f"session_inputs {name}")
    if binding["bundle_id"] != binding["content_id"]:
        raise SafetyGateError("session_inputs bundle/content IDs differ")
    setup_hashes = session["setup_hashes"]
    _exact_keys(setup_hashes, {"calibration", "camera", "robot", "reset"}, "setup hashes")
    for digest in setup_hashes.values():
        _digest(digest, "setup hash")
    protocol = session["trial_protocol"]
    protocol_keys = {
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
    required_protocol = {
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
    _exact_keys(protocol, protocol_keys, "rollout trial protocol")
    for name, expected_value in required_protocol.items():
        if protocol.get(name) != expected_value:
            raise SafetyGateError(f"rollout protocol {name} is not canonical")
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
    if protocol["ordered_conditions"] != expected_conditions:
        raise SafetyGateError("rollout ordered conditions are not the seed-1000 protocol")
    expected_order = [condition["condition_id"] for condition in expected_conditions]
    if (
        protocol["schedule_order"] != expected_order
        or protocol["execution_order"] != expected_order
    ):
        raise SafetyGateError("rollout schedule/execution order is not canonical")
    source = session["source_session"]
    source_fields = {
        "sha256",
        "created_at_utc",
        "benchmark_lineage_sha256",
        "policy_bindings_sha256",
        "policy_binding_sha256",
        "physical_setup_binding_sha256",
        "executor_binding_sha256",
    }
    _exact_keys(source, source_fields, "rollout source session")
    for name in source_fields - {"created_at_utc"}:
        _digest(source[name], f"source_session {name}")
    _timestamp(source["created_at_utc"], "source_session created_at_utc")
    executor = session["executor"]
    _exact_keys(
        executor,
        {
            "repository",
            "repository_commit",
            "entrypoint_sha256",
            "attestation_sha256",
            "reviewed_at_utc",
        },
        "rollout executor",
    )
    if executor["repository"] != "starai-viola-lerobot-ops":
        raise SafetyGateError("rollout executor repository is not canonical Repo A")
    if not isinstance(executor["repository_commit"], str) or not _COMMIT.fullmatch(
        executor["repository_commit"]
    ):
        raise SafetyGateError("rollout executor commit is malformed")
    _digest(executor["entrypoint_sha256"], "rollout executor entrypoint")
    _digest(executor["attestation_sha256"], "rollout executor attestation")
    _timestamp(executor["reviewed_at_utc"], "rollout executor reviewed_at_utc")
    estop = session["estop"]
    _exact_keys(
        estop,
        {"operator", "tested_at_utc", "attestation_sha256"},
        "rollout E-stop",
    )
    _digest(estop["attestation_sha256"], "rollout E-stop attestation")


def check_candidate_binding(
    session: Mapping[str, Any],
    session_bundle: VerifiedBundle,
    candidate: Mapping[str, Any],
    candidate_bundle: VerifiedBundle,
) -> None:
    """Prove that the accepted session names the exact accepted candidate."""

    if (
        session["policy_bundle_id"] != candidate_bundle.bundle_id
        or session["policy_content_id"] != candidate_bundle.content_id
    ):
        raise SafetyGateError("rollout session names a different policy candidate")
    policy = session_bundle.manifest.get("lineage", {}).get("policy")
    if policy not in POLICIES or candidate.get("policy") != policy:
        raise SafetyGateError("session and candidate policy identities differ")
    if candidate.get("queue_actions") != 10 or candidate.get("task") != CANONICAL_TASK:
        raise SafetyGateError("candidate task or ten-action queue is not canonical")


def check_session_manifest_lineage(
    session: Mapping[str, Any], bundle: VerifiedBundle
) -> None:
    """Re-run the typed Repo-B payload/manifest binding before motion.

    Transport inspection protects bytes, but the generic inspector does not
    call the typed rollout-session producer validator.  Keep this projection
    explicit so a copied or malformed live manifest cannot mint authority.
    """

    lineage = bundle.manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise SafetyGateError("rollout session lacks typed manifest lineage")
    expected_keys = {
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
    _exact_keys(lineage, expected_keys, "rollout session lineage")
    source = session["source_session"]
    inputs = session["session_inputs_binding"]
    expected = {
        "session_id": session["session_id"],
        "policy_bundle_id": session["policy_bundle_id"],
        "policy_content_id": session["policy_content_id"],
        "session_manifest_sha256": source["sha256"],
        "source_session_sha256": source["sha256"],
        "session_inputs_bundle_id": inputs["bundle_id"],
        "session_inputs_content_id": inputs["content_id"],
        "session_inputs_manifest_sha256": inputs["manifest_sha256"],
        "session_inputs_payload_sha256": inputs["payload_sha256"],
        "session_inputs_setup_record_inventory_sha256": inputs[
            "setup_record_inventory_sha256"
        ],
        "benchmark_lineage_sha256": source["benchmark_lineage_sha256"],
        "policy_bindings_sha256": source["policy_bindings_sha256"],
        "policy_binding_sha256": source["policy_binding_sha256"],
        "physical_setup_binding_sha256": source["physical_setup_binding_sha256"],
        "executor_binding_sha256": source["executor_binding_sha256"],
        "executor_attestation_sha256": session["executor"]["attestation_sha256"],
        "estop_attestation_sha256": session["estop"]["attestation_sha256"],
        "blockers": [],
        "act_infrastructure_clearance": session["act_infrastructure_clearance"],
    }
    for name, expected_value in expected.items():
        if lineage.get(name) != expected_value:
            raise SafetyGateError(f"rollout session manifest lineage differs at {name}")
    policy = lineage["policy"]
    if policy not in POLICIES:
        raise SafetyGateError("rollout session lineage names an unsupported policy")
    _validate_act_clearance(session["act_infrastructure_clearance"], policy=policy)


def _validate_act_clearance(value: Any, *, policy: str) -> None:
    """Validate the portable ACT-first infrastructure clearance projection."""

    if policy == "act":
        if value is not None:
            raise SafetyGateError("ACT rollout session cannot authorize itself as clearance")
        return
    if not isinstance(value, Mapping):
        raise SafetyGateError("non-ACT live session lacks ACT infrastructure clearance")
    fields = {
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
    _exact_keys(value, fields, "ACT infrastructure clearance")
    _digest(value["act_outcome_sha256"], "ACT outcome")
    if value["mode"] == "shared_infrastructure_proven":
        if (
            value["act_terminal_status"] != "scored"
            or value["attestation_sha256"] is not None
            or any(
                value[name] is not None
                for name in ("blocker_stage", "blocker_code", "reviewer", "reviewed_at")
            )
        ):
            raise SafetyGateError("ACT shared-infrastructure clearance is malformed")
        _digest(value["act_rollout_evidence_bundle_id"], "ACT rollout evidence bundle")
        return
    if value["mode"] != "policy_specific_act_blocker":
        raise SafetyGateError("ACT infrastructure clearance mode is unsupported")
    allowed_statuses = {
        "ineligible_training",
        "ineligible_offline",
        "ineligible_pc_runtime",
        "unsafe_shadow",
        "unsafe_shakedown",
    }
    if (
        value["act_terminal_status"] not in allowed_statuses
        or value["act_rollout_evidence_bundle_id"] is not None
        or not isinstance(value["blocker_stage"], str)
        or not value["blocker_stage"].strip()
        or not isinstance(value["blocker_code"], str)
        or re.fullmatch(r"[a-z][a-z0-9_]{2,63}", value["blocker_code"]) is None
        or not isinstance(value["reviewer"], str)
        or not value["reviewer"].strip()
    ):
        raise SafetyGateError("ACT policy-specific blocker clearance is malformed")
    _digest(value["attestation_sha256"], "ACT blocker attestation")
    _timestamp(value["reviewed_at"], "ACT blocker reviewed_at")


def _check_act_clearance_attachment(
    session: Mapping[str, Any], bundle: VerifiedBundle
) -> None:
    """Bind the embedded ACT outcome used to authorize non-ACT infrastructure."""

    policy = bundle.manifest["lineage"]["policy"]
    files = {entry["path"] for entry in bundle.manifest["payload"]["files"]}
    if policy == "act":
        if files != {"rollout_session.json"}:
            raise SafetyGateError("ACT session contains an unexpected clearance attachment")
        return
    clearance = session["act_infrastructure_clearance"]
    expected_files = {"rollout_session.json", "act_infrastructure_outcome.json"}
    if clearance["mode"] == "policy_specific_act_blocker":
        expected_files.add("act_infrastructure_attestation.json")
    if files != expected_files:
        raise SafetyGateError("non-ACT live session lacks its embedded ACT clearance evidence")
    path = bundle.payload_file("act_infrastructure_outcome.json")
    if _sha256_file(path) != clearance["act_outcome_sha256"]:
        raise SafetyGateError("embedded ACT outcome hash differs from live-session clearance")
    outcome = _canonical_object(path)
    if outcome.get("policy") != "act" or outcome.get("status") != clearance[
        "act_terminal_status"
    ]:
        raise SafetyGateError("embedded ACT outcome does not prove scored infrastructure")
    if clearance["mode"] == "shared_infrastructure_proven":
        if outcome.get("rollout_evidence_bundle_id") != clearance[
            "act_rollout_evidence_bundle_id"
        ]:
            raise SafetyGateError("embedded ACT outcome names different scored evidence")
        return

    attestation_path = bundle.payload_file("act_infrastructure_attestation.json")
    if _sha256_file(attestation_path) != clearance["attestation_sha256"]:
        raise SafetyGateError("embedded ACT blocker attestation hash differs")
    attestation = _canonical_object(attestation_path)
    attestation_fields = {
        "schema_version",
        "kind",
        "experiment",
        "policy",
        "terminal_status",
        "blocker_stage",
        "blocker_code",
        "classification",
        "shared_infrastructure_implicated",
        "shared_component_exclusions",
        "reviewer",
        "reviewed_at",
        "repo_b_commit",
        "repository_clean",
        "wandb",
    }
    _exact_keys(attestation, attestation_fields, "ACT blocker attestation")
    expected_exclusions = [
        "handoff_contract",
        "dataset_pipeline",
        "session_preregistration",
        "camera_pipeline",
        "robot_driver",
        "shared_control_loop",
        "safety_interlocks",
        "motion_evidence_pipeline",
        "wandb_evidence_pipeline",
    ]
    if (
        attestation["schema_version"] != 1
        or attestation["kind"] != "act_policy_specific_blocker_attestation"
        or attestation["experiment"] != bundle.manifest["experiment"]
        or attestation["policy"] != "act"
        or attestation["terminal_status"] != clearance["act_terminal_status"]
        or attestation["blocker_stage"] != clearance["blocker_stage"]
        or attestation["blocker_code"] != clearance["blocker_code"]
        or attestation["reviewer"] != clearance["reviewer"]
        or attestation["reviewed_at"] != clearance["reviewed_at"]
        or attestation["classification"] != "policy_specific"
        or attestation["shared_infrastructure_implicated"] is not False
        or attestation["shared_component_exclusions"] != expected_exclusions
        or attestation["repository_clean"] is not True
        or attestation["repo_b_commit"]
        != bundle.manifest["producer"]["repository_commit"]
    ):
        raise SafetyGateError("embedded ACT blocker attestation is inconsistent")
    wandb = attestation["wandb"]
    _exact_keys(wandb, {"entity", "project", "run_id", "url"}, "ACT blocker W&B")
    expected_url = (
        f"https://wandb.ai/{wandb['entity']}/{wandb['project']}/runs/{wandb['run_id']}"
    )
    if wandb["url"] != expected_url or wandb["project"] != bundle.manifest["wandb"]["project"]:
        raise SafetyGateError("embedded ACT blocker W&B identity is inconsistent")
    evidence = outcome.get("evidence")
    if not isinstance(evidence, list):
        raise SafetyGateError("embedded ACT outcome lacks blocker evidence")
    matches = [
        item
        for item in evidence
        if isinstance(item, Mapping)
        and item.get("kind") == "act_policy_specific_blocker_attestation"
    ]
    if (
        len(matches) != 1
        or matches[0].get("sha256") != clearance["attestation_sha256"]
        or matches[0].get("wandb_url") != wandb["url"]
    ):
        raise SafetyGateError("embedded ACT outcome does not bind its blocker attestation")


def check_current_checkout(session: Mapping[str, Any], repository_root: Path) -> None:
    """Require the exact reviewed clean executor checkout and runtime."""

    executor = session.get("executor")
    if not isinstance(executor, Mapping):
        raise SafetyGateError("rollout session lacks executor evidence")
    expected = executor.get("repository_commit")
    if not isinstance(expected, str) or not _COMMIT.fullmatch(expected):
        raise SafetyGateError("executor commit is malformed")
    root = repository_root.resolve()
    head = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if head != expected:
        raise SafetyGateError("current Repo-A commit differs from reviewed executor")
    if dirty:
        raise SafetyGateError("motion requires a clean Repo-A worktree")
    if sys.version_info[:2] != (3, 12):
        raise SafetyGateError("motion requires Python 3.12")
    if os.environ.get("CONDA_DEFAULT_ENV") != "lerobot":
        raise SafetyGateError("motion requires the lerobot Conda environment")
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise SafetyGateError("motion requires LeRobot 0.6.1")
    if importlib.metadata.version("wandb") != "0.27.2":
        raise SafetyGateError("motion requires W&B 0.27.2")
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        raise SafetyGateError("motion requires online W&B evidence")
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    if mode != "online":
        raise SafetyGateError("motion requires WANDB_MODE=online")


def check_estop_freshness(session: Mapping[str, Any], now: datetime) -> None:
    """Require passed, same-operator E-stop evidence less than 24 hours old."""

    estop = session.get("estop")
    if not isinstance(estop, Mapping):
        raise SafetyGateError("rollout session lacks E-stop evidence")
    if estop.get("operator") != session.get("operator"):
        raise SafetyGateError("rollout operator differs from tested E-stop owner")
    tested = _timestamp(estop.get("tested_at_utc"), "E-stop tested_at_utc")
    if tested > now:
        raise SafetyGateError("E-stop evidence is future-dated")
    if now - tested >= ESTOP_MAX_AGE:
        raise SafetyGateError("E-stop evidence is 24 hours old or older")
    _digest(estop.get("attestation_sha256"), "E-stop attestation")


def check_phase_predecessors(
    phase: str,
    hold_path: Path | None,
    shakedown_path: Path | None,
    *,
    session: VerifiedBundle,
    candidate: VerifiedBundle,
    handoff_root: Path,
) -> None:
    """Require accepted, current evidence before advancing a physical phase."""

    if phase == "hold":
        if hold_path is not None or shakedown_path is not None:
            raise SafetyGateError("hold cannot name predecessor evidence")
        return
    if hold_path is None:
        raise SafetyGateError(f"{phase} requires accepted hold evidence")
    hold = _accepted_bundle(
        hold_path,
        "rollout_evidence",
        "evidence_only",
        receiver_role="pc_b",
    )
    _check_prior_payload(
        hold,
        "hold",
        session,
        candidate,
        expected_hold=None,
        expected_shakedown=None,
    )
    require_active_canonical_source(
        hold,
        handoff_root=handoff_root,
        expected_kind="rollout_evidence",
        required_permission="evidence_only",
    )
    if phase == "shakedown":
        if shakedown_path is not None:
            raise SafetyGateError("shakedown cannot name shakedown predecessor evidence")
        return
    if shakedown_path is None:
        raise SafetyGateError("scored execution requires accepted shakedown evidence")
    shakedown = _accepted_bundle(
        shakedown_path,
        "rollout_evidence",
        "evidence_only",
        receiver_role="pc_b",
    )
    _check_prior_payload(
        shakedown,
        "shakedown",
        session,
        candidate,
        expected_hold=hold,
        expected_shakedown=None,
    )
    require_active_canonical_source(
        shakedown,
        handoff_root=handoff_root,
        expected_kind="rollout_evidence",
        required_permission="evidence_only",
    )


def validate_action(
    action: Mapping[str, Any],
    reference: Mapping[str, Any],
    permit: MotionPermit,
) -> dict[str, float]:
    """Return an exact safe action or reject it without clamping."""

    expected = {f"{joint}.pos" for joint in JOINTS}
    _exact_keys(action, expected, "policy action")
    _exact_keys(reference, expected, "feedback position")
    validated: dict[str, float] = {}
    for joint in JOINTS:
        key = f"{joint}.pos"
        proposed = _finite(action[key], key)
        current = _finite(reference[key], f"feedback {key}")
        lower, upper = permit.absolute_limits[joint]
        if not lower <= current <= upper:
            raise SafetyGateError(f"feedback {key} is outside reviewed absolute limits")
        if not lower <= proposed <= upper:
            raise SafetyGateError(f"{key} is outside reviewed absolute limits")
        allowed = permit.max_step_deltas[joint] * permit.speed_scale
        if abs(proposed - current) > allowed:
            raise SafetyGateError(f"{key} exceeds the reviewed per-step limit")
        validated[key] = proposed
    return validated


def resolve_reviewed_setup(
    session: Mapping[str, Any],
    *,
    handoff_root: Path,
    repository_root: Path,
) -> ResolvedSetup:
    """Resolve and reproduce the setup bound into a rollout session.

    A rollout session carries only portable hashes.  The underlying setup was
    produced by Repo A and accepted by Repo B, so this function reopens that
    immutable canonical bundle, verifies Repo B's receipt, and derives every
    runtime value from its checked bytes.  Nothing supplied on the execution
    command line can replace a port, camera, calibration, or limit.
    """

    binding = session["session_inputs_binding"]
    source_path = resolve_bundle(handoff_root, "session_inputs", binding["bundle_id"])
    setup_bundle = _accepted_bundle(
        source_path,
        "session_inputs",
        "planning_only",
        receiver_role="pc_b",
    )
    if {
        entry["path"] for entry in setup_bundle.manifest["payload"]["files"]
    } != {"hardware_setup.json", "session_inputs.json"}:
        raise SafetyGateError("session_inputs must contain exactly its two reviewed payloads")
    if {item["name"] for item in setup_bundle.manifest["artifacts"]} != {"setup_record"}:
        raise SafetyGateError("session_inputs must contain exactly the setup_record artifact")

    manifest_path = setup_bundle.path / "manifest.json"
    session_inputs_path = setup_bundle.payload_file("session_inputs.json")
    expected_binding = {
        "bundle_id": setup_bundle.bundle_id,
        "content_id": setup_bundle.content_id,
        "manifest_sha256": _sha256_file(manifest_path),
        "payload_sha256": _sha256_file(session_inputs_path),
        "setup_record_inventory_sha256": _artifact_inventory(
            setup_bundle, "setup_record"
        ),
    }
    if dict(binding) != expected_binding:
        raise SafetyGateError("rollout session binds different session-input bytes")

    inputs = _canonical_object(session_inputs_path)
    hardware = _canonical_object(setup_bundle.payload_file("hardware_setup.json"))
    _validate_session_inputs_payload(inputs)
    _validate_hardware_setup(hardware)
    if inputs["setup_id"] != hardware["setup_id"]:
        raise SafetyGateError("session-input and hardware setup IDs differ")
    if dict(inputs["setup_hashes"]) != dict(session["setup_hashes"]):
        raise SafetyGateError("rollout session setup hashes differ from session inputs")

    # These bytes originated on Repo A and were independently checksum-staged
    # by Repo B before it issued the session.  Re-open A's signed producer root
    # and reproduce every hash; B's receiver-local copy is on the other PC.
    artifact = setup_bundle.artifact_root("setup_record").resolve()
    calibration = _inside_artifact(artifact, "calibration.json")
    reset = _inside_artifact(artifact, "reset_protocol.json")
    entrypoint = _inside_artifact(artifact, "executor_entrypoint.py")
    expected_hashes = {
        "calibration": _sha256_file(calibration),
        "camera": _sha256_json(hardware["cameras"]),
        "robot": _sha256_json(
            {
                "robot_port": hardware["robot_port"],
                "joint_limits": hardware["joint_limits"],
                "max_step_deltas": hardware["max_step_deltas"],
                "speed_scale": hardware["speed_scale"],
            }
        ),
        "reset": _sha256_file(reset),
    }
    if dict(inputs["setup_hashes"]) != expected_hashes:
        raise SafetyGateError("reviewed setup hashes do not reproduce from immutable bytes")
    if (
        hardware["calibration_sha256"] != expected_hashes["calibration"]
        or hardware["camera_config_sha256"] != expected_hashes["camera"]
        or hardware["robot_config_sha256"] != expected_hashes["robot"]
        or hardware["reset_protocol_sha256"] != expected_hashes["reset"]
    ):
        raise SafetyGateError("hardware setup duplicates inconsistent setup hashes")

    executor = inputs["executor"]
    hardware_executor = hardware["executor_attestation"]
    _validate_executor(executor, hardware_executor, entrypoint, session)
    current_entrypoint = (repository_root.resolve() / "src/viola_ops/execution.py").resolve()
    if _sha256_file(current_entrypoint) != executor["entrypoint_sha256"]:
        raise SafetyGateError("current executor entrypoint differs from the reviewed snapshot")

    estop = hardware["estop"]
    if (
        estop["passed"] is not True
        or estop["operator"] != inputs["estop_operator"]
        or estop["tested_at"] != inputs["estop_tested_at"]
        or estop["operator"] != session["estop"]["operator"]
        or estop["tested_at"] != session["estop"]["tested_at_utc"]
    ):
        raise SafetyGateError("reviewed setup and rollout E-stop evidence differ")

    limits, deltas = _reviewed_limits(hardware)
    return ResolvedSetup(
        setup_id=inputs["setup_id"],
        robot_port=hardware["robot_port"],
        camera_configs={name: dict(value) for name, value in hardware["cameras"].items()},
        calibration_path=calibration,
        reset_protocol_path=reset,
        executor_entrypoint=entrypoint,
        setup_hashes=expected_hashes,
        absolute_limits=limits,
        max_step_deltas=deltas,
        executor_identity={
            "repository_commit": executor["repository_commit"],
            "python_version": executor["python_version"],
            "lerobot_version": executor["lerobot_version"],
            "conda_environment": executor["conda_environment"],
        },
    )


def _validate_session_inputs_payload(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "setup_id",
        "setup_hashes",
        "calibration_artifact",
        "calibration_relative_path",
        "reset_relative_path",
        "setup_record_inventory_sha256",
        "executor",
        "estop_operator",
        "estop_tested_at",
    }
    _exact_keys(value, expected, "session_inputs")
    if (
        value["schema_version"] != 1
        or value["calibration_artifact"] != "setup_record"
        or value["calibration_relative_path"] != "calibration.json"
        or value["reset_relative_path"] != "reset_protocol.json"
    ):
        raise SafetyGateError("session_inputs differs from the reviewed v1 layout")
    _nonempty(value["setup_id"], "session_inputs setup_id")
    _nonempty(value["estop_operator"], "session_inputs E-stop operator")
    _timestamp(value["estop_tested_at"], "session_inputs E-stop tested_at")
    _exact_keys(
        value["setup_hashes"],
        {"calibration", "camera", "robot", "reset"},
        "session_inputs setup hashes",
    )
    for digest in value["setup_hashes"].values():
        _digest(digest, "session_inputs setup hash")
    _digest(value["setup_record_inventory_sha256"], "setup_record inventory")


def _validate_hardware_setup(value: Mapping[str, Any]) -> None:
    expected = {
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
    _exact_keys(value, expected, "hardware_setup")
    if value["schema_version"] != 1 or value["speed_scale"] != 1.0:
        raise SafetyGateError("hardware setup must use schema 1 and speed_scale=1.0")
    port = _nonempty(value["robot_port"], "reviewed robot port")
    if not port.startswith("/dev/"):
        raise SafetyGateError("reviewed robot port must be an explicit /dev path")
    _exact_keys(value["cameras"], {"front", "up"}, "reviewed cameras")
    camera_paths: list[str] = []
    for name in ("front", "up"):
        camera = value["cameras"][name]
        _exact_keys(camera, {"type", "index_or_path", "width", "height", "fps"}, name)
        if (
            camera["type"] != "opencv"
            or (camera["width"], camera["height"], camera["fps"]) != (640, 480, 30)
            or not isinstance(camera["index_or_path"], str)
            or not camera["index_or_path"].startswith("/dev/")
        ):
            raise SafetyGateError(f"reviewed {name} camera must be OpenCV 640x480 at 30 fps")
        camera_paths.append(camera["index_or_path"])
    if len(set(camera_paths)) != 2:
        raise SafetyGateError("front and up cameras must use distinct reviewed devices")
    _exact_keys(value["estop"], {"tested_at", "operator", "passed"}, "setup E-stop")


def _validate_executor(
    executor: Mapping[str, Any],
    hardware: Mapping[str, Any],
    entrypoint: Path,
    session: Mapping[str, Any],
) -> None:
    expected = {
        "repository",
        "repository_commit",
        "repository_clean",
        "python_version",
        "lerobot_version",
        "conda_environment",
        "execution_backend",
        "entrypoint_artifact",
        "entrypoint_relative_path",
        "entrypoint_sha256",
        "reviewer",
        "reviewed_at",
        "capabilities",
    }
    _exact_keys(executor, expected, "session_inputs executor")
    if (
        executor["repository"] != "starai-viola-lerobot-ops"
        or executor["repository_clean"] is not True
        or executor["lerobot_version"] != "0.6.1"
        or executor["conda_environment"] != "lerobot"
        or executor["execution_backend"] != "direct_lerobot_fashionstar"
        or executor["entrypoint_artifact"] != "setup_record"
        or executor["entrypoint_relative_path"] != "executor_entrypoint.py"
        or executor["capabilities"] != list(EXECUTOR_CAPABILITIES)
        or not isinstance(executor["python_version"], str)
        or not executor["python_version"].startswith("3.12.")
    ):
        raise SafetyGateError("reviewed executor differs from the Repo-A v1 contract")
    if _sha256_file(entrypoint) != executor["entrypoint_sha256"]:
        raise SafetyGateError("reviewed executor snapshot hash differs")
    _exact_keys(
        hardware,
        {
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
        },
        "hardware executor attestation",
    )
    projection = {
        "reviewed": True,
        "clean_commit": True,
        "commit": executor["repository_commit"],
        "reviewer": executor["reviewer"],
        "reviewed_at": executor["reviewed_at"],
        "repository": executor["repository"],
        "python_version": executor["python_version"],
        "lerobot_version": executor["lerobot_version"],
        "conda_environment": executor["conda_environment"],
        "execution_backend": executor["execution_backend"],
        "entrypoint_path": "/producer/executor_entrypoint.py",
        "entrypoint_sha256": executor["entrypoint_sha256"],
        "capabilities": executor["capabilities"],
    }
    # The producer path is descriptive and may use another fixed /producer
    # filename; every security-relevant field and the actual bytes must agree.
    hardware_without_path = dict(hardware)
    projection_without_path = dict(projection)
    hardware_without_path.pop("entrypoint_path")
    projection_without_path.pop("entrypoint_path")
    if hardware_without_path != projection_without_path:
        raise SafetyGateError("session_inputs and hardware executor attestations differ")
    signed = session["executor"]
    if (
        signed["repository"] != executor["repository"]
        or signed["repository_commit"] != executor["repository_commit"]
        or signed["entrypoint_sha256"] != executor["entrypoint_sha256"]
    ):
        raise SafetyGateError("rollout session binds a different reviewed executor")


def _reviewed_limits(
    hardware: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    raw_limits = hardware.get("joint_limits")
    raw_deltas = hardware.get("max_step_deltas")
    if not isinstance(raw_limits, Mapping) or not isinstance(raw_deltas, Mapping):
        raise SafetyGateError("reviewed setup lacks joint limits or step limits")
    if set(raw_limits) != set(JOINTS) or set(raw_deltas) != set(JOINTS):
        raise SafetyGateError("reviewed limits must name exactly seven Viola joints")
    limits: dict[str, tuple[float, float]] = {}
    deltas: dict[str, float] = {}
    for joint in JOINTS:
        pair = raw_limits[joint]
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise SafetyGateError(f"reviewed limit for {joint} must have lower/upper values")
        lower, upper = (_finite(pair[0], f"{joint} lower"), _finite(pair[1], f"{joint} upper"))
        if lower >= upper:
            raise SafetyGateError(f"reviewed limit for {joint} is not increasing")
        delta = _finite(raw_deltas[joint], f"{joint} max_step_delta")
        if delta <= 0:
            raise SafetyGateError(f"reviewed step limit for {joint} must be positive")
        limits[joint] = (lower, upper)
        deltas[joint] = delta
    return limits, deltas


def _check_prior_payload(
    bundle: VerifiedBundle,
    phase: str,
    session: VerifiedBundle,
    candidate: VerifiedBundle,
    *,
    expected_hold: VerifiedBundle | None,
    expected_shakedown: VerifiedBundle | None,
) -> None:
    payload = _canonical_object(bundle.payload_file("rollout_evidence.json"))
    if payload.get("phase") != phase or payload.get("status") != "completed":
        raise SafetyGateError(f"prior {phase} evidence is not completed")
    if (
        payload.get("session_bundle_id") != session.bundle_id
        or payload.get("session_content_id") != session.content_id
        or payload.get("policy_bundle_id") != candidate.bundle_id
        or payload.get("policy_content_id") != candidate.content_id
    ):
        raise SafetyGateError(f"prior {phase} evidence belongs to another session or policy")
    expected_prior = {
        "hold": (
            None
            if expected_hold is None
            else {
                "bundle_id": expected_hold.bundle_id,
                "content_id": expected_hold.content_id,
            }
        ),
        "shakedown": (
            None
            if expected_shakedown is None
            else {
                "bundle_id": expected_shakedown.bundle_id,
                "content_id": expected_shakedown.content_id,
            }
        ),
    }
    if payload.get("prior_phase_bundles") != expected_prior:
        raise SafetyGateError(f"prior {phase} evidence has a broken predecessor chain")
    lineage = bundle.manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise SafetyGateError(f"prior {phase} evidence lacks manifest lineage")
    expected_lineage = {
        "session_id": payload.get("session_id"),
        "rollout_session_bundle_id": session.bundle_id,
        "rollout_session_content_id": session.content_id,
        "policy_candidate_bundle_id": candidate.bundle_id,
        "policy_candidate_content_id": candidate.content_id,
        "policy": payload.get("policy"),
        "phase": phase,
    }
    for name, value in expected_lineage.items():
        if lineage.get(name) != value:
            raise SafetyGateError(f"prior {phase} manifest lineage differs at {name}")
    if expected_hold is not None and (
        lineage.get("hold_rollout_evidence_bundle_id") != expected_hold.bundle_id
        or lineage.get("hold_rollout_evidence_content_id") != expected_hold.content_id
    ):
        raise SafetyGateError("prior shakedown manifest does not bind the supplied hold")


def _accepted_bundle(
    path: Path,
    kind: str,
    permission: str,
    *,
    receiver_role: str,
) -> VerifiedBundle:
    try:
        bundle = inspect_bundle(
            path,
            verify_artifacts=True,
            expected_kind=kind,
            required_permission=permission,
        )
    except Exception as exc:
        raise SafetyGateError(f"invalid accepted {kind} bundle: {exc}") from exc
    accepted = [
        receipt
        for receipt in bundle.receipts
        if receipt.get("status") == "accepted"
        and isinstance(receipt.get("actor"), Mapping)
        and receipt["actor"].get("role") == receiver_role
        and receipt.get("bundle_id") == bundle.bundle_id
        and receipt.get("content_id") == bundle.content_id
    ]
    if not accepted:
        raise SafetyGateError(f"{kind} bundle has no {receiver_role} acceptance receipt")
    expected_artifacts = {item["name"] for item in bundle.manifest["artifacts"]}
    if any(
        set(receipt.get("accepted_artifacts", {})) != expected_artifacts
        for receipt in accepted
    ):
        raise SafetyGateError(f"{kind} acceptance does not bind every artifact")
    return bundle


def _artifact_inventory(bundle: VerifiedBundle, name: str) -> str:
    matches = [item for item in bundle.manifest["artifacts"] if item.get("name") == name]
    if len(matches) != 1:
        raise SafetyGateError(f"bundle must contain exactly one {name!r} artifact")
    return _digest(matches[0].get("inventory_sha256"), f"{name} inventory")


def _inside_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SafetyGateError("reviewed setup path escapes its artifact") from exc
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SafetyGateError(f"reviewed setup file is missing: {relative}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SafetyGateError(f"cannot hash reviewed file {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SafetyGateError(f"cannot read canonical evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SafetyGateError(f"evidence must be a JSON object: {path}")
    canonical = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    if raw != canonical:
        raise SafetyGateError(f"evidence is not canonical JSON: {path}")
    return value


class _OperatorPromptTerminal:
    """Readable and writable text handles for one physical operator TTY."""

    def __init__(self, path: str = "/dev/tty") -> None:
        # A single TextIO opened as ``r+`` requires a seekable stream on this
        # Python build. Terminals are not seekable, so keep one handle in each
        # direction instead.
        self._reader = open(path, "r", encoding="utf-8", buffering=1)
        try:
            self._writer = open(path, "w", encoding="utf-8", buffering=1)
        except BaseException:
            self._reader.close()
            raise

    def isatty(self) -> bool:
        return self._reader.isatty() and self._writer.isatty()

    def readline(self) -> str:
        return self._reader.readline()

    def write(self, value: str) -> int:
        return self._writer.write(value)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        try:
            self._reader.close()
        finally:
            self._writer.close()


def _open_operator_terminal() -> _OperatorPromptTerminal:
    try:
        return _OperatorPromptTerminal()
    except OSError as exc:
        raise SafetyGateError("operator confirmation requires an available /dev/tty") from exc


def _is_terminal(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, OSError):
        return False


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SafetyGateError(f"cannot inspect Repo-A checkout: {exc}") from exc


def _utc_now(value: datetime | None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise SafetyGateError("gate clock must be timezone-aware UTC")
    return result.astimezone(UTC)


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise SafetyGateError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SafetyGateError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SafetyGateError(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def _require_phase(value: str) -> str:
    if value not in PHASES:
        raise SafetyGateError(f"phase must be one of {', '.join(PHASES)}")
    return value


def _safe_local_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise SafetyGateError(f"{label} must be one path-safe component")
    return value


def _local_act_duration(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 < float(value) <= 60.0
        or not float("-inf") < float(value) < float("inf")
    ):
        raise SafetyGateError("local ACT duration must be between 0 and 60 seconds")
    return float(value)


def _read_local_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SafetyGateError(f"cannot read local {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SafetyGateError(f"local {label} must be a JSON object")
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SafetyGateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SafetyGateError(f"{label} must be nonempty")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SafetyGateError(f"{label} must be numeric")
    result = float(value)
    if not float("-inf") < result < float("inf"):
        raise SafetyGateError(f"{label} must be finite")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise SafetyGateError(
            f"{label} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
