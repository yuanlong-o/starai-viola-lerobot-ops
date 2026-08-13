"""Independent Repo-A orchestration for the previously deployed ACT policy.

This path deliberately does not consume or mint Repo-B handoffs.  It binds one
exact historical checkpoint, the frozen training dataset, a versioned local
hardware setup, the current clean Repo-A revision, a current E-stop assertion,
and a real operator TTY action.  It then enters the same public-API robot,
limits, 30 Hz control loop, and evidence writer used by the benchmark runner.
"""

from __future__ import annotations

import getpass
import math
import os
import shlex
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viola_handoff import RuntimeIdentity, canonical_json_bytes

from .dataset import DEFAULT_DATASET_ROOT, RELEASE_ID
from .errors import SafetyGateError, ValidationError
from .execute_ops import (
    InteractiveTrialOperator,
    _execution_lease,
    _external_root,
    _material_root_identity,
    _material_root_is_stable,
    _phase_result_payload,
    _safe_component,
)
from .jsonutil import (
    read_json_object,
    require_exact_keys,
    sha256_file,
    sha256_json,
    write_canonical_json,
)
from .local_act import (
    DEFAULT_ACT_CHECKPOINT,
    LEGACY_ACT_MODEL_SHA256,
    LocalActCandidate,
    inspect_local_act_candidate,
    load_local_act_runtime,
    snapshot_local_act_runtime,
)
from .publication_guard import snapshot_tree
from .safety import (
    JOINTS,
    LocalActCheckpointBinding,
    LocalActGateRequest,
    MotionPermit,
    ResolvedSetup,
    authorize_local_act_motion,
    revalidate_local_act_motion,
)
from .wandb_ops import WandbRunIdentity, planned_run, publish_finished_run

DEFAULT_LOCAL_ACT_SETUP = Path("config/local_act_setup.json")
DEFAULT_LOCAL_ACT_EVIDENCE = Path("/mnt/nas02/yz/starai/evidence/v1/local-act")
DEFAULT_WANDB_ENTITY = "yuanlongzhang94"
DEFAULT_WANDB_PROJECT = "starai-viola-policy-benchmark"
DEFAULT_DURATION_S = 10.0

_SETUP_FIELDS = {
    "schema_version",
    "setup_id",
    "robot_port",
    "cameras",
    "calibration_path",
    "reset_protocol_path",
    "joint_limits",
    "max_step_deltas",
}
_CAMERA_FIELDS = {
    "type",
    "index_or_path",
    "width",
    "height",
    "fps",
    "fourcc",
    "warmup_s",
}
_FAILURE_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "authority",
    "session_id",
    "session_binding_id",
    "candidate_id",
    "checkpoint_model_sha256",
    "dataset_release_id",
    "trial",
    "operator",
    "estop_tested_at_utc",
    "setup_id",
    "setup_hashes",
    "duration_s",
    "speed_scale",
    "repo_commit",
    "stage",
    "error_type",
    "error_message",
    "recorded_at_utc",
    "intent_receipt_sha256",
    "partial_motion_inventory_sha256",
    "partial_motion_inventory_error",
    "wandb_entity",
    "wandb_project",
    "motion_may_have_started",
    "ready",
    "rerun_in_place",
    "recovery_command",
}


@dataclass(frozen=True, slots=True)
class LocalActExecutionApi:
    """Lazy hardware-facing dependencies, injectable in disconnected tests."""

    execute_phase: Callable[..., Any]
    robot_factory: Callable[[MotionPermit], Any]
    evidence_factory: Callable[[Path], Any]
    operator_factory: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class LocalActRunOutcome:
    status: str
    task_success: bool
    failure_code: str
    session_id: str
    trial: str
    candidate_id: str
    material_root: Path
    result_path: Path
    wandb_url: str

    def render_text(self) -> str:
        return "\n".join(
            [
                f"Control status: {self.status}",
                f"Task success: {self.task_success}",
                f"Task failure code: {self.failure_code}",
                f"Session: {self.session_id}",
                f"Trial: {self.trial}",
                f"Candidate: {self.candidate_id}",
                f"Evidence: {self.material_root}",
                f"W&B: {self.wandb_url}",
                "Authority: Repo A local-only; this is not a Repo-B READY handoff.",
            ]
        )


@dataclass(frozen=True, slots=True)
class LocalActRecoveryOutcome:
    status: str
    material_root: Path
    receipt_path: Path
    wandb_url: str

    def render_text(self) -> str:
        return "\n".join(
            [
                "Local ACT failure evidence synchronized",
                f"Attempt: {self.material_root}",
                f"Receipt: {self.receipt_path}",
                f"W&B: {self.wandb_url}",
                "No camera, serial device, motor, policy, or operator prompt was opened.",
            ]
        )


def run_command(
    *,
    repo_root: Path,
    checkpoint: Path = DEFAULT_ACT_CHECKPOINT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    setup_path: Path = DEFAULT_LOCAL_ACT_SETUP,
    evidence_root: Path = DEFAULT_LOCAL_ACT_EVIDENCE,
    operator: str | None = None,
    trial: str | None = None,
    duration_s: float = DEFAULT_DURATION_S,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    identity_capture: Callable[..., RuntimeIdentity] = RuntimeIdentity.capture,
    intent_publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
    execution_api_factory: Callable[[], LocalActExecutionApi] | None = None,
    now: datetime | None = None,
) -> LocalActRunOutcome:
    """Run one supervised local ACT inference without any Repo-B dependency."""

    repository = repo_root.resolve()
    _require_local_act_cuda()
    output = _external_root(evidence_root, repository, "local ACT evidence_root")
    duration = _duration(duration_s)
    current_time = _utc_now(now)
    operator_name = _safe_component(operator or getpass.getuser(), "operator")
    trial_name = _safe_component(
        trial or current_time.strftime("act-%Y%m%dT%H%M%SZ"), "trial"
    )

    identity = identity_capture(role="pc_a", repo_root=repository)
    setup = load_local_act_setup(
        setup_path,
        repo_root=repository,
        identity=identity,
    )
    _require_local_devices(setup)
    candidate = inspect_local_act_candidate(checkpoint, dataset_root)
    gate_request = _gate_request(
        setup=setup,
        candidate=candidate,
        operator=operator_name,
        trial=trial_name,
        duration_s=duration,
        repository=repository,
        tested_at=current_time,
    )

    # This prompt is the operator's physical E-stop assertion and ARM action.
    # It is local authority; no Repo-B receipt, bundle, or confirmation exists.
    permit = authorize_local_act_motion(gate_request, identity=identity)
    if permit.candidate_bundle_id != candidate.candidate_id:
        raise SafetyGateError("local ACT permit differs from the inspected checkpoint")

    with _execution_lease(permit.robot_port):
        return _run_authorized(
            request=gate_request,
            permit=permit,
            candidate=candidate,
            identity=identity,
            repository=repository,
            output=output,
            duration_s=duration,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            identity_capture=identity_capture,
            intent_publisher=intent_publisher,
            execution_api_factory=execution_api_factory or _production_execution_api,
        )


def load_local_act_setup(
    path: str | Path,
    *,
    repo_root: Path,
    identity: RuntimeIdentity,
) -> ResolvedSetup:
    """Load the short, versioned setup file and reproduce every safety hash."""

    repository = repo_root.resolve()
    source = Path(path)
    if not source.is_absolute():
        source = repository / source
    raw = require_exact_keys(
        read_json_object(source, label="local ACT setup"),
        _SETUP_FIELDS,
        label="local ACT setup",
    )
    if raw["schema_version"] != 1:
        raise ValidationError("local ACT setup schema_version must be 1")
    setup_id = _safe_component(raw["setup_id"], "setup_id")
    robot_port = _device_path(raw["robot_port"], "robot_port")
    cameras = _camera_setup(raw["cameras"])
    limits = _joint_limits(raw["joint_limits"])
    deltas = _step_limits(raw["max_step_deltas"])
    calibration = _repo_file(raw["calibration_path"], repository, "calibration")
    reset = _repo_file(raw["reset_protocol_path"], repository, "reset protocol")
    entrypoint = _repo_file(
        "src/viola_ops/execution.py", repository, "local ACT executor"
    )
    setup_hashes = {
        "calibration": sha256_file(calibration),
        "camera": sha256_json(cameras),
        "robot": sha256_json(
            {
                "robot_port": robot_port,
                "joint_limits": {name: list(value) for name, value in limits.items()},
                "max_step_deltas": deltas,
                "speed_scale": 1.0,
            }
        ),
        "reset": sha256_file(reset),
    }
    return ResolvedSetup(
        setup_id=setup_id,
        robot_port=robot_port,
        camera_configs=cameras,
        calibration_path=calibration,
        reset_protocol_path=reset,
        executor_entrypoint=entrypoint,
        setup_hashes=setup_hashes,
        absolute_limits=limits,
        max_step_deltas=deltas,
        executor_identity={
            "repository_commit": identity.repository_commit,
            "python_version": identity.python_version,
            "lerobot_version": identity.lerobot_version,
            "conda_environment": identity.conda_environment,
        },
    )


def recover_failure_command(
    attempt: Path,
    *,
    repo_root: Path,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    identity_capture: Callable[..., RuntimeIdentity] = RuntimeIdentity.capture,
    publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
) -> LocalActRecoveryOutcome:
    """Publish an immutable failure marker without repeating motion or opening hardware."""

    repository = repo_root.resolve()
    material = attempt.absolute()
    _external_root(material, repository, "local ACT recovery attempt")
    root_identity = _material_root_identity(material)
    identity = identity_capture(role="pc_a", repo_root=repository)
    marker_path = material / "LOCAL_ACT_FAILURE.json"
    marker = _load_failure_marker(marker_path)
    if marker["repo_commit"] != identity.repository_commit:
        raise ValidationError("local ACT failure belongs to another Repo-A revision")
    if (
        marker["wandb_entity"] != wandb_entity
        or marker["wandb_project"] != wandb_project
    ):
        raise ValidationError("local ACT failure belongs to another W&B destination")
    intent_path = material / "LOCAL_ACT_INTENT_WANDB_SYNCED.json"
    intent_sha256 = marker["intent_receipt_sha256"]
    marker_sha256 = sha256_file(marker_path)
    if sha256_file(intent_path) != intent_sha256:
        raise ValidationError("local ACT failure intent receipt changed")
    _validate_failure_intent(marker, intent_path)
    _verify_partial_motion(material, marker)
    if not _material_root_is_stable(
        material, material, repository, root_identity
    ):
        raise ValidationError("local ACT recovery attempt directory was replaced")
    run, receipt = _publish_failure(
        marker=marker,
        marker_path=marker_path,
        material=material,
        entity=wandb_entity,
        project=wandb_project,
        publisher=publisher,
    )
    if not _material_root_is_stable(
        material, material, repository, root_identity
    ):
        raise ValidationError("local ACT recovery attempt changed during publication")
    if sha256_file(marker_path) != marker_sha256:
        raise ValidationError("local ACT failure marker changed during publication")
    if sha256_file(intent_path) != intent_sha256:
        raise ValidationError("local ACT intent receipt changed during publication")
    _verify_partial_motion(material, marker)
    return LocalActRecoveryOutcome("failure_recorded", material, receipt, run.url)


def _run_authorized(
    *,
    request: LocalActGateRequest,
    permit: MotionPermit,
    candidate: LocalActCandidate,
    identity: RuntimeIdentity,
    repository: Path,
    output: Path,
    duration_s: float,
    wandb_entity: str,
    wandb_project: str,
    identity_capture: Callable[..., RuntimeIdentity],
    intent_publisher: Callable[..., WandbRunIdentity],
    execution_api_factory: Callable[[], LocalActExecutionApi],
) -> LocalActRunOutcome:
    """Execute while the caller holds the reviewed robot's host-wide lease."""

    material = output / permit.session_id / permit.trial
    material.parent.mkdir(parents=True, exist_ok=True)
    try:
        material.mkdir()
    except FileExistsError as exc:
        raise ValidationError(
            f"local ACT attempt already exists; refusing to repeat motion: {material}"
        ) from exc
    root_identity = _material_root_identity(material)

    def require_material(boundary: str) -> None:
        if not _material_root_is_stable(
            material, output, repository, root_identity
        ):
            raise SafetyGateError(f"local ACT evidence path changed {boundary}")

    def revalidate(
        boundary: str,
        runtime_snapshot: Any | None = None,
        *,
        execution_finished: bool = False,
    ) -> RuntimeIdentity:
        current = identity_capture(role="pc_a", repo_root=repository)
        revalidate_local_act_motion(
            request,
            permit,
            identity=current,
            allow_consumed=execution_finished,
        )
        current_candidate = inspect_local_act_candidate(
            candidate.checkpoint, candidate.dataset_root
        )
        if current_candidate.candidate_id != candidate.candidate_id:
            raise SafetyGateError(f"local ACT source bytes changed {boundary}")
        if runtime_snapshot is not None:
            runtime_snapshot.verify()
        require_material(boundary)
        return current

    intent_path = _publish_intent(
        permit=permit,
        candidate=candidate,
        identity=identity,
        material=material,
        duration_s=duration_s,
        entity=wandb_entity,
        project=wandb_project,
        publisher=intent_publisher,
    )
    require_material("while publishing the pre-hardware intent")
    identity = revalidate("before the runtime snapshot")

    result_path = material / "LOCAL_ACT_RESULT.json"
    stage = "runtime_snapshot"
    motion_may_have_started = False
    try:
        with snapshot_local_act_runtime(candidate) as runtime_snapshot:
            identity = revalidate("before hardware import", runtime_snapshot)
            stage = "hardware_import"
            api = execution_api_factory()
            evidence = api.evidence_factory(material / "motion_record")
            stage = "operator_setup"
            trial_operator = api.operator_factory()
            try:
                def revalidate_trial() -> None:
                    nonlocal identity
                    identity = revalidate(
                        "immediately after START", runtime_snapshot
                    )

                stage = "execution"
                motion_may_have_started = True
                result = api.execute_phase(
                    permit,
                    runtime_snapshot.candidate,
                    runtime_factory=load_local_act_runtime,
                    robot_factory=api.robot_factory,
                    evidence_factory=evidence,
                    operator=trial_operator,
                    safety_monitor=trial_operator,
                    revalidate_authority=revalidate_trial,
                    local_act_duration_s=duration_s,
                )
            finally:
                trial_operator.close()
            stage = "local_result"
            _validate_local_result(result, permit=permit, duration_s=duration_s)
            motion_snapshot = snapshot_tree(
                material / "motion_record", label="local ACT motion evidence"
            )
            result_payload = {
                "schema_version": 1,
                "kind": "repo_a_local_act_result",
                "authority": "local_only",
                "ready": False,
                "benchmark_evidence": False,
                "session_id": permit.session_id,
                "candidate_id": candidate.candidate_id,
                "checkpoint_model_sha256": candidate.model_sha256,
                "dataset_release_id": candidate.dataset_release_id,
                "deployment_queue_actions": 10,
                "duration_s": duration_s,
                "operator": permit.operator,
                "repo_commit": identity.repository_commit,
                "setup_id": permit.setup_id,
                "setup_hashes": dict(permit.setup_hashes),
                "intent_receipt_sha256": sha256_file(intent_path),
                "motion_inventory_sha256": motion_snapshot.inventory[
                    "inventory_sha256"
                ],
                "phase_result": _phase_result_payload(result),
            }
            write_canonical_json(result_path, result_payload)
            identity = revalidate(
                "after hardware disconnect",
                runtime_snapshot,
                execution_finished=True,
            )
            _require_same_tree(
                motion_snapshot,
                boundary="after hardware disconnect",
            )
            run = _publish_result(
                payload=result_payload,
                result_path=result_path,
                permit=permit,
                candidate=candidate,
                identity=identity,
                material=material,
                entity=wandb_entity,
                project=wandb_project,
                publisher=intent_publisher,
            )
            if sha256_file(result_path) != sha256_json(result_payload):
                raise SafetyGateError("local ACT result bytes changed during publication")
            revalidate(
                "after result publication",
                runtime_snapshot,
                execution_finished=True,
            )
            _require_same_tree(motion_snapshot, boundary="after result publication")
    except BaseException as exc:
        motion_inventory, motion_error = _partial_motion_binding(material)
        marker = {
            "schema_version": 1,
            "kind": "repo_a_local_act_failure",
            "status": "failed_not_ready",
            "authority": "local_only",
            "session_id": permit.session_id,
            "session_binding_id": permit.session_bundle_id,
            "candidate_id": candidate.candidate_id,
            "checkpoint_model_sha256": candidate.model_sha256,
            "dataset_release_id": candidate.dataset_release_id,
            "trial": permit.trial,
            "operator": permit.operator,
            "estop_tested_at_utc": permit.estop_tested_at.isoformat(),
            "setup_id": permit.setup_id,
            "setup_hashes": dict(permit.setup_hashes),
            "duration_s": duration_s,
            "speed_scale": permit.speed_scale,
            "repo_commit": identity.repository_commit,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": _safe_error(exc),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "intent_receipt_sha256": sha256_file(intent_path),
            "partial_motion_inventory_sha256": motion_inventory,
            "partial_motion_inventory_error": motion_error,
            "wandb_entity": wandb_entity,
            "wandb_project": wandb_project,
            "motion_may_have_started": motion_may_have_started,
            "ready": False,
            "rerun_in_place": False,
            "recovery_command": _failure_recovery_command(
                material,
                entity=wandb_entity,
                project=wandb_project,
            ),
        }
        try:
            require_material("while recording failure evidence")
            marker_path = write_canonical_json(
                material / "LOCAL_ACT_FAILURE.json", marker
            )
            try:
                _validate_failure_intent(marker, intent_path)
                _verify_partial_motion(material, marker)
                _publish_failure(
                    marker=marker,
                    marker_path=marker_path,
                    material=material,
                    entity=wandb_entity,
                    project=wandb_project,
                    publisher=intent_publisher,
                )
                require_material("after publishing failure evidence")
                if sha256_file(marker_path) != sha256_json(marker):
                    raise ValidationError(
                        "local ACT failure marker changed during publication"
                    )
                if sha256_file(intent_path) != marker["intent_receipt_sha256"]:
                    raise ValidationError(
                        "local ACT intent receipt changed during failure publication"
                    )
                _verify_partial_motion(material, marker)
            except Exception as publish_error:
                exc.add_note(
                    "failure W&B publication also failed; use the recorded "
                    f"recovery command ({type(publish_error).__name__})"
                )
        except Exception as marker_error:
            exc.add_note(f"local failure marker also failed: {marker_error}")
        raise

    return LocalActRunOutcome(
        status=result.status,
        task_success=result.trials[0].outcome.success,
        failure_code=result.trials[0].outcome.failure_code,
        session_id=permit.session_id,
        trial=permit.trial,
        candidate_id=candidate.candidate_id,
        material_root=material,
        result_path=result_path,
        wandb_url=run.url,
    )


def _gate_request(
    *,
    setup: ResolvedSetup,
    candidate: LocalActCandidate,
    operator: str,
    trial: str,
    duration_s: float,
    repository: Path,
    tested_at: datetime,
) -> LocalActGateRequest:
    estop = {
        "schema_version": 1,
        "operator": operator,
        "tested_at_utc": tested_at.isoformat(),
        "passed": True,
        "attestation": "typed local ACT ARM phrase after physical E-stop test",
    }
    return LocalActGateRequest(
        setup=setup,
        checkpoint=LocalActCheckpointBinding(
            root=candidate.checkpoint,
            inventory=candidate.checkpoint_inventory,
            model_sha256=candidate.model_sha256,
            dataset_release_id=candidate.dataset_release_id,
            dataset_inventory_sha256=candidate.dataset_inventory[
                "inventory_sha256"
            ],
            dataset_metadata_inventory_sha256=candidate.dataset_metadata_inventory[
                "inventory_sha256"
            ],
        ),
        operator=operator,
        estop_tested_at=tested_at,
        estop_attestation_sha256=sha256_json(estop),
        estop_passed=True,
        trial=trial,
        duration_s=duration_s,
        repository_root=repository,
    )


def _publish_intent(
    *,
    permit: MotionPermit,
    candidate: LocalActCandidate,
    identity: RuntimeIdentity,
    material: Path,
    duration_s: float,
    entity: str,
    project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> Path:
    binding = {
        "operation": "repo_a_local_act_intent",
        "session_id": permit.session_id,
        "session_binding_id": permit.session_bundle_id,
        "candidate_id": candidate.candidate_id,
        "checkpoint_model_sha256": candidate.model_sha256,
        "dataset_release_id": candidate.dataset_release_id,
        "trial": permit.trial,
        "operator": permit.operator,
        "estop_tested_at_utc": permit.estop_tested_at.isoformat(),
        "duration_s": duration_s,
        "speed_scale": permit.speed_scale,
        "repo_commit": identity.repository_commit,
        "setup_id": permit.setup_id,
        "setup_hashes": dict(permit.setup_hashes),
    }
    run = planned_run(entity, project, f"local-act-intent-{sha256_json(binding)[:16]}")
    publisher(
        run,
        job_type="viola-local-act-intent",
        config=binding,
        summary={
            "operator_authorized": True,
            "repo_b_confirmation_required": False,
            "hardware_imported": False,
            "hardware_connected": False,
        },
    )
    return write_canonical_json(
        material / "LOCAL_ACT_INTENT_WANDB_SYNCED.json",
        {
            "schema_version": 1,
            "binding": binding,
            "wandb": run.binding(),
        },
    )


def _publish_result(
    *,
    payload: Mapping[str, Any],
    result_path: Path,
    permit: MotionPermit,
    candidate: LocalActCandidate,
    identity: RuntimeIdentity,
    material: Path,
    entity: str,
    project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> WandbRunIdentity:
    result_sha256 = sha256_file(result_path)
    binding = {
        "operation": "repo_a_local_act_result",
        "session_id": permit.session_id,
        "candidate_id": candidate.candidate_id,
        "trial": permit.trial,
        "repo_commit": identity.repository_commit,
        "result_sha256": result_sha256,
        "motion_inventory_sha256": payload["motion_inventory_sha256"],
    }
    run = planned_run(entity, project, f"local-act-result-{result_sha256[:16]}")
    phase = payload["phase_result"]
    trials = phase["trials"]
    trial_outcome = trials[0]["outcome"]
    publisher(
        run,
        job_type="viola-local-act-result",
        config=binding,
        summary={
            "status": "completed" if phase["terminal_event"] is None else "unsafe",
            "terminal_event": phase["terminal_event"],
            "task_success": trial_outcome["success"],
            "task_failure_code": trial_outcome["failure_code"],
            "actions_sent": sum(int(item["actions"]) for item in trials),
            "ready": False,
            "benchmark_evidence": False,
        },
    )
    write_canonical_json(
        material / "LOCAL_ACT_RESULT_WANDB_SYNCED.json",
        {
            "schema_version": 1,
            "binding": binding,
            "wandb": run.binding(),
        },
    )
    return run


def _publish_failure(
    *,
    marker: Mapping[str, Any],
    marker_path: Path,
    material: Path,
    entity: str,
    project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> tuple[WandbRunIdentity, Path]:
    """Publish or exactly resume terminal failure evidence."""

    stored = _load_failure_marker(marker_path)
    if stored != dict(marker):
        raise ValidationError("local ACT failure marker differs from supplied evidence")
    marker_sha256 = sha256_file(marker_path)
    binding = {
        "operation": "repo_a_local_act_failure",
        "session_id": stored["session_id"],
        "candidate_id": stored["candidate_id"],
        "trial": stored["trial"],
        "repo_commit": stored["repo_commit"],
        "stage": stored["stage"],
        "marker_sha256": marker_sha256,
        "partial_motion_inventory_sha256": stored[
            "partial_motion_inventory_sha256"
        ],
    }
    run = planned_run(entity, project, f"local-act-failure-{marker_sha256[:16]}")
    publisher(
        run,
        job_type="viola-local-act-failure",
        config=binding,
        summary={
            "status": "failed_not_ready",
            "motion_may_have_started": stored["motion_may_have_started"],
            "terminal_recorded": True,
            "ready": False,
            "rerun_in_place": False,
        },
    )
    receipt = {
        "schema_version": 1,
        "operation": "repo_a_local_act_failure",
        "binding": binding,
        "wandb": run.binding(),
    }
    return run, write_canonical_json(
        material / "LOCAL_ACT_FAILURE_WANDB_SYNCED.json", receipt
    )


def _production_execution_api() -> LocalActExecutionApi:
    """Import hardware-capable code only after ARM and online intent evidence."""

    from .evidence import PhaseEvidenceFactory
    from .execution import execute_phase
    from .hardware import SafeViolaRobot, config_from_permit

    return LocalActExecutionApi(
        execute_phase=execute_phase,
        robot_factory=lambda permit: SafeViolaRobot(config_from_permit(permit), permit),
        evidence_factory=lambda root: PhaseEvidenceFactory(root, fps=30, queue_size=64),
        operator_factory=InteractiveTrialOperator,
    )


def _load_failure_marker(path: Path) -> dict[str, Any]:
    marker = require_exact_keys(
        read_json_object(path, label="local ACT failure marker"),
        _FAILURE_FIELDS,
        label="local ACT failure marker",
    )
    if path.read_bytes() != canonical_json_bytes(marker):
        raise ValidationError("local ACT failure marker is not canonical JSON")
    if (
        marker["schema_version"] != 1
        or marker["kind"] != "repo_a_local_act_failure"
        or marker["status"] != "failed_not_ready"
        or marker["authority"] != "local_only"
        or marker["ready"] is not False
        or marker["rerun_in_place"] is not False
        or not isinstance(marker["motion_may_have_started"], bool)
    ):
        raise ValidationError("local ACT failure marker has invalid terminal semantics")
    for field in (
        "session_id",
        "dataset_release_id",
        "candidate_id",
        "trial",
        "operator",
        "setup_id",
        "stage",
        "error_type",
        "error_message",
        "recovery_command",
        "wandb_entity",
        "wandb_project",
    ):
        if not isinstance(marker[field], str) or not marker[field]:
            raise ValidationError(f"local ACT failure {field} must be nonempty")
    for field in (
        "session_binding_id",
        "candidate_id",
        "checkpoint_model_sha256",
        "repo_commit",
        "intent_receipt_sha256",
    ):
        expected_length = 40 if field == "repo_commit" else 64
        if (
            not isinstance(marker[field], str)
            or len(marker[field]) != expected_length
            or any(character not in "0123456789abcdef" for character in marker[field])
        ):
            raise ValidationError(f"local ACT failure {field} is malformed")
    _duration(marker["duration_s"])
    if marker["speed_scale"] != 0.25:
        raise ValidationError("local ACT failure speed scale is not 0.25")
    if not isinstance(marker["setup_hashes"], Mapping) or set(
        marker["setup_hashes"]
    ) != {"calibration", "camera", "robot", "reset"}:
        raise ValidationError("local ACT failure setup hash binding is incomplete")
    for name, digest in marker["setup_hashes"].items():
        if not _is_digest(digest):
            raise ValidationError(f"local ACT failure setup hash {name} is malformed")
    for field in ("estop_tested_at_utc", "recorded_at_utc"):
        _utc_timestamp(marker[field], f"local ACT failure {field}")
    if _utc_timestamp(
        marker["estop_tested_at_utc"], "local ACT failure E-stop timestamp"
    ) > _utc_timestamp(marker["recorded_at_utc"], "local ACT failure recorded timestamp"):
        raise ValidationError("local ACT failure predates its E-stop assertion")
    if marker["session_id"] != f"local-act-{marker['session_binding_id'][:24]}":
        raise ValidationError("local ACT failure session identity is inconsistent")
    if marker["checkpoint_model_sha256"] != LEGACY_ACT_MODEL_SHA256:
        raise ValidationError("local ACT failure names an unreviewed model")
    if marker["dataset_release_id"] != RELEASE_ID:
        raise ValidationError("local ACT failure names another dataset release")
    if marker["stage"] not in {
        "runtime_snapshot",
        "hardware_import",
        "operator_setup",
        "execution",
        "local_result",
    }:
        raise ValidationError("local ACT failure stage is invalid")
    planned_run(
        marker["wandb_entity"],
        marker["wandb_project"],
        "local-act-failure-validation",
    )
    expected_recovery = _failure_recovery_command(
        path.parent.absolute(),
        entity=marker["wandb_entity"],
        project=marker["wandb_project"],
    )
    if marker["recovery_command"] != expected_recovery:
        raise ValidationError("local ACT failure recovery command is not path-bound")
    return dict(marker)


def _validate_failure_intent(marker: Mapping[str, Any], intent_path: Path) -> None:
    """Bind a terminal marker to the exact finished pre-hardware intent."""

    intent = require_exact_keys(
        read_json_object(intent_path, label="local ACT intent receipt"),
        {"schema_version", "binding", "wandb"},
        label="local ACT intent receipt",
    )
    if intent_path.read_bytes() != canonical_json_bytes(intent):
        raise ValidationError("local ACT intent receipt is not canonical JSON")
    binding = require_exact_keys(
        intent["binding"],
        {
            "operation",
            "session_id",
            "session_binding_id",
            "candidate_id",
            "checkpoint_model_sha256",
            "dataset_release_id",
            "trial",
            "operator",
            "estop_tested_at_utc",
            "duration_s",
            "speed_scale",
            "repo_commit",
            "setup_id",
            "setup_hashes",
        },
        label="local ACT intent binding",
    )
    wandb = require_exact_keys(
        intent["wandb"],
        {"run_id", "url"},
        label="local ACT intent W&B binding",
    )
    if intent["schema_version"] != 1 or binding["operation"] != "repo_a_local_act_intent":
        raise ValidationError("local ACT intent receipt has invalid semantics")
    expected_binding = {
        "session_id": marker["session_id"],
        "session_binding_id": marker["session_binding_id"],
        "candidate_id": marker["candidate_id"],
        "checkpoint_model_sha256": marker["checkpoint_model_sha256"],
        "dataset_release_id": marker["dataset_release_id"],
        "trial": marker["trial"],
        "operator": marker["operator"],
        "estop_tested_at_utc": marker["estop_tested_at_utc"],
        "duration_s": marker["duration_s"],
        "speed_scale": marker["speed_scale"],
        "repo_commit": marker["repo_commit"],
        "setup_id": marker["setup_id"],
        "setup_hashes": marker["setup_hashes"],
    }
    if {name: binding[name] for name in expected_binding} != expected_binding:
        raise ValidationError("local ACT failure differs from its online intent")
    expected_run = f"local-act-intent-{sha256_json(dict(binding))[:16]}"
    expected_url = (
        f"https://wandb.ai/{marker['wandb_entity']}/{marker['wandb_project']}"
        f"/runs/{expected_run}"
    )
    if wandb["run_id"] != expected_run or wandb["url"] != expected_url:
        raise ValidationError("local ACT intent W&B destination or run identity differs")


def _failure_recovery_command(material: Path, *, entity: str, project: str) -> str:
    return shlex.join(
        [
            "viola-ops",
            "act",
            "recover-failure",
            "--attempt",
            str(material.absolute()),
            "--wandb-entity",
            entity,
            "--wandb-project",
            project,
        ]
    )


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValidationError(f"{label} must use UTC")
    return parsed.astimezone(UTC)


def _partial_motion_binding(material: Path) -> tuple[str | None, str | None]:
    motion = material / "motion_record"
    if not motion.exists():
        return None, "motion_record_not_created"
    try:
        snapshot = snapshot_tree(motion, label="partial local ACT motion evidence")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {_safe_error(exc)}"
    return str(snapshot.inventory["inventory_sha256"]), None


def _verify_partial_motion(material: Path, marker: Mapping[str, Any]) -> None:
    motion = material / "motion_record"
    expected = marker["partial_motion_inventory_sha256"]
    error = marker["partial_motion_inventory_error"]
    if expected is None:
        if error != "motion_record_not_created" or motion.exists():
            raise ValidationError(
                "local ACT failure cannot prove that motion evidence was absent"
            )
        return
    if not isinstance(expected, str) or len(expected) != 64 or error is not None:
        raise ValidationError("local ACT failure partial-motion binding is malformed")
    current = snapshot_tree(
        motion, label="partial local ACT motion evidence"
    )
    if current.inventory["inventory_sha256"] != expected:
        raise ValidationError("local ACT partial motion evidence changed")


def _validate_local_result(
    result: Any,
    *,
    permit: MotionPermit,
    duration_s: float,
) -> None:
    """Reject a partial or contradictory result before publishing evidence."""

    if (
        getattr(result, "session_id", None) != permit.session_id
        or getattr(result, "policy", None) != "act"
        or getattr(result, "phase", None) != "local_act"
        or getattr(result, "speed_scale", None) != 0.25
        or getattr(result, "held_action", None) is not None
    ):
        raise ValidationError("local ACT result differs from its motion permit")
    trials = getattr(result, "trials", None)
    if not isinstance(trials, tuple) or len(trials) != 1:
        raise ValidationError("local ACT result must contain exactly one trial")
    trial = trials[0]
    expected_trial_id = f"{permit.session_id}-local_act-01"
    if (
        trial.trial_id != expected_trial_id
        or trial.index != 0
        or dict(trial.condition)
        != {
            "condition_id": "local_act_01",
            "stratum": "local_act",
            "blue_axis": None,
            "blue_offset_mm": 0.0,
            "red_axis": None,
            "red_offset_mm": 0.0,
        }
    ):
        raise ValidationError("local ACT result has an unexpected trial identity")
    terminal = getattr(result, "terminal_event", None)
    safety_events = tuple(trial.safety_events)
    if terminal is None:
        expected_actions = round(duration_s * 30)
        if (
            getattr(result, "terminal_reason", None) is not None
            or safety_events
            or trial.actions != expected_actions
            or trial.replans != (expected_actions + 9) // 10
            or len(trial.inference_latency_ms) != expected_actions
            or len(trial.control_latency_ms) != expected_actions
            or abs(float(trial.duration_sec) - duration_s) > 0.5
        ):
            raise ValidationError("completed local ACT result lacks its full bounded trace")
    elif (
        safety_events != (terminal,)
        or not isinstance(getattr(result, "terminal_reason", None), str)
        or not result.terminal_reason
        or getattr(result, "status", None) != "unsafe_local_act"
    ):
        raise ValidationError("unsafe local ACT result has contradictory terminal evidence")
    trial.outcome.validate()
    for milestone in (
        trial.outcome.blue_completed_sec,
        trial.outcome.red_completed_sec,
        trial.outcome.completion_time_sec,
    ):
        if milestone is not None and milestone > duration_s:
            raise ValidationError("local ACT outcome milestone exceeds the run duration")
    for name, samples in (
        ("inference", trial.inference_latency_ms),
        ("control", trial.control_latency_ms),
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in samples
        ):
            raise ValidationError(f"local ACT {name} timings must be finite and nonnegative")


def _camera_setup(value: Any) -> dict[str, dict[str, Any]]:
    cameras = require_exact_keys(value, {"front", "up"}, label="local ACT cameras")
    result: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    for name in ("front", "up"):
        camera = dict(
            require_exact_keys(
                cameras[name], _CAMERA_FIELDS, label=f"local ACT {name} camera"
            )
        )
        if camera["type"] != "opencv":
            raise ValidationError(f"local ACT {name} camera type must be opencv")
        path = _device_path(camera["index_or_path"], f"{name} camera")
        if (camera["width"], camera["height"], camera["fps"]) != (640, 480, 30):
            raise ValidationError(f"local ACT {name} camera must be 640x480 at 30 fps")
        expected_fourcc = "MJPG" if name == "front" else "YUYV"
        if camera["fourcc"] != expected_fourcc or camera["warmup_s"] != 8:
            raise ValidationError(
                f"local ACT {name} camera must retain the historical "
                f"{expected_fourcc} format and 8-second warmup"
            )
        camera["index_or_path"] = path
        result[name] = camera
        paths.append(path)
    if len(set(paths)) != 2:
        raise ValidationError("local ACT cameras must use distinct devices")
    return result


def _require_local_devices(setup: ResolvedSetup) -> None:
    """Fail before ARM/W&B when this login cannot open the reviewed devices."""

    paths = {
        "robot serial": setup.robot_port,
        "front camera": str(setup.camera_configs["front"]["index_or_path"]),
        "up camera": str(setup.camera_configs["up"]["index_or_path"]),
    }
    for label, value in paths.items():
        path = Path(value)
        try:
            details = path.stat()
        except OSError as exc:
            raise ValidationError(f"local ACT {label} is unavailable: {path}: {exc}") from exc
        if not stat.S_ISCHR(details.st_mode):
            raise ValidationError(f"local ACT {label} is not a character device: {path}")
        if not os.access(path, os.R_OK | os.W_OK):
            if label == "robot serial":
                raise ValidationError(
                    "local ACT robot serial access is unavailable in this login. Run from "
                    "the dialout group: sg dialout -c 'CUDA_VISIBLE_DEVICES=0 "
                    "WANDB_MODE=online viola-ops act run'"
                )
            raise ValidationError(f"local ACT {label} is not readable/writable: {path}")


def _require_local_act_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValidationError(
            "local ACT requires CUDA_VISIBLE_DEVICES=0 to reproduce the deployed GPU"
        )


def _joint_limits(value: Any) -> dict[str, tuple[float, float]]:
    limits = require_exact_keys(value, set(JOINTS), label="local ACT joint limits")
    result: dict[str, tuple[float, float]] = {}
    for joint in JOINTS:
        pair = limits[joint]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValidationError(f"local ACT {joint} limits must be [lower, upper]")
        lower = _finite(pair[0], f"{joint} lower limit")
        upper = _finite(pair[1], f"{joint} upper limit")
        if lower >= upper:
            raise ValidationError(f"local ACT {joint} limits are not increasing")
        result[joint] = (lower, upper)
    return result


def _step_limits(value: Any) -> dict[str, float]:
    deltas = require_exact_keys(value, set(JOINTS), label="local ACT step limits")
    result = {joint: _finite(deltas[joint], f"{joint} step limit") for joint in JOINTS}
    if any(delta <= 0 for delta in result.values()):
        raise ValidationError("local ACT step limits must be positive")
    return result


def _repo_file(value: Any, repository: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"local ACT {label} path must be nonempty")
    path = Path(value)
    if not path.is_absolute():
        path = repository / path
    resolved = path.resolve()
    if not resolved.is_relative_to(repository) or not resolved.is_file():
        raise ValidationError(f"local ACT {label} must be a file inside Repo A")
    return resolved


def _device_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/dev/"):
        raise ValidationError(f"local ACT {label} must be an explicit /dev path")
    return value


def _duration(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 60.0
    ):
        raise ValidationError("local ACT duration must be between 0 and 60 seconds")
    return float(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"local ACT {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"local ACT {label} must be finite")
    return result


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() != UTC.utcoffset(current):
        raise ValidationError("local ACT time must use UTC")
    return current


def _require_same_tree(snapshot: Any, *, boundary: str) -> None:
    current = snapshot_tree(snapshot.path, label="local ACT motion evidence")
    if (current.device, current.inode) != (snapshot.device, snapshot.inode):
        raise SafetyGateError(f"local ACT motion evidence root changed {boundary}")
    if current.inventory != snapshot.inventory:
        raise SafetyGateError(f"local ACT motion evidence bytes changed {boundary}")


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:500] or type(exc).__name__


__all__ = [
    "DEFAULT_DURATION_S",
    "DEFAULT_LOCAL_ACT_EVIDENCE",
    "DEFAULT_LOCAL_ACT_SETUP",
    "DEFAULT_WANDB_ENTITY",
    "LocalActExecutionApi",
    "LocalActRecoveryOutcome",
    "LocalActRunOutcome",
    "load_local_act_setup",
    "recover_failure_command",
    "run_command",
]
