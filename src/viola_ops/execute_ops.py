"""Human-facing orchestration for one authorized physical inference phase."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import select
import socket
import stat
import termios
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viola_handoff import RuntimeIdentity, VerifiedBundle, canonical_json_bytes, inspect_bundle

from .errors import SafetyGateError, ValidationError
from .jsonutil import read_json_object, require_exact_keys, sha256_json, write_canonical_json
from .safety import GateRequest, MotionPermit, authorize_motion, revalidate_motion
from .wandb_ops import WandbRunIdentity, planned_run, publish_finished_run

_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_EXECUTION_FAILURE_STAGES = {
    "support_import",
    "runtime_snapshot",
    "final_revalidation",
    "hardware_import",
    "operator_setup",
    "execution_revalidation",
    "execution",
    "local_result",
    "post_execution_revalidation",
}
_EXECUTION_FAILURE_NOTE = (
    "The authorized attempt failed after its online intent. It is retained as "
    "blocker evidence, is never READY, and cannot be rerun at this path."
)
_EXECUTION_FAILURE_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "session_id",
    "session_bundle_id",
    "candidate_bundle_id",
    "policy",
    "phase",
    "trial",
    "operator",
    "setup_hashes",
    "speed_scale",
    "repo_commit",
    "stage",
    "termination_kind",
    "error_type",
    "error_message",
    "recorded_at_utc",
    "hardware_may_have_connected",
    "motion_may_have_started",
    "ready",
    "sealed",
    "rerun_allowed",
    "note",
}


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: str
    phase: str
    session_id: str
    material_root: Path
    bundle: VerifiedBundle | None
    note: str

    def render_text(self) -> str:
        lines = [
            f"Execution status: {self.status}",
            f"Session: {self.session_id}",
            f"Phase: {self.phase}",
            f"Local evidence: {self.material_root}",
        ]
        if self.bundle is not None:
            lines.extend(
                [
                    f"Handoff bundle: {self.bundle.path}",
                    f"Bundle ID: {self.bundle.bundle_id}",
                ]
            )
        lines.append(self.note)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _MaterialRootIdentity:
    """Filesystem identity of the one reserved live-attempt directory."""

    device: int
    inode: int


@contextmanager
def _execution_lease(robot_port: str) -> Iterator[None]:
    """Hold one host-wide, nonblocking lease for the reviewed robot device.

    Linux's abstract Unix-socket namespace has no filesystem entry that another
    process can unlink and recreate while this process owns it.
    """

    if not isinstance(robot_port, str) or not robot_port.startswith("/dev/"):
        raise SafetyGateError("motion permit does not name a reviewed robot device")
    lexical_path = os.path.abspath(os.path.normpath(robot_port))
    resolved_path = os.path.realpath(robot_port)
    robot_identities = {f"path:{lexical_path}", f"path:{resolved_path}"}
    try:
        device = os.stat(robot_port)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise SafetyGateError(f"cannot identify reviewed robot device: {exc}") from exc
    else:
        if stat.S_ISCHR(device.st_mode):
            robot_identities.add(
                f"device:{os.major(device.st_rdev)}:{os.minor(device.st_rdev)}"
            )

    # The path lease is always held, including while the device is unplugged.
    # The device lease additionally makes distinct aliases contend. Acquiring
    # the deterministic set nonblockingly avoids both hot-plug races and
    # lock-order deadlocks.
    leases: list[socket.socket] = []
    try:
        for robot_identity in sorted(robot_identities):
            digest = hashlib.sha256(robot_identity.encode("utf-8")).hexdigest()
            lease_name = f"\0viola-ops-robot-{digest}"
            lease = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                lease.bind(lease_name)
            except OSError as exc:
                lease.close()
                if exc.errno != errno.EADDRINUSE:
                    raise SafetyGateError(
                        f"cannot acquire the execution lease for {robot_port}: {exc}"
                    ) from exc
                raise SafetyGateError(
                    f"another process already holds the execution lease for {robot_port}"
                ) from exc
            leases.append(lease)
        yield
    finally:
        for lease in reversed(leases):
            lease.close()


class _OperatorTerminal:
    """One text interface over separate readable and writable TTY handles."""

    def __init__(self, path: str = "/dev/tty") -> None:
        # Keep input unbuffered so blocking setup reads cannot hide a later
        # STOP line from the control loop's descriptor-level polling.
        self._reader = open(path, "rb", buffering=0)
        try:
            self._writer = open(path, "w", encoding="utf-8", buffering=1)
        except BaseException:
            self._reader.close()
            raise

    def fileno(self) -> int:
        return self._reader.fileno()

    def isatty(self) -> bool:
        return self._reader.isatty() and self._writer.isatty()

    def readline(self) -> str:
        return self._reader.readline().decode("utf-8")

    def discard_pending_input(self) -> None:
        """Drop bytes typed before the next explicit operator prompt."""

        termios.tcflush(self._reader.fileno(), termios.TCIFLUSH)

    def write(self, value: str) -> int:
        return self._writer.write(value)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        try:
            self._reader.close()
        finally:
            self._writer.close()


class InteractiveTrialOperator:
    """TTY-only trial reset, abort, and outcome input."""

    def __init__(self) -> None:
        try:
            self._terminal = _OperatorTerminal()
        except OSError as exc:
            raise SafetyGateError("live execution requires an available /dev/tty") from exc
        if not self._terminal.isatty():
            self._terminal.close()
            raise SafetyGateError("live execution requires an interactive terminal")
        self._pending_event: str | None = None
        self._input_buffer = bytearray()

    def prepare_trial(self, session_id: str, trial_id: str, condition: dict[str, Any]) -> None:
        phrase = f"START {trial_id}"
        # An ARM response and a predicted START line must never be pasted as a
        # single replayable input block. Only input typed after this prompt is
        # eligible to start the trial.
        self._input_buffer.clear()
        self._pending_event = None
        try:
            self._terminal.discard_pending_input()
        except (AttributeError, OSError, termios.error) as exc:
            raise SafetyGateError("cannot clear stale operator input before START") from exc
        self._terminal.write(
            "\nPrepare the reviewed reset and cube condition:\n"
            f"{_render_trial_condition(condition)}\n"
            "Keep ownership of the physical E-stop. During motion, type STOP, "
            "COLLISION, or INTERVENTION followed by Enter.\n"
            f"Type exactly when the workspace is ready: {phrase}\n> "
        )
        self._terminal.flush()
        if self._terminal.readline().rstrip("\r\n") != phrase:
            raise SafetyGateError(f"operator did not start reviewed trial {trial_id}")
        self._pending_event = None

    def abort_requested(self) -> bool:
        self._poll()
        if self._pending_event == "operator_abort":
            self._pending_event = None
            return True
        return False

    def event(self) -> str | None:
        self._poll()
        if self._pending_event in {"collision", "intervention"}:
            value, self._pending_event = self._pending_event, None
            return value
        return None

    def outcome(self, trial_id: str) -> Any:
        from .execution import TrialOutcome

        self._terminal.write(
            "\nRecord the operator-reviewed outcome with one exact line:\n"
            "  RESULT success BLUE <seconds> RED <seconds> COMPLETE <seconds>\n"
            "or\n"
            "  RESULT failure <failure_code>\n"
            "Allowed ordinary failure codes: collision, drop, wrong_order, timeout, "
            "intervention, other\n> "
        )
        self._terminal.flush()
        parts = self._terminal.readline().strip().split()
        if len(parts) == 8 and parts[:2] == ["RESULT", "success"]:
            if parts[2] != "BLUE" or parts[4] != "RED" or parts[6] != "COMPLETE":
                raise ValidationError(f"invalid success annotation for {trial_id}")
            try:
                blue, red, complete = float(parts[3]), float(parts[5]), float(parts[7])
            except ValueError as exc:
                raise ValidationError("trial milestone seconds must be numeric") from exc
            result = TrialOutcome(True, "success", "none", blue, red, complete, 3.0)
        elif len(parts) == 3 and parts[:2] == ["RESULT", "failure"]:
            allowed = {"collision", "drop", "wrong_order", "timeout", "intervention", "other"}
            if parts[2] not in allowed:
                raise ValidationError(f"unsupported ordinary failure code: {parts[2]}")
            result = TrialOutcome(False, "failure", parts[2], None, None, None, 0.0)
        else:
            raise ValidationError(f"invalid operator outcome line for {trial_id}")
        result.validate()
        return result

    def close(self) -> None:
        self._terminal.close()

    def _poll(self) -> None:
        if self._pending_event is not None:
            return
        if self._consume_buffered_line():
            return
        try:
            readable, _, _ = select.select([self._terminal], [], [], 0.0)
        except (OSError, ValueError):
            self._pending_event = "operator_abort"
            return
        if not readable:
            return
        try:
            descriptor = self._terminal.fileno()
            was_blocking = os.get_blocking(descriptor)
            try:
                if was_blocking:
                    os.set_blocking(descriptor, False)
                chunk = os.read(descriptor, 4096)
            finally:
                if was_blocking:
                    os.set_blocking(descriptor, True)
        except (BlockingIOError, OSError, ValueError):
            self._pending_event = "operator_abort"
            return
        if not chunk:
            self._pending_event = "operator_abort"
            return
        self._input_buffer.extend(chunk)
        if len(self._input_buffer) > 4096:
            self._input_buffer.clear()
            self._pending_event = "operator_abort"
            return
        self._consume_buffered_line()

    def _consume_buffered_line(self) -> bool:
        """Consume one complete command without ever waiting for more bytes."""

        try:
            newline = self._input_buffer.index(b"\n")
        except ValueError:
            return False
        raw = bytes(self._input_buffer[:newline])
        del self._input_buffer[: newline + 1]
        try:
            command = raw.decode("utf-8").strip().upper()
        except UnicodeDecodeError:
            command = ""
        self._pending_event = {
            "COLLISION": "collision",
            "INTERVENTION": "intervention",
        }.get(command, "operator_abort")
        return True


def _render_trial_condition(condition: Mapping[str, Any]) -> str:
    """Render the signed cube placement as short operator instructions."""

    condition_id = str(condition.get("condition_id", ""))
    stratum = str(condition.get("stratum", ""))
    lines = [f"  Condition: {condition_id} ({stratum})"]
    for cube in ("blue", "red"):
        axis = condition.get(f"{cube}_axis")
        offset = condition.get(f"{cube}_offset_mm")
        if axis is None:
            lines.append(f"  {cube.title()} cube: nominal reviewed position")
        else:
            lines.append(
                f"  {cube.title()} cube: {float(offset):+g} mm along {axis}"
            )
    return "\n".join(lines)


def execute_command(
    session: Path,
    *,
    candidate: Path,
    phase: str,
    trial: str,
    repo_root: Path,
    handoff_root: Path,
    evidence_root: Path,
    wandb_entity: str,
    wandb_project: str = "starai-viola-policy-benchmark",
    prior_hold_bundle: Path | None = None,
    prior_shakedown_bundle: Path | None = None,
    intent_publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
) -> ExecutionOutcome:
    """Gate, run, record, sync, and seal one physical inference phase."""

    _safe_component(trial, "trial")
    repository = repo_root.resolve()
    output = _external_root(evidence_root, repository, "live evidence_root")
    handoff = _external_root(handoff_root, repository, "live handoff_root")

    # This call performs every trust/revision/E-stop/predecessor/operator check.
    # No policy, robot, camera, serial, or motor module is imported above it.
    gate_request = GateRequest(
        session_bundle=session,
        candidate_bundle=candidate,
        phase=phase,
        trial=trial,
        repository_root=repository,
        handoff_root=handoff,
        prior_hold_bundle=prior_hold_bundle,
        prior_shakedown_bundle=prior_shakedown_bundle,
    )
    permit = authorize_motion(gate_request)
    _safe_component(permit.session_id, "session_id")
    _safe_component(permit.policy, "policy")
    _safe_component(permit.phase, "phase")
    _safe_component(permit.trial, "permit trial")
    if permit.trial != trial:
        raise SafetyGateError("motion permit belongs to another trial")
    with _execution_lease(permit.robot_port):
        return _execute_authorized_command(
            gate_request,
            permit,
            output=output,
            handoff=handoff,
            evidence_root=evidence_root,
            handoff_root=handoff_root,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            intent_publisher=intent_publisher,
        )


def _execute_authorized_command(
    gate_request: GateRequest,
    permit: MotionPermit,
    *,
    output: Path,
    handoff: Path,
    evidence_root: Path,
    handoff_root: Path,
    wandb_entity: str,
    wandb_project: str,
    intent_publisher: Callable[..., WandbRunIdentity],
) -> ExecutionOutcome:
    """Execute while the caller holds the reviewed robot's host-wide lease."""

    repository = gate_request.repository_root
    session = gate_request.session_bundle
    candidate = gate_request.candidate_bundle
    phase = permit.phase
    trial = permit.trial
    prior_hold_bundle = gate_request.prior_hold_bundle
    prior_shakedown_bundle = gate_request.prior_shakedown_bundle
    identity = RuntimeIdentity.capture(role="pc_a", repo_root=repository)

    def capture_current_identity() -> RuntimeIdentity:
        """Recapture the exact reviewed checkout for final evidence boundaries."""

        return RuntimeIdentity.capture(role="pc_a", repo_root=repository)

    # Semantic inspection and a finished online intent run still happen before
    # importing any camera, serial, robot, or motor implementation.
    from .policy_runtime import inspect_candidate

    accepted_session = inspect_bundle(
        session,
        verify_artifacts=True,
        expected_kind="rollout_session",
        required_permission="live_session",
    )
    accepted_candidate = inspect_candidate(candidate)
    hold = _optional_evidence(prior_hold_bundle)
    shakedown = _optional_evidence(prior_shakedown_bundle)
    material = output / permit.session_id / permit.policy / permit.phase / trial
    if not material.is_relative_to(output):
        raise ValidationError("live evidence path escapes the requested evidence_root")
    # The leaf mkdir is the atomic attempt reservation. Existing terminal
    # evidence may take its upload-only recovery path; an empty or partial leaf
    # must never become a second motion attempt.
    try:
        material.mkdir(parents=True)
    except FileExistsError:
        existed = True
        material_identity = _material_root_identity(material)
        if not _material_root_is_stable(
            material, output, repository, material_identity
        ):
            raise ValidationError("existing live attempt path is not a stable directory")
    else:
        existed = False
        material_identity = _material_root_identity(material)
        if not _material_root_is_stable(
            material, output, repository, material_identity
        ):
            raise ValidationError("new live attempt path is not a stable directory")

    def require_material_root(boundary: str) -> None:
        """Refuse to follow a replacement for this attempt directory."""

        if not _material_root_is_stable(
            material, output, repository, material_identity
        ):
            raise SafetyGateError(f"live evidence path changed {boundary}")

    result_path = material / "PHASE_RESULT.json"
    failure_path = material / "EXECUTION_FAILURE.json"
    resume_result = None
    resume_failure = None
    if result_path.is_file():
        resume_result = _load_phase_result(result_path, permit=permit)
    if failure_path.is_file():
        resume_failure = _load_execution_failure(
            failure_path,
            permit=permit,
            identity=identity,
        )
    if resume_result is not None and resume_failure is not None:
        if resume_failure["stage"] != "post_execution_revalidation":
            raise ValidationError("local execution contains both a result and a failure marker")
    elif (
        resume_failure is not None
        and resume_failure["stage"] == "post_execution_revalidation"
    ):
        raise ValidationError(
            "post-execution revalidation failure is missing its preserved phase result"
        )
    elif resume_result is None and resume_failure is None and existed:
        raise ValidationError(
            "an incomplete live attempt is preserved at this evidence path; refusing to "
            "repeat motion or overwrite it"
        )
    require_material_root("while inspecting terminal attempt evidence")
    _publish_execution_intent(
        permit=permit,
        session=accepted_session,
        candidate=accepted_candidate.bundle,
        identity=identity,
        material=material,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        publisher=intent_publisher,
    )
    require_material_root("while publishing the execution intent")

    def revalidate_current_authority(
        runtime_snapshot: Any | None,
        *,
        boundary: str,
        allow_consumed: bool = False,
    ) -> RuntimeIdentity:
        """Reopen every signed input before another trusted boundary."""

        current_identity = RuntimeIdentity.capture(role="pc_a", repo_root=repository)
        current_session, current_candidate = revalidate_motion(
            gate_request,
            permit,
            identity=current_identity,
            allow_consumed=allow_consumed,
        )
        _require_same_bundle(current_session, accepted_session, "rollout session")
        _require_same_bundle(
            current_candidate,
            accepted_candidate.bundle,
            "policy candidate",
        )
        current_material = inspect_candidate(candidate)
        _require_same_bundle(
            current_material.bundle,
            accepted_candidate.bundle,
            "policy candidate runtime material",
        )
        if dict(current_material.runtime_binding) != dict(
            accepted_candidate.runtime_binding
        ):
            raise SafetyGateError(
                f"accepted policy runtime bytes changed {boundary}"
            )
        if runtime_snapshot is not None:
            runtime_snapshot.verify()
        if _external_root(evidence_root, repository, "live evidence_root") != output:
            raise SafetyGateError(f"live evidence_root changed {boundary}")
        if _external_root(handoff_root, repository, "live handoff_root") != handoff:
            raise SafetyGateError(f"live handoff_root changed {boundary}")
        require_material_root(boundary)
        return current_identity

    def record_execution_failure(
        error: BaseException,
        *,
        failure_stage: str,
    ) -> ExecutionOutcome:
        """Retain a terminal, non-READY failure without repeating motion."""

        marker = _execution_failure_payload(
            permit,
            identity=identity,
            stage=failure_stage,
            error=error,
        )
        process_control = not isinstance(error, Exception)
        if not _material_root_is_stable(
            material, output, repository, material_identity
        ):
            # The original attempt may still hold raw evidence, but following a
            # replacement path would put the failure marker in the wrong place.
            raise error.with_traceback(error.__traceback__)
        try:
            marker_path = write_canonical_json(failure_path, marker)
            require_material_root("while recording the execution failure")
            _publish_execution_failure(
                marker=marker,
                marker_path=marker_path,
                permit=permit,
                session=accepted_session,
                candidate=accepted_candidate.bundle,
                identity=identity,
                material=material,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                publisher=intent_publisher,
            )
            require_material_root("while publishing the execution failure")
        except Exception:
            # KeyboardInterrupt, SystemExit, and other process-control
            # exceptions keep their language-level semantics. The immutable
            # local marker remains available for an upload-only retry.
            if process_control:
                raise error.with_traceback(error.__traceback__)
            raise
        if process_control:
            raise error.with_traceback(error.__traceback__)
        return ExecutionOutcome(
            str(marker["status"]),
            permit.phase,
            permit.session_id,
            material,
            None,
            "The failed attempt is retained, is not READY, and will never be rerun in place.",
        )

    if resume_failure is not None:
        require_material_root("before republishing the execution failure")
        _publish_execution_failure(
            marker=resume_failure,
            marker_path=failure_path,
            permit=permit,
            session=accepted_session,
            candidate=accepted_candidate.bundle,
            identity=identity,
            material=material,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            publisher=intent_publisher,
        )
        require_material_root("after republishing the execution failure")
        return ExecutionOutcome(
            str(resume_failure["status"]),
            permit.phase,
            permit.session_id,
            material,
            None,
            "The failed attempt is recorded and cannot be rerun or sealed as READY evidence.",
        )

    # Hardware-capable modules are unreachable until all gates and the online
    # intent receipt above have succeeded.  A completed local result takes the
    # upload-only recovery path and never imports the hardware adapter.
    if resume_result is not None:
        try:
            identity = revalidate_current_authority(
                None,
                boundary="before recovered-result finalization",
            )
        except BaseException as exc:
            return record_execution_failure(
                exc,
                failure_stage="post_execution_revalidation",
            )

        from .evidence import PhaseEvidenceFactory
        from .rollout_evidence import (
            UnsafeTerminalVideoUnavailableError,
            seal_completed_phase,
            seal_unsafe_phase,
        )

        result = resume_result
        factory = (
            None
            if phase == "hold"
            else PhaseEvidenceFactory.reopen_completed(material / "motion_record")
        )
    else:
        stage = "support_import"
        runtime_stack = ExitStack()
        try:
            from .evidence import PhaseEvidenceFactory
            from .policy_runtime import _snapshot_candidate_runtime, load_lerobot_runtime
            from .rollout_evidence import (
                UnsafeTerminalVideoUnavailableError,
                seal_completed_phase,
                seal_unsafe_phase,
            )

            runtime_snapshot = None
            runtime_candidate = accepted_candidate
            if phase != "hold":
                stage = "runtime_snapshot"
                runtime_snapshot = runtime_stack.enter_context(
                    _snapshot_candidate_runtime(accepted_candidate)
                )
                runtime_candidate = runtime_snapshot.candidate

            # The operator may spend time confirming and the online intent may
            # take time to finish.  Reopen all authority and recapture the exact
            # signed runtime at the final boundary before hardware-capable code.
            stage = "final_revalidation"
            final_identity = RuntimeIdentity.capture(role="pc_a", repo_root=repository)
            current_session, current_candidate = revalidate_motion(
                gate_request,
                permit,
                identity=final_identity,
            )
            _require_same_bundle(current_session, accepted_session, "rollout session")
            _require_same_bundle(
                current_candidate,
                accepted_candidate.bundle,
                "policy candidate",
            )
            if _external_root(evidence_root, repository, "live evidence_root") != output:
                raise SafetyGateError("live evidence_root changed before hardware import")
            if _external_root(handoff_root, repository, "live handoff_root") != handoff:
                raise SafetyGateError("live handoff_root changed before hardware import")
            if not _material_root_is_stable(
                material, output, repository, material_identity
            ):
                raise SafetyGateError("live evidence path changed before hardware import")
            identity = final_identity

            stage = "hardware_import"
            from .execution import execute_phase
            from .hardware import SafeViolaRobot, config_from_permit

            factory = (
                None
                if phase == "hold"
                else PhaseEvidenceFactory(material / "motion_record", fps=30, queue_size=64)
            )
            stage = "operator_setup"
            operator = InteractiveTrialOperator()
            try:
                def revalidate_execution_authority(
                    *,
                    failure_stage: str = "execution_revalidation",
                    success_stage: str = "execution",
                    boundary: str = "at an execution boundary",
                    allow_consumed: bool = False,
                ) -> None:
                    """Reopen every live input at one motion boundary."""

                    nonlocal identity, stage
                    stage = failure_stage
                    identity = revalidate_current_authority(
                        runtime_snapshot,
                        boundary=boundary,
                        allow_consumed=allow_consumed,
                    )
                    stage = success_stage

                # Entering the state machine gets a fresh check.  It receives
                # the same callback and repeats the full check immediately
                # after every successful per-trial START.
                stage = "execution_revalidation"
                revalidate_execution_authority()
                stage = "execution"
                result = execute_phase(
                    permit,
                    runtime_candidate,
                    runtime_factory=load_lerobot_runtime,
                    robot_factory=lambda approved: SafeViolaRobot(
                        config_from_permit(approved), approved
                    ),
                    evidence_factory=factory,
                    operator=operator,
                    safety_monitor=operator,
                    revalidate_authority=revalidate_execution_authority,
                )
            except BaseException:
                # Preserve the active failure, especially KeyboardInterrupt or
                # SystemExit, if ordinary terminal cleanup also fails.
                try:
                    operator.close()
                except Exception:
                    pass
                raise
            else:
                operator.close()
            stage = "local_result"
            require_material_root("before preserving the phase result")
            write_canonical_json(result_path, _phase_result_payload(result))
            require_material_root("after preserving the phase result")
            revalidate_execution_authority(
                failure_stage="post_execution_revalidation",
                success_stage="post_execution_revalidation",
                boundary="after hardware disconnect and before finalization",
                allow_consumed=True,
            )
        except BaseException as exc:
            return record_execution_failure(exc, failure_stage=stage)
        finally:
            runtime_stack.close()

    require_material_root("before terminal finalization")
    if result.status == "unsafe_shakedown" and result.phase == "shakedown":
        if factory is None:
            raise ValidationError("unsafe shakedown is missing its motion evidence")
        require_material_root("before unsafe-evidence finalization")
        try:
            sealed = seal_unsafe_phase(
                result,
                session=accepted_session,
                candidate=accepted_candidate,
                identity=identity,
                experiment=accepted_session.manifest["experiment"],
                material_root=material,
                handoff_root=handoff,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                evidence_factory=factory,
                prior_hold=hold,
                prior_shakedown=None,
                publisher=intent_publisher,
                identity_capture=capture_current_identity,
            )
            require_material_root("after unsafe-evidence finalization")
        except UnsafeTerminalVideoUnavailableError:
            require_material_root("before recording the unsealed unsafe outcome")
            outcome = _record_unsealed_unsafe_outcome(
                result=result,
                blocker="repo_b_unsafe_terminal_video_unavailable",
                note=(
                    "The shakedown stopped before a transferable camera frame was retained. "
                    "Raw terminal evidence remains local and non-READY."
                ),
                permit=permit,
                session=accepted_session,
                candidate=accepted_candidate.bundle,
                identity=identity,
                material=material,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                publisher=intent_publisher,
            )
            require_material_root("after recording the unsealed unsafe outcome")
            return outcome
        return ExecutionOutcome(
            result.status,
            result.phase,
            result.session_id,
            material,
            sealed.bundle,
            (
                "Unsafe terminal evidence is sealed for Repo B; it cannot authorize "
                "another motion phase."
            ),
        )

    if result.status == "unsafe_shakedown" and result.phase == "scored":
        require_material_root("before recording the unsealed unsafe outcome")
        outcome = _record_unsealed_unsafe_outcome(
            result=result,
            blocker="repo_b_unsafe_scored_predecessor_schema_mismatch",
            note=(
                "Rich local evidence is retained. No READY bundle was minted because Repo B "
                "requires mutually exclusive compact and rich schemas for the completed "
                "shakedown predecessor of a scored phase."
            ),
            permit=permit,
            session=accepted_session,
            candidate=accepted_candidate.bundle,
            identity=identity,
            material=material,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            publisher=intent_publisher,
        )
        require_material_root("after recording the unsealed unsafe outcome")
        return outcome

    if result.status != "completed":
        raise ValidationError(f"unsupported physical phase result status: {result.status}")

    require_material_root("before completed-evidence finalization")
    sealed = seal_completed_phase(
        result,
        session=accepted_session,
        candidate=accepted_candidate,
        identity=identity,
        experiment=accepted_session.manifest["experiment"],
        material_root=material,
        handoff_root=handoff,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        evidence_factory=factory,
        prior_hold=hold,
        prior_shakedown=shakedown,
        identity_capture=capture_current_identity,
    )
    require_material_root("after completed-evidence finalization")
    return ExecutionOutcome(
        "completed",
        result.phase,
        result.session_id,
        material,
        sealed.bundle,
        "Completed evidence is sealed; Repo B must accept and validate it before the next phase.",
    )


def _publish_execution_intent(
    *,
    permit: MotionPermit,
    session: VerifiedBundle,
    candidate: VerifiedBundle,
    identity: RuntimeIdentity,
    material: Path,
    wandb_entity: str,
    wandb_project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> Path:
    """Finish and record the online pre-hardware intent run."""

    binding: dict[str, Any] = {
        "session_id": permit.session_id,
        "session_bundle_id": session.bundle_id,
        "candidate_bundle_id": candidate.bundle_id,
        "policy": permit.policy,
        "phase": permit.phase,
        "trial": permit.trial,
        "operator": permit.operator,
        "repo_commit": identity.repository_commit,
        "setup_hashes": dict(permit.setup_hashes),
    }
    run = planned_run(
        wandb_entity,
        wandb_project,
        f"intent-{permit.policy}-{permit.phase}-{sha256_json(binding)[:12]}",
    )
    receipt_path = material / "EXECUTION_INTENT_WANDB_SYNCED.json"
    if receipt_path.is_file():
        existing = read_json_object(receipt_path, label="execution intent receipt")
        if receipt_path.read_bytes() != canonical_json_bytes(existing):
            raise ValidationError("existing execution intent receipt is not canonical JSON")
        if (
            existing.get("schema_version") != 1
            or existing.get("operation") != "policy_execution_intent"
            or existing.get("status") != "operator_authorized_pre_hardware"
            or existing.get("binding") != binding
            or existing.get("wandb") != run.binding()
        ):
            raise ValidationError("existing execution intent receipt differs from this request")
        # A local receipt is only a binding, not proof that its remote run is
        # still present and finished.  Re-open the stable W&B identity before
        # any recovery finalization can proceed.
        publisher(
            run,
            job_type="viola-policy-execution-intent",
            config={"operation": "policy_execution_intent", **binding},
            summary={
                "operator_authorized": True,
                "hardware_imported": False,
                "hardware_connected": False,
            },
        )
        return receipt_path
    publisher(
        run,
        job_type="viola-policy-execution-intent",
        config={"operation": "policy_execution_intent", **binding},
        summary={
            "operator_authorized": True,
            "hardware_imported": False,
            "hardware_connected": False,
        },
    )
    receipt: Mapping[str, Any] = {
        "schema_version": 1,
        "operation": "policy_execution_intent",
        "status": "operator_authorized_pre_hardware",
        "binding": binding,
        "wandb": run.binding(),
    }
    return write_canonical_json(receipt_path, receipt)


def _record_unsealed_unsafe_outcome(
    *,
    result: PhaseResult,
    blocker: str,
    note: str,
    permit: MotionPermit,
    session: VerifiedBundle,
    candidate: VerifiedBundle,
    identity: RuntimeIdentity,
    material: Path,
    wandb_entity: str,
    wandb_project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> ExecutionOutcome:
    """Persist and publish a typed terminal that cannot form a Repo-B bundle."""

    marker = {
        "schema_version": 1,
        "status": result.status,
        "session_id": result.session_id,
        "policy": result.policy,
        "phase": result.phase,
        "terminal_event": result.terminal_event,
        "terminal_reason": result.terminal_reason,
        "ready": False,
        "blocker": blocker,
        "note": note,
    }
    marker_path = write_canonical_json(material / "UNSAFE_NOT_SEALED.json", marker)
    _publish_unsafe_outcome(
        marker=marker,
        marker_path=marker_path,
        permit=permit,
        session=session,
        candidate=candidate,
        identity=identity,
        material=material,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        publisher=publisher,
    )
    return ExecutionOutcome(
        result.status,
        result.phase,
        result.session_id,
        material,
        None,
        note,
    )


def _publish_unsafe_outcome(
    *,
    marker: Mapping[str, Any],
    marker_path: Path,
    permit: MotionPermit,
    session: VerifiedBundle,
    candidate: VerifiedBundle,
    identity: RuntimeIdentity,
    material: Path,
    wandb_entity: str,
    wandb_project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> Path:
    """Record an unsafe terminal online without granting a READY bundle."""

    material_path = Path(material).resolve()
    expected_marker_path = material_path / "UNSAFE_NOT_SEALED.json"
    if Path(marker_path).resolve() != expected_marker_path:
        raise ValidationError("unsafe marker is outside its authorized material path")
    stored = read_json_object(expected_marker_path, label="unsafe terminal marker")
    if expected_marker_path.read_bytes() != canonical_json_bytes(stored):
        raise ValidationError("unsafe terminal marker is not canonical JSON")
    if stored != dict(marker):
        raise ValidationError("unsafe terminal marker differs from the supplied outcome")
    if session.bundle_id != permit.session_bundle_id:
        raise ValidationError("unsafe outcome session bundle differs from the motion permit")
    if candidate.bundle_id != permit.candidate_bundle_id:
        raise ValidationError("unsafe outcome candidate bundle differs from the motion permit")

    binding = {
        "session_id": permit.session_id,
        "session_bundle_id": session.bundle_id,
        "candidate_bundle_id": candidate.bundle_id,
        "policy": permit.policy,
        "phase": permit.phase,
        "trial": permit.trial,
        "repo_commit": identity.repository_commit,
        "status": stored["status"],
        "terminal_event": stored["terminal_event"],
        "marker_sha256": sha256_json(stored),
    }
    run = planned_run(
        wandb_entity,
        wandb_project,
        f"unsafe-{permit.policy}-{permit.phase}-{sha256_json(binding)[:12]}",
    )
    receipt_path = material / "UNSAFE_WANDB_SYNCED.json"
    if receipt_path.is_file():
        existing = read_json_object(receipt_path, label="unsafe W&B receipt")
        expected = {
            "schema_version": 1,
            "operation": "policy_unsafe_outcome",
            "status": "recorded_not_sealed",
            "evidence_file": marker_path.name,
            "binding": binding,
            "wandb": run.binding(),
        }
        if receipt_path.read_bytes() != canonical_json_bytes(existing):
            raise ValidationError("existing unsafe W&B receipt is not canonical JSON")
        if existing != expected:
            raise ValidationError("existing unsafe W&B receipt differs from this outcome")
        publisher(
            run,
            job_type="viola-policy-unsafe-outcome",
            config={"operation": "policy_unsafe_outcome", **binding},
            summary={"ready": False, "motion_stopped": True, "terminal_recorded": True},
        )
        return receipt_path
    publisher(
        run,
        job_type="viola-policy-unsafe-outcome",
        config={"operation": "policy_unsafe_outcome", **binding},
        summary={"ready": False, "motion_stopped": True, "terminal_recorded": True},
    )
    receipt = {
        "schema_version": 1,
        "operation": "policy_unsafe_outcome",
        "status": "recorded_not_sealed",
        "evidence_file": marker_path.name,
        "binding": binding,
        "wandb": run.binding(),
    }
    return write_canonical_json(receipt_path, receipt)


def _execution_failure_payload(
    permit: MotionPermit,
    *,
    identity: RuntimeIdentity,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    """Describe one terminal execution failure without granting readiness."""

    hardware_may_have_connected = stage in {
        "execution_revalidation",
        "execution",
        "local_result",
        "post_execution_revalidation",
    }
    marker = {
        "schema_version": 1,
        "kind": "viola_execution_failure",
        "status": "execution_failed_not_ready",
        "session_id": permit.session_id,
        "session_bundle_id": permit.session_bundle_id,
        "candidate_bundle_id": permit.candidate_bundle_id,
        "policy": permit.policy,
        "phase": permit.phase,
        "trial": permit.trial,
        "operator": permit.operator,
        "setup_hashes": dict(permit.setup_hashes),
        "speed_scale": permit.speed_scale,
        "repo_commit": _failure_identity_commit(identity),
        "stage": stage,
        "termination_kind": _termination_kind(error),
        "error_type": _failure_type_name(error),
        "error_message": _failure_message(error),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "hardware_may_have_connected": hardware_may_have_connected,
        "motion_may_have_started": hardware_may_have_connected,
        "ready": False,
        "sealed": False,
        "rerun_allowed": False,
        "note": _EXECUTION_FAILURE_NOTE,
    }
    _validate_execution_failure(marker, permit=permit, identity=identity)
    return marker


def _load_execution_failure(
    path: Path,
    *,
    permit: MotionPermit,
    identity: RuntimeIdentity,
) -> dict[str, Any]:
    """Load a canonical terminal marker without entering the hardware path."""

    source = Path(path)
    if source.name != "EXECUTION_FAILURE.json":
        raise ValidationError("execution failure marker must be named EXECUTION_FAILURE.json")
    _validate_failure_material(source.parent, permit)
    value = read_json_object(source, label="execution failure marker")
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read execution failure marker {source}: {exc}") from exc
    if raw != canonical_json_bytes(value):
        raise ValidationError("execution failure marker is not canonical JSON")
    _validate_execution_failure(value, permit=permit, identity=identity)
    return value


def _publish_execution_failure(
    *,
    marker: Mapping[str, Any],
    marker_path: Path,
    permit: MotionPermit,
    session: VerifiedBundle,
    candidate: VerifiedBundle,
    identity: RuntimeIdentity,
    material: Path,
    wandb_entity: str,
    wandb_project: str,
    publisher: Callable[..., WandbRunIdentity],
) -> Path:
    """Publish one stable failure identity and retain it as non-READY evidence."""

    material_path = _validate_failure_material(material, permit)
    expected_marker_path = material_path / "EXECUTION_FAILURE.json"
    if Path(marker_path).resolve() != expected_marker_path:
        raise ValidationError("execution failure marker is outside its authorized material path")
    stored = _load_execution_failure(expected_marker_path, permit=permit, identity=identity)
    if dict(marker) != stored:
        raise ValidationError("execution failure marker bytes differ from the supplied failure")
    if session.bundle_id != permit.session_bundle_id:
        raise ValidationError("execution failure session bundle differs from the motion permit")
    if candidate.bundle_id != permit.candidate_bundle_id:
        raise ValidationError("execution failure candidate bundle differs from the motion permit")

    marker_sha256 = sha256_json(stored)
    binding: dict[str, Any] = {
        "session_id": permit.session_id,
        "session_bundle_id": permit.session_bundle_id,
        "candidate_bundle_id": permit.candidate_bundle_id,
        "policy": permit.policy,
        "phase": permit.phase,
        "trial": permit.trial,
        "repo_commit": _failure_identity_commit(identity),
        "stage": stored["stage"],
        "termination_kind": stored["termination_kind"],
        "status": stored["status"],
        "marker_sha256": marker_sha256,
    }
    run = planned_run(
        wandb_entity,
        wandb_project,
        f"failure-{permit.policy}-{permit.phase}-{marker_sha256[:12]}",
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "operation": "policy_execution_failure",
        "status": "recorded_not_ready",
        "evidence_file": "EXECUTION_FAILURE.json",
        "binding": binding,
        "wandb": run.binding(),
    }
    receipt_path = material_path / "EXECUTION_FAILURE_WANDB_SYNCED.json"
    if receipt_path.is_file():
        existing = read_json_object(receipt_path, label="execution failure W&B receipt")
        try:
            receipt_bytes = receipt_path.read_bytes()
        except OSError as exc:
            raise ValidationError(
                f"cannot read execution failure W&B receipt {receipt_path}: {exc}"
            ) from exc
        if receipt_bytes != canonical_json_bytes(existing):
            raise ValidationError("execution failure W&B receipt is not canonical JSON")
        if existing != receipt:
            raise ValidationError(
                "existing execution failure W&B receipt differs from this failure"
            )

    # Re-opening the same deterministic run on retry verifies that the remote
    # finished receipt still exists.  No hardware-capable module is involved.
    publisher(
        run,
        job_type="viola-policy-execution-failure",
        config={"operation": "policy_execution_failure", **binding},
        summary={
            "ready": False,
            "sealed": False,
            "rerun_allowed": False,
            "hardware_may_have_connected": stored["hardware_may_have_connected"],
            "motion_may_have_started": stored["motion_may_have_started"],
        },
    )
    return write_canonical_json(receipt_path, receipt)


def _validate_execution_failure(
    value: Mapping[str, Any],
    *,
    permit: MotionPermit,
    identity: RuntimeIdentity,
) -> None:
    """Require a failure marker to match its one reviewed permit and revision."""

    marker = require_exact_keys(value, _EXECUTION_FAILURE_FIELDS, label="execution failure marker")
    if type(marker["schema_version"]) is not int or marker["schema_version"] != 1:
        raise ValidationError("execution failure marker has an unsupported schema version")
    if marker["kind"] != "viola_execution_failure":
        raise ValidationError("execution failure marker has an unexpected kind")
    if marker["status"] != "execution_failed_not_ready":
        raise ValidationError("execution failure marker has an unexpected status")

    for field, expected in (
        ("session_id", permit.session_id),
        ("session_bundle_id", permit.session_bundle_id),
        ("candidate_bundle_id", permit.candidate_bundle_id),
        ("policy", permit.policy),
        ("phase", permit.phase),
        ("trial", permit.trial),
        ("operator", permit.operator),
    ):
        if marker[field] != expected:
            raise ValidationError(
                f"execution failure marker {field} differs from the motion permit"
            )

    for field in ("session_id", "policy", "phase", "trial"):
        _safe_component(marker[field], f"execution failure {field}")
    for field in ("session_bundle_id", "candidate_bundle_id"):
        if not isinstance(marker[field], str) or not _SHA256.fullmatch(marker[field]):
            raise ValidationError(f"execution failure marker {field} must be a SHA-256 digest")
    if not isinstance(marker["operator"], str) or not marker["operator"].strip():
        raise ValidationError("execution failure marker operator must be nonempty")

    setup_hashes = require_exact_keys(
        marker["setup_hashes"],
        {"calibration", "camera", "robot", "reset"},
        label="execution failure setup hashes",
    )
    permit_hashes = require_exact_keys(
        permit.setup_hashes,
        {"calibration", "camera", "robot", "reset"},
        label="motion permit setup hashes",
    )
    invalid_hash = any(
        not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        for digest in setup_hashes.values()
    )
    if invalid_hash:
        raise ValidationError("execution failure setup hashes must be SHA-256 digests")
    if dict(setup_hashes) != dict(permit_hashes):
        raise ValidationError("execution failure setup hashes differ from the motion permit")

    speed_scale = _nonnegative_finite(marker["speed_scale"], "execution failure speed scale")
    permit_speed = _nonnegative_finite(permit.speed_scale, "motion permit speed scale")
    if speed_scale != permit_speed:
        raise ValidationError("execution failure speed scale differs from the motion permit")
    if marker["repo_commit"] != _failure_identity_commit(identity):
        raise ValidationError("execution failure marker belongs to another repository revision")
    if marker["stage"] not in _EXECUTION_FAILURE_STAGES:
        raise ValidationError("execution failure marker has an unknown execution stage")

    termination_kind = marker["termination_kind"]
    if termination_kind not in {
        "exception",
        "keyboard_interrupt",
        "system_exit",
        "base_exception",
    }:
        raise ValidationError("execution failure marker has an unknown termination kind")
    error_type = marker["error_type"]
    if not isinstance(error_type, str) or not _ERROR_TYPE.fullmatch(error_type):
        raise ValidationError("execution failure marker has an invalid error type")
    if (termination_kind == "keyboard_interrupt") != (error_type == "KeyboardInterrupt"):
        raise ValidationError("execution failure marker has inconsistent interrupt fields")
    if (termination_kind == "system_exit") != (error_type == "SystemExit"):
        raise ValidationError("execution failure marker has inconsistent system-exit fields")
    message = marker["error_message"]
    if not isinstance(message, str) or message != _normalize_failure_message(message):
        raise ValidationError("execution failure marker has an invalid error message")

    recorded_at = _utc_datetime(marker["recorded_at_utc"], "execution failure recorded_at_utc")
    if recorded_at.isoformat() != marker["recorded_at_utc"]:
        raise ValidationError("execution failure timestamp must use canonical UTC ISO format")
    hardware_expected = marker["stage"] in {
        "execution_revalidation",
        "execution",
        "local_result",
        "post_execution_revalidation",
    }
    if marker["hardware_may_have_connected"] is not hardware_expected:
        raise ValidationError("execution failure hardware exposure differs from its stage")
    if marker["motion_may_have_started"] is not hardware_expected:
        raise ValidationError("execution failure motion exposure differs from its stage")
    if (
        marker["ready"] is not False
        or marker["sealed"] is not False
        or marker["rerun_allowed"] is not False
    ):
        raise ValidationError("execution failure marker must remain unsealed and not READY")
    if marker["note"] != _EXECUTION_FAILURE_NOTE:
        raise ValidationError("execution failure marker note differs from the terminal contract")


def _validate_failure_material(material: Path, permit: MotionPermit) -> Path:
    """Bind a failure artifact to session/policy/phase/trial path components."""

    expected = (
        _safe_component(permit.session_id, "failure session_id"),
        _safe_component(permit.policy, "failure policy"),
        _safe_component(permit.phase, "failure phase"),
        _safe_component(permit.trial, "failure trial"),
    )
    resolved = Path(material).resolve()
    if len(resolved.parts) < len(expected) or tuple(resolved.parts[-4:]) != expected:
        raise ValidationError("execution failure material path differs from the motion permit")
    return resolved


def _failure_identity_commit(identity: RuntimeIdentity) -> str:
    """Return the clean PC-A revision used for an execution failure."""

    if getattr(identity, "role", None) != "pc_a":
        raise ValidationError("execution failure evidence requires the pc_a runtime identity")
    if getattr(identity, "repository_clean", None) is not True:
        raise ValidationError("execution failure evidence requires a clean repository identity")
    commit = getattr(identity, "repository_commit", None)
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValidationError("execution failure evidence requires a full repository commit")
    return commit


def _termination_kind(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "keyboard_interrupt"
    if isinstance(error, SystemExit):
        return "system_exit"
    if isinstance(error, Exception):
        return "exception"
    return "base_exception"


def _failure_type_name(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "KeyboardInterrupt"
    if isinstance(error, SystemExit):
        return "SystemExit"
    name = type(error).__name__
    if _ERROR_TYPE.fullmatch(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)[:128]
    if not cleaned or not re.match(r"[A-Za-z_]", cleaned):
        cleaned = f"Failure_{cleaned}"[:128]
    return cleaned


def _failure_message(error: BaseException) -> str:
    try:
        message = str(error)
    except BaseException:
        message = "The failure did not provide a printable message."
    return _normalize_failure_message(message)


def _normalize_failure_message(message: str) -> str:
    valid_unicode = message.encode("utf-8", errors="replace").decode("utf-8")
    printable = "".join(
        character if character.isprintable() else " " for character in valid_unicode
    )
    normalized = " ".join(printable.split())[:2048].rstrip()
    return normalized or "The failure did not provide a message."


def _phase_result_payload(result: Any) -> dict[str, Any]:
    """Freeze the completed local control result before any network finalization."""

    return {
        "schema_version": 1,
        "session_id": result.session_id,
        "policy": result.policy,
        "phase": result.phase,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "speed_scale": result.speed_scale,
        "held_action": None if result.held_action is None else dict(result.held_action),
        "terminal_event": result.terminal_event,
        "terminal_reason": result.terminal_reason,
        "trials": [
            {
                "trial_id": item.trial_id,
                "index": item.index,
                "condition": dict(item.condition),
                "started_at": item.started_at,
                "completed_at": item.completed_at,
                "duration_sec": item.duration_sec,
                "actions": item.actions,
                "replans": item.replans,
                "inference_latency_ms": list(item.inference_latency_ms),
                "control_latency_ms": list(item.control_latency_ms),
                "outcome": {
                    "success": item.outcome.success,
                    "outcome": item.outcome.outcome,
                    "failure_code": item.outcome.failure_code,
                    "blue_completed_sec": item.outcome.blue_completed_sec,
                    "red_completed_sec": item.outcome.red_completed_sec,
                    "completion_time_sec": item.outcome.completion_time_sec,
                    "stable_duration_sec": item.outcome.stable_duration_sec,
                },
                "safety_events": list(item.safety_events),
                "trace_path": item.trace_path,
                "front_video_path": item.front_video_path,
                "up_video_path": item.up_video_path,
            }
            for item in result.trials
        ],
    }


def _load_phase_result(path: Path, *, permit: MotionPermit) -> Any:
    """Load immutable local results for finalization without repeating motion."""

    from .execution import (
        ACTION_KEYS,
        SAFETY_EVENTS,
        PhaseResult,
        TrialOutcome,
        TrialResult,
        canonical_scored_conditions,
        shakedown_conditions,
    )

    raw = path.read_bytes()
    value = read_json_object(path, label="local phase result")
    if raw != canonical_json_bytes(value):
        raise ValidationError("local phase result is not canonical JSON")
    require_exact_keys(
        value,
        {
            "schema_version",
            "session_id",
            "policy",
            "phase",
            "started_at",
            "completed_at",
            "speed_scale",
            "held_action",
            "terminal_event",
            "terminal_reason",
            "trials",
        },
        label="local phase result",
    )
    if (
        value["schema_version"] != 1
        or value["session_id"] != permit.session_id
        or value["policy"] != permit.policy
        or value["phase"] != permit.phase
        or value["speed_scale"] != permit.speed_scale
        or not isinstance(value["trials"], list)
    ):
        raise ValidationError("local phase result differs from the current motion permit")
    _ordered_utc(value["started_at"], value["completed_at"], label="local phase")
    terminal_event = value["terminal_event"]
    terminal_reason = value["terminal_reason"]
    if terminal_event is None:
        if terminal_reason is not None:
            raise ValidationError("completed local phase cannot contain a terminal reason")
    elif (
        terminal_event not in SAFETY_EVENTS
        or not isinstance(terminal_reason, str)
        or not terminal_reason.strip()
    ):
        raise ValidationError("unsafe local phase requires a known event and reason")

    if permit.phase == "hold":
        if terminal_event is not None or value["trials"]:
            raise ValidationError("local hold result must be completed without trials")
        held_action = _phase_action(value["held_action"], label="local held action")
        if set(permit.absolute_limits) != {key.removesuffix(".pos") for key in ACTION_KEYS}:
            raise ValidationError("motion permit lacks exact seven-joint absolute limits")
        for key in ACTION_KEYS:
            joint = key.removesuffix(".pos")
            lower, upper = permit.absolute_limits[joint]
            if not lower <= held_action[key] <= upper:
                raise ValidationError(f"local held action {key} is outside reviewed limits")
    else:
        if value["held_action"] is not None:
            raise ValidationError("local motion result contains an unexpected hold action")
        held_action = None

    expected_conditions = (
        shakedown_conditions() if permit.phase == "shakedown" else canonical_scored_conditions()
    )
    expected_count = len(expected_conditions)
    actual_count = len(value["trials"])
    if permit.phase != "hold":
        if terminal_event is None and actual_count != expected_count:
            raise ValidationError(
                f"completed {permit.phase} requires exactly {expected_count} trials"
            )
        if terminal_event is not None and not 1 <= actual_count <= expected_count:
            raise ValidationError(
                f"unsafe {permit.phase} must retain one through {expected_count} trials"
            )

    trials = []
    trial_fields = {
        "trial_id",
        "index",
        "condition",
        "started_at",
        "completed_at",
        "duration_sec",
        "actions",
        "replans",
        "inference_latency_ms",
        "control_latency_ms",
        "outcome",
        "safety_events",
        "trace_path",
        "front_video_path",
        "up_video_path",
    }
    outcome_fields = {
        "success",
        "outcome",
        "failure_code",
        "blue_completed_sec",
        "red_completed_sec",
        "completion_time_sec",
        "stable_duration_sec",
    }
    for index, item in enumerate(value["trials"]):
        require_exact_keys(item, trial_fields, label=f"local trial {index}")
        outcome_value = require_exact_keys(
            item["outcome"], outcome_fields, label=f"local trial {index} outcome"
        )
        expected_trial_id = f"{permit.session_id}-{permit.phase}-{index + 1:02d}"
        if item["trial_id"] != expected_trial_id or item["index"] != index:
            raise ValidationError(f"local trial {index} has a reordered identity")
        expected_condition = expected_conditions[index].to_dict()
        if item["condition"] != expected_condition:
            raise ValidationError(f"local trial {index} differs from the signed schedule")
        _ordered_utc(
            item["started_at"], item["completed_at"], label=f"local trial {index}"
        )
        duration = _nonnegative_finite(item["duration_sec"], f"local trial {index} duration")
        if duration > 61.0:
            raise ValidationError(f"local trial {index} exceeds the 60-second window")
        actions = _bounded_integer(
            item["actions"], minimum=0, maximum=1800, label=f"local trial {index} actions"
        )
        replans = _bounded_integer(
            item["replans"], minimum=0, maximum=180, label=f"local trial {index} replans"
        )
        if replans != (actions + 9) // 10:
            raise ValidationError(f"local trial {index} has an inconsistent ten-action queue")
        inference = _latency_samples(
            item["inference_latency_ms"], actions, label=f"local trial {index} inference"
        )
        control = _latency_samples(
            item["control_latency_ms"], actions, label=f"local trial {index} control"
        )
        safety_events = item["safety_events"]
        if not isinstance(safety_events, list) or any(
            event not in SAFETY_EVENTS for event in safety_events
        ):
            raise ValidationError(f"local trial {index} has invalid safety events")
        if index < actual_count - 1 and safety_events:
            raise ValidationError("only the terminal trial may contain a safety event")
        if terminal_event is None:
            if safety_events or actions != 1800 or not 59.0 <= duration <= 61.0:
                raise ValidationError(
                    f"completed local trial {index} lacks its full safe 60-second trace"
                )
        elif index == actual_count - 1 and safety_events != [terminal_event]:
            raise ValidationError("terminal trial does not match the phase safety event")

        if not isinstance(outcome_value["success"], bool):
            raise ValidationError(f"local trial {index} outcome success must be boolean")
        for name in ("outcome", "failure_code"):
            if not isinstance(outcome_value[name], str) or not outcome_value[name]:
                raise ValidationError(f"local trial {index} outcome {name} must be nonempty")
        outcome = TrialOutcome(
            success=outcome_value["success"],
            outcome=outcome_value["outcome"],
            failure_code=outcome_value["failure_code"],
            blue_completed_sec=_optional_finite(
                outcome_value["blue_completed_sec"], f"local trial {index} blue milestone"
            ),
            red_completed_sec=_optional_finite(
                outcome_value["red_completed_sec"], f"local trial {index} red milestone"
            ),
            completion_time_sec=_optional_finite(
                outcome_value["completion_time_sec"],
                f"local trial {index} completion milestone",
            ),
            stable_duration_sec=_nonnegative_finite(
                outcome_value["stable_duration_sec"],
                f"local trial {index} stable duration",
            ),
        )
        outcome.validate()
        if safety_events and (
            outcome.success
            or outcome.outcome != "safety_abort"
            or outcome.failure_code != terminal_event
        ):
            raise ValidationError("terminal trial outcome differs from its safety event")
        expected_trace = f"trials/trial-{index:02d}.jsonl"
        expected_front = f"videos/trial-{index:02d}-front.mp4"
        expected_up = f"videos/trial-{index:02d}-up.mp4"
        if (
            item["trace_path"] != expected_trace
            or item["front_video_path"] != expected_front
            or item["up_video_path"] != expected_up
        ):
            raise ValidationError(f"local trial {index} evidence paths are not canonical")
        trials.append(
            TrialResult(
                trial_id=item["trial_id"],
                index=index,
                condition=expected_condition,
                started_at=item["started_at"],
                completed_at=item["completed_at"],
                duration_sec=duration,
                actions=actions,
                replans=replans,
                inference_latency_ms=inference,
                control_latency_ms=control,
                outcome=outcome,
                safety_events=tuple(safety_events),
                trace_path=expected_trace,
                front_video_path=expected_front,
                up_video_path=expected_up,
            )
        )
    result = PhaseResult(
        session_id=value["session_id"],
        policy=value["policy"],
        phase=value["phase"],
        started_at=value["started_at"],
        completed_at=value["completed_at"],
        speed_scale=_nonnegative_finite(value["speed_scale"], "local phase speed scale"),
        held_action=held_action,
        trials=tuple(trials),
        terminal_event=terminal_event,
        terminal_reason=terminal_reason,
    )
    return result


def _phase_action(value: Any, *, label: str) -> dict[str, float]:
    from .execution import ACTION_KEYS

    if not isinstance(value, Mapping) or set(value) != set(ACTION_KEYS):
        raise ValidationError(f"{label} must name the exact seven joints")
    return {key: _finite_number(value[key], f"{label} {key}") for key in ACTION_KEYS}


def _latency_samples(value: Any, count: int, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValidationError(f"{label} samples must match the sent-action count")
    return tuple(_nonnegative_finite(item, f"{label} sample") for item in value)


def _bounded_integer(value: Any, *, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValidationError(f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite_number(value, label)


def _nonnegative_finite(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0:
        raise ValidationError(f"{label} must be nonnegative")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def _ordered_utc(started: Any, completed: Any, *, label: str) -> None:
    start = _utc_datetime(started, f"{label} started_at")
    finish = _utc_datetime(completed, f"{label} completed_at")
    if finish < start:
        raise ValidationError(f"{label} completed before it started")


def _utc_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValidationError(f"{label} must use UTC")
    return parsed


def _safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _PATH_COMPONENT.fullmatch(value):
        raise ValidationError(
            f"{label} must be one path-safe component using letters, numbers, '.', '_', or '-'"
        )
    return value


def _external_root(path: Path, repository: Path, label: str) -> Path:
    """Return a nonsymlink write root disjoint from the reviewed checkout."""

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValidationError(f"cannot inspect {label} path {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink path is forbidden for {label}: {current}")
        if not stat.S_ISDIR(mode):
            raise ValidationError(f"{label} component is not a directory: {current}")
    checkout = repository.resolve()
    if (
        root == checkout
        or root.is_relative_to(checkout)
        or checkout.is_relative_to(root)
    ):
        raise ValidationError(
            f"{label} must be outside the Git worktree, and the root cannot contain it"
        )
    return root


def _require_same_bundle(current: Any, original: Any, label: str) -> None:
    """Require final inspection to recover the exact content-addressed bundle."""

    if (
        getattr(current, "bundle_id", None) != getattr(original, "bundle_id", None)
        or getattr(current, "content_id", None) != getattr(original, "content_id", None)
    ):
        raise SafetyGateError(f"revalidated {label} differs from the authorized bundle")


def _material_root_identity(material: Path) -> _MaterialRootIdentity:
    """Capture the directory entry reserved for one live attempt."""

    try:
        details = material.lstat()
    except OSError as exc:
        raise ValidationError(f"cannot inspect live attempt directory: {exc}") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ValidationError("live attempt path is not a directory")
    return _MaterialRootIdentity(device=details.st_dev, inode=details.st_ino)


def _material_root_is_stable(
    material: Path,
    output: Path,
    repository: Path,
    expected_identity: _MaterialRootIdentity | None = None,
) -> bool:
    """Return whether the approved path still names the reserved directory."""

    try:
        current_output = output.resolve()
        current_material = material.resolve()
        checkout = repository.resolve()
        current_identity = _material_root_identity(material)
    except OSError:
        return False
    except ValidationError:
        return False
    return (
        current_output == output
        and current_material == material
        and material.is_relative_to(output)
        and output != checkout
        and not output.is_relative_to(checkout)
        and not checkout.is_relative_to(output)
        and (
            expected_identity is None
            or current_identity == expected_identity
        )
    )


def _optional_evidence(path: Path | None) -> VerifiedBundle | None:
    if path is None:
        return None
    return inspect_bundle(
        path,
        verify_artifacts=True,
        expected_kind="rollout_evidence",
        required_permission="evidence_only",
    )


__all__ = ["ExecutionOutcome", "InteractiveTrialOperator", "execute_command"]
