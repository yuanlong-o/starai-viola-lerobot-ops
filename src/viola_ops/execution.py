"""The small, fail-closed state machine used for live Viola inference.

This file is intentionally linear.  The motion gate creates a :class:`MotionPermit`;
only then are the policy and robot factories called.  Every control iteration
checks observation freshness, the deadline, the proposed seven-axis action,
and evidence capacity before the public robot adapter may write once.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from .errors import SafetyGateError, ValidationError
from .policies import CANONICAL_TASK, JOINT_NAMES
from .policy_runtime import AcceptedPolicyCandidate, PolicyRuntime, finite_action, prepare_observation
from .safety import (
    MotionPermit,
    assert_permit_current,
    begin_permit_execution,
    finish_permit_execution,
    validate_action,
)

ACTION_KEYS = tuple(f"{joint}.pos" for joint in JOINT_NAMES)
SAFETY_EVENTS = (
    "malformed_action",
    "clamped_action",
    "stale_observation",
    "deadline_miss",
    "feedback_loss",
    "collision",
    "intervention",
    "safety_abort",
    "operator_abort",
)
TARGET_HZ = 30.0
TARGET_PERIOD_S = 1.0 / TARGET_HZ
RATE_TOLERANCE_MS = 5.0
CONTROL_DEADLINE_MS = 300.0
OBSERVATION_DEADLINE_MS = 100.0
TRIAL_DURATION_S = 60.0
STABLE_SUCCESS_S = 3.0


class RolloutRobot(Protocol):
    """Methods supplied by the reviewed public-API robot adapter."""

    last_receipt: Any

    def connect(self, calibrate: bool = False) -> None: ...

    def disconnect(self) -> None: ...

    def capture_observation(self) -> Any: ...

    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...


class EvidenceReservation(Protocol):
    """A capacity slot acquired before a motor write."""

    def commit(self, row: Mapping[str, Any]) -> None: ...

    def cancel(self) -> None: ...


class TrialEvidence(Protocol):
    """Bounded, nonblocking handoff to a recorder thread."""

    def reserve(self, *, front: Any, up: Any) -> EvidenceReservation | None: ...

    def record_terminal(self, row: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class EvidenceFactory(Protocol):
    def start_trial(self, trial_id: str) -> TrialEvidence: ...


class TrialOperator(Protocol):
    """Operator-owned reset, abort, and post-trial annotation surface."""

    def prepare_trial(self, session_id: str, trial_id: str, condition: Mapping[str, Any]) -> None: ...

    def abort_requested(self) -> bool: ...

    def outcome(self, trial_id: str) -> "TrialOutcome": ...


class SafetyMonitor(Protocol):
    """Optional nonblocking collision/intervention signal."""

    def event(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class TrialCondition:
    condition_id: str
    stratum: str
    blue_axis: str | None = None
    blue_offset_mm: float = 0.0
    red_axis: str | None = None
    red_offset_mm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "stratum": self.stratum,
            "blue_axis": self.blue_axis,
            "blue_offset_mm": self.blue_offset_mm,
            "red_axis": self.red_axis,
            "red_offset_mm": self.red_offset_mm,
        }


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    success: bool
    outcome: str
    failure_code: str
    blue_completed_sec: float | None
    red_completed_sec: float | None
    completion_time_sec: float | None
    stable_duration_sec: float

    def validate(self) -> None:
        if self.success:
            if self.outcome != "success" or self.failure_code != "none":
                raise ValidationError("successful trial must use outcome=success and failure_code=none")
            if (
                self.blue_completed_sec is None
                or self.red_completed_sec is None
                or self.completion_time_sec is None
                or self.stable_duration_sec != STABLE_SUCCESS_S
            ):
                raise ValidationError("successful trial lacks reviewed task milestones")
            if not (
                0
                < self.blue_completed_sec
                < self.red_completed_sec
                <= self.completion_time_sec
                and self.completion_time_sec - self.red_completed_sec >= STABLE_SUCCESS_S
            ):
                raise ValidationError("success requires blue, then red, then three stable seconds")
        elif self.outcome == "success" or not self.failure_code:
            raise ValidationError("failed trial requires a nonempty failure code")
        for name, value in (
            ("blue_completed_sec", self.blue_completed_sec),
            ("red_completed_sec", self.red_completed_sec),
            ("completion_time_sec", self.completion_time_sec),
        ):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= TRIAL_DURATION_S):
                raise ValidationError(f"{name} must fall within the 60-second trial")


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial_id: str
    index: int
    condition: Mapping[str, Any]
    started_at: str
    completed_at: str
    duration_sec: float
    actions: int
    replans: int
    inference_latency_ms: tuple[float, ...]
    control_latency_ms: tuple[float, ...]
    outcome: TrialOutcome
    safety_events: tuple[str, ...]
    trace_path: str
    front_video_path: str
    up_video_path: str

    @property
    def completed_safely(self) -> bool:
        return not self.safety_events


@dataclass(frozen=True, slots=True)
class PhaseResult:
    session_id: str
    policy: str
    phase: str
    started_at: str
    completed_at: str
    speed_scale: float
    held_action: Mapping[str, float] | None
    trials: tuple[TrialResult, ...]
    terminal_event: str | None
    terminal_reason: str | None

    @property
    def status(self) -> str:
        if self.terminal_event is None:
            return "completed"
        return "unsafe_shakedown"


@dataclass(frozen=True, slots=True)
class _Abort(Exception):
    event: str
    reason: str


def canonical_scored_conditions() -> tuple[TrialCondition, ...]:
    """Return the preregistered seed-1000 order shared with Repo B."""

    nominal = tuple(
        TrialCondition(f"nominal_{index:02d}", "nominal") for index in range(1, 7)
    )
    offsets = ((-25.0, -25.0), (-25.0, 25.0), (25.0, -25.0), (25.0, 25.0))
    robustness = tuple(
        TrialCondition(
            f"robustness_{index:02d}",
            "robustness",
            blue_axis="pad_x",
            blue_offset_mm=blue,
            red_axis="pad_y",
            red_offset_mm=red,
        )
        for index, (blue, red) in enumerate(offsets, start=1)
    )
    return nominal + robustness


def shakedown_conditions() -> tuple[TrialCondition, ...]:
    return tuple(
        TrialCondition(f"shakedown_{index:02d}", "shakedown") for index in range(1, 3)
    )


def execute_phase(
    permit: MotionPermit,
    candidate: AcceptedPolicyCandidate,
    *,
    runtime_factory: Callable[[AcceptedPolicyCandidate], PolicyRuntime],
    robot_factory: Callable[[MotionPermit], RolloutRobot],
    evidence_factory: EvidenceFactory,
    operator: TrialOperator,
    safety_monitor: SafetyMonitor,
    clock: Callable[[], float] = time.perf_counter,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleep: Callable[[float], None] = time.sleep,
    trial_runner: Callable[..., TrialResult] | None = None,
) -> PhaseResult:
    """Execute exactly one authorized phase and always tear the robot down."""

    assert_permit_current(permit)
    if candidate.bundle_id != permit.candidate_bundle_id or candidate.policy != permit.policy:
        raise SafetyGateError("motion permit belongs to another accepted policy candidate")
    if permit.phase not in {"hold", "shakedown", "scored"}:
        raise SafetyGateError("motion permit names an unsupported phase")

    started_at = _utc_now()
    begin_permit_execution(permit)
    robot: RolloutRobot | None = None
    connected = False
    try:
        robot = robot_factory(permit)
        robot.connect(calibrate=False)
        connected = True
        if permit.phase == "hold":
            receipt = robot.capture_observation()
            held = _state_from_observation(receipt.observation)
            return PhaseResult(
                permit.session_id,
                permit.policy,
                permit.phase,
                started_at,
                _utc_now(),
                permit.speed_scale,
                held,
                (),
                None,
                None,
            )

        # Loading a model is intentionally after the permit and after the
        # reviewed robot has successfully entered current-pose hold.
        runtime = runtime_factory(candidate)
        conditions = (
            shakedown_conditions() if permit.phase == "shakedown" else canonical_scored_conditions()
        )
        run = trial_runner or run_control_trial
        results: list[TrialResult] = []
        for index, condition in enumerate(conditions):
            trial_id = f"{permit.session_id}-{permit.phase}-{index + 1:02d}"
            operator.prepare_trial(permit.session_id, trial_id, condition.to_dict())
            runtime.reset()
            recorder = evidence_factory.start_trial(trial_id)
            try:
                result = run(
                    permit,
                    candidate,
                    robot,
                    runtime,
                    recorder,
                    operator,
                    safety_monitor,
                    index=index,
                    trial_id=trial_id,
                    condition=condition,
                    clock=clock,
                    clock_ns=clock_ns,
                    sleep=sleep,
                )
            finally:
                recorder.close()
            results.append(result)
            if not result.completed_safely:
                event = result.safety_events[-1]
                return PhaseResult(
                    permit.session_id,
                    permit.policy,
                    permit.phase,
                    started_at,
                    _utc_now(),
                    permit.speed_scale,
                    None,
                    tuple(results),
                    event,
                    f"trial {trial_id} terminated with {event}",
                )
        return PhaseResult(
            permit.session_id,
            permit.policy,
            permit.phase,
            started_at,
            _utc_now(),
            permit.speed_scale,
            None,
            tuple(results),
            None,
            None,
        )
    finally:
        try:
            if connected and robot is not None:
                robot.disconnect()
        finally:
            finish_permit_execution(permit)


def run_control_trial(
    permit: MotionPermit,
    candidate: AcceptedPolicyCandidate,
    robot: RolloutRobot,
    runtime: PolicyRuntime,
    recorder: TrialEvidence,
    operator: TrialOperator,
    safety_monitor: SafetyMonitor,
    *,
    index: int,
    trial_id: str,
    condition: TrialCondition,
    clock: Callable[[], float] = time.perf_counter,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> TrialResult:
    """Run one canonical 60-second control loop, aborting before unsafe writes."""

    assert_permit_current(permit)
    started_wall = datetime.now(UTC)
    started = clock()
    next_tick = started
    previous_write: float | None = None
    previous_elapsed_s = 0.0
    inference_samples: list[float] = []
    control_samples: list[float] = []
    actions = 0
    replans = 0
    safety_events: list[str] = []
    initial_receipt = getattr(robot, "last_receipt", None)
    previous_command_sequence = int(getattr(initial_receipt, "command_sequence", 0))
    previous_feedback_received_ns = (
        0
        if previous_command_sequence == 0
        else int(getattr(initial_receipt, "feedback_received_ns", 0))
    )

    try:
        for step in range(round(TRIAL_DURATION_S * TARGET_HZ)):
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
            iteration_started = clock()
            if operator.abort_requested():
                raise _Abort("operator_abort", "operator requested a stop")
            monitor_event = safety_monitor.event()
            if monitor_event is not None:
                if monitor_event not in {"collision", "intervention", "operator_abort"}:
                    raise _Abort("safety_abort", f"unsupported safety monitor event {monitor_event}")
                raise _Abort(monitor_event, f"safety monitor reported {monitor_event}")
            if previous_write is not None:
                period_ms = (iteration_started - previous_write) * 1_000.0
                if period_ms > 1_000.0 / TARGET_HZ + RATE_TOLERANCE_MS:
                    raise _Abort("deadline_miss", f"control period was {period_ms:.3f} ms")

            try:
                observation_receipt = robot.capture_observation()
                observation = _policy_observation(observation_receipt.observation)
                freshness = _camera_freshness(observation_receipt, clock_ns())
            except _Abort:
                raise
            except Exception as exc:
                raise _Abort(
                    "feedback_loss",
                    f"observation/feedback transaction failed: {type(exc).__name__}: {exc}",
                ) from exc
            if (
                observation_receipt.observation_read_ms > OBSERVATION_DEADLINE_MS
                or any(
                    value > OBSERVATION_DEADLINE_MS
                    for value in freshness["frame_age_ms"].values()
                )
            ):
                raise _Abort("stale_observation", "camera observation exceeded 100 ms")

            inference_started = clock()
            try:
                proposed = finite_action(
                    runtime.infer(prepare_observation(candidate.spec, observation))
                )
            except Exception as exc:
                raise _Abort("malformed_action", f"policy action failed validation: {exc}") from exc
            inference_ms = (clock() - inference_started) * 1_000.0
            elapsed_ms = (clock() - iteration_started) * 1_000.0
            if inference_ms >= CONTROL_DEADLINE_MS or elapsed_ms >= CONTROL_DEADLINE_MS:
                raise _Abort("deadline_miss", "inference/control deadline reached before write")

            proposed_map = dict(zip(ACTION_KEYS, proposed, strict=True))
            state = _state_from_observation(observation_receipt.observation)
            try:
                validate_action(proposed_map, state, permit)
            except SafetyGateError as exc:
                raise _Abort("clamped_action", f"action would require clamping: {exc}") from exc

            # Validate/copy camera bytes before a motor write.  The production
            # recorder can therefore guarantee that a successful reservation
            # already owns the camera frames for the action it precedes.
            try:
                reservation = recorder.reserve(
                    front=observation_receipt.observation["front"],
                    up=observation_receipt.observation["up"],
                )
            except Exception as exc:
                raise _Abort(
                    "safety_abort", f"evidence reservation failed: {exc}"
                ) from exc
            if reservation is None:
                raise _Abort("safety_abort", "evidence queue has no capacity")
            try:
                sent = robot.send_action(proposed_map)
            except SafetyGateError as exc:
                reservation.cancel()
                raise _Abort("clamped_action", f"write-side safety check rejected action: {exc}") from exc
            except Exception as exc:
                failed_receipt = _post_write_failure_receipt(
                    exc,
                    robot.last_receipt,
                    expected_sequence=previous_command_sequence + 1,
                    expected_previous_feedback_received_ns=previous_feedback_received_ns,
                )
                if failed_receipt is None:
                    reservation.cancel()
                    raise _Abort("feedback_loss", f"write/feedback transaction failed: {exc}") from exc

                # The motor write is irreversible at this point.  Consume the
                # pre-write reservation with an explicit missing-feedback row
                # instead of discarding the only frames bound to that action.
                write_finished = clock()
                control_ms = (write_finished - iteration_started) * 1_000.0
                elapsed_s = write_finished - started
                period_ms = (elapsed_s - previous_elapsed_s) * 1_000.0
                actions += 1
                replans += int(step % 10 == 0)
                inference_samples.append(inference_ms)
                control_samples.append(control_ms)
                failure_row = {
                    "trial": trial_id,
                    "condition": condition.to_dict(),
                    "index": step,
                    "elapsed_s": elapsed_s,
                    "observation_read_ms": observation_receipt.observation_read_ms,
                    "camera_freshness": freshness,
                    "replan": step % 10 == 0,
                    "state": state,
                    "proposed_action": proposed_map,
                    "sent_action": failed_receipt["sent_action"],
                    "feedback_action": None,
                    "feedback_check": failed_receipt["feedback_check"],
                    "inference_ms": inference_ms,
                    "control_ms": control_ms,
                    "deadline_ms": CONTROL_DEADLINE_MS,
                    "deadline_passed": control_ms < CONTROL_DEADLINE_MS,
                    "target_period_ms": 1_000.0 / TARGET_HZ,
                    "rate_tolerance_ms": RATE_TOLERANCE_MS,
                    "period_ms": period_ms,
                    "rate_passed": period_ms <= 1_000.0 / TARGET_HZ + RATE_TOLERANCE_MS,
                    "limit_check": _limit_receipt(permit, state, proposed_map),
                }
                try:
                    reservation.commit(failure_row)
                except Exception as evidence_error:
                    raise _Abort(
                        "safety_abort",
                        f"evidence writer failed after confirmed motor write: {evidence_error}",
                    ) from evidence_error
                raise _Abort("feedback_loss", str(exc)) from exc

            # A normal return confirms the SDK write.  Count it even if later
            # receipt validation or evidence persistence reports a fault.
            actions += 1
            write_finished = clock()
            control_ms = (write_finished - iteration_started) * 1_000.0
            elapsed_s = write_finished - started
            period_ms = (elapsed_s - previous_elapsed_s) * 1_000.0
            rate_passed = period_ms <= 1_000.0 / TARGET_HZ + RATE_TOLERANCE_MS
            row = {
                "trial": trial_id,
                "condition": condition.to_dict(),
                "index": step,
                "elapsed_s": elapsed_s,
                "observation_read_ms": observation_receipt.observation_read_ms,
                "camera_freshness": freshness,
                "replan": step % 10 == 0,
                "state": state,
                "proposed_action": proposed_map,
                "sent_action": sent,
                "feedback_action": None,
                "feedback_check": None,
                "inference_ms": inference_ms,
                "control_ms": control_ms,
                "deadline_ms": CONTROL_DEADLINE_MS,
                "deadline_passed": control_ms < CONTROL_DEADLINE_MS,
                "target_period_ms": 1_000.0 / TARGET_HZ,
                "rate_tolerance_ms": RATE_TOLERANCE_MS,
                "period_ms": period_ms,
                "rate_passed": rate_passed,
                "limit_check": _limit_receipt(permit, state, proposed_map),
            }
            try:
                feedback = _feedback_after(robot.last_receipt)
                row["feedback_action"] = feedback
                feedback_check = _feedback_receipt(
                    permit,
                    proposed_map,
                    feedback,
                    robot.last_receipt,
                    expected_sequence=previous_command_sequence + 1,
                    expected_previous_feedback_received_ns=previous_feedback_received_ns,
                )
                row["feedback_check"] = feedback_check
            except Exception as exc:
                # send_action returned, so the motor write is confirmed.  Keep
                # the reserved frames and action even when its receipt is bad;
                # cancelling here would erase the only evidence for that write.
                row["feedback_check"] = {
                    "freshness_source": "bounded_synchronous_fashionstar_monitor_poll",
                    "joint_order": list(JOINT_NAMES),
                    "receipt_violations": ["invalid_post_write_receipt"],
                    "detail": f"{type(exc).__name__}: {exc}",
                    "passed": False,
                }
                try:
                    reservation.commit(row)
                except Exception as evidence_error:
                    raise _Abort(
                        "safety_abort",
                        f"evidence writer failed after confirmed motor write: {evidence_error}",
                    ) from evidence_error
                replans += int(step % 10 == 0)
                inference_samples.append(inference_ms)
                control_samples.append(control_ms)
                raise _Abort(
                    "feedback_loss", f"post-write receipt is invalid: {exc}"
                ) from exc
            try:
                reservation.commit(
                    row,
                )
            except Exception as exc:
                reservation.cancel()
                raise _Abort("safety_abort", f"evidence writer failed after action: {exc}") from exc

            replans += int(step % 10 == 0)
            inference_samples.append(inference_ms)
            control_samples.append(control_ms)
            previous_write = write_finished
            previous_elapsed_s = elapsed_s
            previous_command_sequence = feedback_check["command_sequence"]
            previous_feedback_received_ns = feedback_check["feedback_received_ns"]
            if not feedback_check["passed"]:
                raise _Abort("feedback_loss", "post-write feedback receipt failed validation")
            if control_ms >= CONTROL_DEADLINE_MS:
                raise _Abort("deadline_miss", "motor feedback missed the control deadline")
            if not rate_passed:
                raise _Abort("deadline_miss", "control rate missed the reviewed tolerance")
            next_tick += TARGET_PERIOD_S
        remaining = started + TRIAL_DURATION_S - clock()
        if remaining > 0:
            sleep(remaining)
    except _Abort as abort:
        safety_events.append(abort.event)
        recorder.record_terminal(
            {
                "trial": trial_id,
                "condition": condition.to_dict(),
                "index": actions,
                "elapsed_s": max(clock() - started, 0.0),
                "event": abort.event,
                "detail": abort.reason,
                "actions_sent": actions,
            }
        )

    completed_wall = datetime.now(UTC)
    if safety_events:
        outcome = TrialOutcome(False, "safety_abort", safety_events[-1], None, None, None, 0.0)
    else:
        outcome = operator.outcome(trial_id)
        outcome.validate()
    return TrialResult(
        trial_id=trial_id,
        index=index,
        condition=condition.to_dict(),
        started_at=started_wall.isoformat(),
        completed_at=completed_wall.isoformat(),
        duration_sec=max(clock() - started, 0.0),
        actions=actions,
        replans=replans,
        inference_latency_ms=tuple(inference_samples),
        control_latency_ms=tuple(control_samples),
        outcome=outcome,
        safety_events=tuple(safety_events),
        trace_path=f"trials/trial-{index:02d}.jsonl",
        front_video_path=f"videos/trial-{index:02d}-front.mp4",
        up_video_path=f"videos/trial-{index:02d}-up.mp4",
    )


def _policy_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    state = _state_from_observation(value)
    if "front" not in value or "up" not in value:
        raise _Abort("stale_observation", "reviewed front/up camera samples are missing")
    return {
        "observation.state": tuple(state[key] for key in ACTION_KEYS),
        "observation.images.front": value["front"],
        "observation.images.up": value["up"],
        "task": CANONICAL_TASK,
    }


def _state_from_observation(value: Mapping[str, Any]) -> dict[str, float]:
    try:
        return {key: _finite(value[key], f"state {key}") for key in ACTION_KEYS}
    except KeyError as exc:
        raise _Abort("feedback_loss", f"state feedback is missing {exc.args[0]}") from exc


def _feedback_after(receipt: Any) -> dict[str, float]:
    if receipt is None or not isinstance(receipt.feedback_after, Mapping):
        raise SafetyGateError("robot write did not produce synchronous feedback")
    return {key: _finite(receipt.feedback_after[key], f"feedback {key}") for key in ACTION_KEYS}


def _post_write_failure_receipt(
    error: Exception,
    robot_receipt: Any,
    *,
    expected_sequence: int,
    expected_previous_feedback_received_ns: int,
) -> dict[str, Any] | None:
    """Return evidence only for the adapter's definitive post-write failure."""

    receipt = getattr(error, "action_receipt", None)
    if receipt is None or receipt is not robot_receipt:
        return None
    sequence = int(getattr(receipt, "command_sequence", 0))
    previous_received = int(getattr(receipt, "previous_feedback_received_ns", -1))
    control_started = int(getattr(receipt, "control_started_ns", 0))
    prewrite_received = int(getattr(receipt, "prewrite_received_ns", 0))
    write_started = int(getattr(receipt, "write_started_ns", 0))
    write_completed = int(getattr(receipt, "write_completed_ns", 0))
    feedback_failed = int(getattr(receipt, "feedback_failed_ns", 0))
    feedback_error = getattr(receipt, "feedback_error", None)
    if (
        sequence != expected_sequence
        or previous_received != expected_previous_feedback_received_ns
        or not isinstance(feedback_error, str)
        or not feedback_error
        or not (
            0 <= previous_received <= control_started < prewrite_received < write_started
            <= write_completed <= feedback_failed
        )
    ):
        return None
    try:
        before = _receipt_action(receipt, "feedback_before")
        sent = _receipt_action(receipt, "sent")
        minimum = _minimum_progress(receipt)
    except _Abort:
        return None
    check = {
        "freshness_source": "bounded_synchronous_fashionstar_monitor_poll",
        "joint_order": list(JOINT_NAMES),
        "command_sequence": sequence,
        "previous_feedback_received_ns": previous_received,
        "control_started_ns": control_started,
        "prewrite_received_ns": prewrite_received,
        "write_started_ns": write_started,
        "write_completed_ns": write_completed,
        "feedback_received_ns": 0,
        "feedback_failed_ns": feedback_failed,
        "poll_count": int(getattr(receipt, "poll_count", 0)),
        "prewrite_action": before,
        "target_action": sent,
        "feedback_action": None,
        "minimum_observable_progress": minimum,
        "receipt_violations": ["feedback_read_failed"],
        "feedback_error": feedback_error,
        "passed": False,
    }
    return {"sent_action": sent, "feedback_check": check}


def _camera_freshness(receipt: Any, now_ns: int) -> dict[str, Any]:
    if set(receipt.camera_reads) != {"front", "up"}:
        raise _Abort("stale_observation", "camera receipt does not bind front and up")
    ages: dict[str, float] = {}
    intervals: dict[str, None] = {}
    for name in ("front", "up"):
        item = receipt.camera_reads[name]
        received = int(item["received_at_ns"])
        _finite(item["read_ms"], f"{name} camera read_ms")
        age_ms = max((now_ns - received) / 1_000_000.0, 0.0)
        ages[name] = age_ms
        intervals[name] = None
    return {
        "public_api": "OpenCVCamera.async_read",
        "arrival_clock": "time.perf_counter",
        "read_deadline_ms": OBSERVATION_DEADLINE_MS,
        "frame_age_ms": ages,
        "interarrival_ms": intervals,
    }


def _limit_receipt(
    permit: MotionPermit,
    state: Mapping[str, float],
    proposed: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "joint_order": list(JOINT_NAMES),
        "reference_action": dict(state),
        "lower": {joint: permit.absolute_limits[joint][0] for joint in JOINT_NAMES},
        "upper": {joint: permit.absolute_limits[joint][1] for joint in JOINT_NAMES},
        "reviewed_max_step_deltas": dict(permit.max_step_deltas),
        "speed_scale": permit.speed_scale,
        "allowed_step_deltas": {
            joint: permit.max_step_deltas[joint] * permit.speed_scale for joint in JOINT_NAMES
        },
        "absolute_violations": [],
        "rate_violations": [],
        "would_be_clamped": [],
        "passed": True,
        "proposed_action": dict(proposed),
    }


def _feedback_receipt(
    permit: MotionPermit,
    proposed: Mapping[str, float],
    feedback: Mapping[str, float],
    receipt: Any,
    *,
    expected_sequence: int,
    expected_previous_feedback_received_ns: int,
) -> dict[str, Any]:
    """Recompute the synchronous feedback and receipt safety decision."""

    before = _receipt_action(receipt, "feedback_before")
    minimum = _minimum_progress(receipt)
    allowed = {
        joint: permit.max_step_deltas[joint] * permit.speed_scale for joint in JOINT_NAMES
    }
    command_delta = {
        joint: proposed[f"{joint}.pos"] - before[f"{joint}.pos"] for joint in JOINT_NAMES
    }
    material = [
        joint
        for joint in JOINT_NAMES
        if abs(command_delta[joint]) >= math.nextafter(minimum[joint], 0.0)
    ]
    signed_progress = {
        joint: (
            0.0
            if joint not in material
            else math.copysign(1.0, command_delta[joint])
            * (feedback[f"{joint}.pos"] - before[f"{joint}.pos"])
        )
        for joint in JOINT_NAMES
    }
    target_overshoot = {
        joint: (
            0.0
            if joint not in material
            else max(0.0, signed_progress[joint] - abs(command_delta[joint]))
        )
        for joint in JOINT_NAMES
    }
    absolute_error = {
        joint: abs(feedback[f"{joint}.pos"] - proposed[f"{joint}.pos"])
        for joint in JOINT_NAMES
    }
    progress_violations = [
        joint
        for joint in JOINT_NAMES
        if joint in material
        and signed_progress[joint] < math.nextafter(minimum[joint], 0.0)
    ]
    tracking_violations = [
        joint
        for joint in JOINT_NAMES
        if absolute_error[joint] > math.nextafter(allowed[joint], math.inf)
    ]
    overshoot_violations = [
        joint
        for joint in JOINT_NAMES
        if target_overshoot[joint] > math.nextafter(allowed[joint], math.inf)
    ]
    limit_violations = [
        joint
        for joint in JOINT_NAMES
        if not permit.absolute_limits[joint][0]
        <= feedback[f"{joint}.pos"]
        <= permit.absolute_limits[joint][1]
    ]
    sequence = int(getattr(receipt, "command_sequence", 0))
    previous_received = int(getattr(receipt, "previous_feedback_received_ns", 0))
    control_started = int(getattr(receipt, "control_started_ns", 0))
    prewrite_received = int(getattr(receipt, "prewrite_received_ns", 0))
    write_started = int(getattr(receipt, "write_started_ns", 0))
    write_completed = int(getattr(receipt, "write_completed_ns", 0))
    feedback_received = int(getattr(receipt, "feedback_received_ns", 0))
    command_deadline = control_started + round(CONTROL_DEADLINE_MS * 1_000_000.0)
    receipt_violations: list[str] = []
    if sequence != expected_sequence:
        receipt_violations.append("nonconsecutive_command_sequence")
    if previous_received != expected_previous_feedback_received_ns:
        receipt_violations.append("broken_receipt_chain")
    if not (
        0 <= previous_received <= control_started < prewrite_received < write_started
        <= write_completed < feedback_received <= command_deadline
    ):
        receipt_violations.append("nonmonotonic_command_receipt")
    if any(
        not permit.absolute_limits[joint][0]
        <= before[f"{joint}.pos"]
        <= permit.absolute_limits[joint][1]
        for joint in JOINT_NAMES
    ):
        receipt_violations.append("prewrite_outside_reviewed_limits")
    if any(
        abs(command_delta[joint]) > math.nextafter(allowed[joint], math.inf)
        for joint in JOINT_NAMES
    ):
        receipt_violations.append("target_exceeds_reviewed_step")
    violations = [
        joint
        for joint in JOINT_NAMES
        if joint
        in set(progress_violations)
        | set(tracking_violations)
        | set(overshoot_violations)
        | set(limit_violations)
    ]
    passed = not violations and not receipt_violations
    poll_count = int(getattr(receipt, "poll_count", 0))
    if poll_count <= 0:
        receipt_violations.append("missing_feedback_poll")
        passed = False
    return {
        "freshness_source": "bounded_synchronous_fashionstar_monitor_poll",
        "joint_order": list(JOINT_NAMES),
        "command_sequence": sequence,
        "previous_feedback_received_ns": previous_received,
        "control_started_ns": control_started,
        "prewrite_received_ns": prewrite_received,
        "write_started_ns": write_started,
        "write_completed_ns": write_completed,
        "feedback_received_ns": feedback_received,
        "command_deadline_ns": command_deadline,
        "poll_count": poll_count,
        "prewrite_action": before,
        "target_action": dict(proposed),
        "feedback_action": dict(feedback),
        "minimum_observable_progress": minimum,
        "material_joints": material,
        "signed_progress": signed_progress,
        "target_overshoot": target_overshoot,
        "absolute_error": absolute_error,
        "allowed_error": allowed,
        "progress_violations": progress_violations,
        "tracking_violations": tracking_violations,
        "overshoot_violations": overshoot_violations,
        "limit_violations": limit_violations,
        "receipt_violations": receipt_violations,
        "violations": violations,
        "passed": passed,
    }


def _receipt_action(receipt: Any, name: str) -> dict[str, float]:
    value = getattr(receipt, name, None)
    if not isinstance(value, Mapping) or set(value) != set(ACTION_KEYS):
        raise _Abort("feedback_loss", f"robot receipt lacks exact {name}")
    return {key: _finite(value[key], f"receipt {name} {key}") for key in ACTION_KEYS}


def _minimum_progress(receipt: Any) -> dict[str, float]:
    value = getattr(receipt, "minimum_observable_progress", None)
    if not isinstance(value, Mapping) or set(value) != set(JOINT_NAMES):
        raise _Abort("feedback_loss", "robot receipt lacks monitor quantization evidence")
    result = {joint: _finite(value[joint], f"{joint} monitor quantum") for joint in JOINT_NAMES}
    if any(item <= 0 for item in result.values()):
        raise _Abort("feedback_loss", "monitor quantization evidence must be positive")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Abort("feedback_loss", f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _Abort("feedback_loss", f"{label} is not finite")
    return result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "ACTION_KEYS",
    "CONTROL_DEADLINE_MS",
    "OBSERVATION_DEADLINE_MS",
    "RATE_TOLERANCE_MS",
    "SAFETY_EVENTS",
    "TARGET_HZ",
    "TRIAL_DURATION_S",
    "PhaseResult",
    "TrialCondition",
    "TrialOutcome",
    "TrialResult",
    "canonical_scored_conditions",
    "execute_phase",
    "run_control_trial",
    "shakedown_conditions",
]
