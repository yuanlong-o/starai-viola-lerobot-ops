from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

import viola_ops.safety as safety
from viola_ops.errors import SafetyGateError
from viola_ops.execution import (
    ACTION_KEYS,
    TrialOutcome,
    TrialResult,
    execute_phase,
    run_control_trial,
)
from viola_ops.policies import get_policy_spec
from viola_ops.safety import JOINTS, MotionPermit


def _permit(phase: str = "shakedown", *, forged: bool = False) -> MotionPermit:
    now = datetime.now(UTC)
    return MotionPermit(
        session_id="session-1",
        session_bundle_id="a" * 64,
        candidate_bundle_id="b" * 64,
        policy="act",
        phase=phase,
        trial="phase",
        operator="operator",
        setup_hashes={name: "c" * 64 for name in ("calibration", "camera", "robot", "reset")},
        absolute_limits={
            joint: ((0.0, 100.0) if joint == "gripper" else (-100.0, 100.0))
            for joint in JOINTS
        },
        max_step_deltas={joint: 4.0 for joint in JOINTS},
        speed_scale=0.25 if phase == "shakedown" else 1.0,
        issued_at=now,
        estop_tested_at=now - timedelta(hours=1),
        _nonce="nonce",
        _authority=object() if forged else safety._PERMIT_AUTHORITY,
    )


def _candidate():
    return SimpleNamespace(
        bundle_id="b" * 64,
        policy="act",
        spec=get_policy_spec("act"),
    )


class _Runtime:
    def __init__(self, action=None):
        self.action = action or [0.0] * 6 + [50.0]
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, _observation):
        return self.action


class _Robot:
    def __init__(self):
        self.connected = False
        self.connects = 0
        self.disconnects = 0
        self.writes = []
        self.last_receipt = None
        self.read_ms = 1.0
        self.camera_age_ns = 0

    def connect(self, calibrate=False):
        assert calibrate is False
        self.connected = True
        self.connects += 1

    def disconnect(self):
        self.connected = False
        self.disconnects += 1

    def capture_observation(self):
        state = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
        now_ns = 1_000_000_000
        return SimpleNamespace(
            observation={
                **state,
                "front": np.zeros((2, 2, 3), dtype=np.uint8),
                "up": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            observation_read_ms=self.read_ms,
            camera_reads={
                name: {"received_at_ns": now_ns - self.camera_age_ns, "read_ms": 1.0}
                for name in ("front", "up")
            },
        )

    def send_action(self, action):
        self.writes.append(dict(action))
        self.last_receipt = SimpleNamespace(feedback_after=dict(action))
        return dict(action)


class _Reservation:
    def __init__(self):
        self.rows = []
        self.cancelled = False

    def commit(self, row):
        self.rows.append(dict(row))

    def cancel(self):
        self.cancelled = True


class _Recorder:
    def __init__(self, *, capacity=True, reservation_error: Exception | None = None):
        self.capacity = capacity
        self.reservation_error = reservation_error
        self.terminal = []
        self.closed = False
        self.reservations = []

    def reserve(self, *, front, up):
        if self.reservation_error is not None:
            raise self.reservation_error
        if not self.capacity:
            return None
        reservation = _Reservation()
        self.reservations.append(reservation)
        return reservation

    def record_terminal(self, row):
        self.terminal.append(dict(row))

    def close(self):
        self.closed = True


class _EvidenceFactory:
    def __init__(self):
        self.recorders = []

    def start_trial(self, _trial_id):
        recorder = _Recorder()
        self.recorders.append(recorder)
        return recorder


class _Operator:
    def __init__(self, *, abort=False):
        self.abort = abort
        self.prepared = []

    def prepare_trial(self, session_id, trial_id, condition):
        self.prepared.append((session_id, trial_id, condition))

    def abort_requested(self):
        return self.abort

    def outcome(self, _trial_id):
        return TrialOutcome(False, "failure", "timeout", None, None, None, 0.0)


class _Monitor:
    def __init__(self, event=None):
        self.value = event

    def event(self):
        return self.value


def _aborting_trial(*_args, index, trial_id, condition, **_kwargs):
    return TrialResult(
        trial_id=trial_id,
        index=index,
        condition=condition.to_dict(),
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        duration_sec=60.0,
        actions=1,
        replans=1,
        inference_latency_ms=(1.0,),
        control_latency_ms=(1.0,),
        outcome=TrialOutcome(False, "failure", "timeout", None, None, None, 0.0),
        safety_events=(),
        trace_path="trace",
        front_video_path="front",
        up_video_path="up",
    )


def _unsafe_trial(*_args, index, trial_id, condition, **_kwargs):
    return TrialResult(
        trial_id=trial_id,
        index=index,
        condition=condition.to_dict(),
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        duration_sec=0.1,
        actions=0,
        replans=0,
        inference_latency_ms=(),
        control_latency_ms=(),
        outcome=TrialOutcome(
            False, "safety_abort", "collision", None, None, None, 0.0
        ),
        safety_events=("collision",),
        trace_path="trace",
        front_video_path="front",
        up_video_path="up",
    )


def test_forged_permit_fails_before_policy_or_hardware_factory() -> None:
    calls = []
    with pytest.raises(SafetyGateError, match="issued by the motion gate"):
        execute_phase(
            _permit(forged=True),
            _candidate(),
            runtime_factory=lambda candidate: calls.append("policy"),
            robot_factory=lambda permit: calls.append("robot"),
            evidence_factory=_EvidenceFactory(),
            operator=_Operator(),
            safety_monitor=_Monitor(),
        )
    assert calls == []


@pytest.mark.parametrize("phase,expected", [("shakedown", 2), ("scored", 10)])
def test_phase_schedule_is_exact_and_teardown_is_guaranteed(phase, expected) -> None:
    robot = _Robot()
    runtime = _Runtime()
    operator = _Operator()
    result = execute_phase(
        _permit(phase),
        _candidate(),
        runtime_factory=lambda candidate: runtime,
        robot_factory=lambda permit: robot,
        evidence_factory=_EvidenceFactory(),
        operator=operator,
        safety_monitor=_Monitor(),
        trial_runner=_aborting_trial,
    )
    assert len(result.trials) == expected
    assert len(operator.prepared) == expected
    assert runtime.resets == expected
    assert robot.connects == robot.disconnects == 1


def test_phase_stops_after_first_unsafe_trial_and_tears_down() -> None:
    robot = _Robot()
    operator = _Operator()
    result = execute_phase(
        _permit("scored"),
        _candidate(),
        runtime_factory=lambda _candidate: _Runtime(),
        robot_factory=lambda _permit: robot,
        evidence_factory=_EvidenceFactory(),
        operator=operator,
        safety_monitor=_Monitor(),
        trial_runner=_unsafe_trial,
    )
    assert result.status == "unsafe_shakedown"
    assert result.terminal_event == "collision"
    assert len(result.trials) == len(operator.prepared) == 1
    assert robot.writes == []
    assert robot.connects == robot.disconnects == 1


def test_process_control_is_not_masked_by_recorder_or_robot_cleanup() -> None:
    signal = KeyboardInterrupt("operator stop")

    class BrokenRecorder(_Recorder):
        def close(self):
            self.closed = True
            raise RuntimeError("recorder close failed")

    class BrokenEvidenceFactory:
        def start_trial(self, _trial_id):
            return BrokenRecorder()

    class BrokenDisconnectRobot(_Robot):
        def disconnect(self):
            super().disconnect()
            raise RuntimeError("robot disconnect failed")

    def interrupt_trial(*_args, **_kwargs):
        raise signal

    robot = BrokenDisconnectRobot()
    with pytest.raises(KeyboardInterrupt) as raised:
        execute_phase(
            _permit("shakedown"),
            _candidate(),
            runtime_factory=lambda _candidate: _Runtime(),
            robot_factory=lambda _permit: robot,
            evidence_factory=BrokenEvidenceFactory(),
            operator=_Operator(),
            safety_monitor=_Monitor(),
            trial_runner=interrupt_trial,
        )

    assert raised.value is signal
    assert robot.disconnects == 1
    notes = getattr(signal, "__notes__", [])
    assert any("trial evidence cleanup also failed" in note for note in notes)
    assert any("execution cleanup also failed" in note for note in notes)


def test_motion_permit_is_consumed_after_one_execution_attempt() -> None:
    permit = _permit("hold")
    robot = _Robot()
    first = execute_phase(
        permit,
        _candidate(),
        runtime_factory=lambda _candidate: _Runtime(),
        robot_factory=lambda _permit: robot,
        evidence_factory=_EvidenceFactory(),
        operator=_Operator(),
        safety_monitor=_Monitor(),
    )
    assert first.status == "completed"
    with pytest.raises(SafetyGateError, match="consumed|exactly one"):
        execute_phase(
            permit,
            _candidate(),
            runtime_factory=lambda _candidate: _Runtime(),
            robot_factory=lambda _permit: _Robot(),
            evidence_factory=_EvidenceFactory(),
            operator=_Operator(),
            safety_monitor=_Monitor(),
        )


@pytest.mark.parametrize(
    "action,event",
    [
        ([float("nan")] + [0.0] * 5 + [50.0], "malformed_action"),
        ([2.0] + [0.0] * 5 + [50.0], "clamped_action"),
    ],
)
def test_bad_policy_action_aborts_without_a_motor_write(action, event) -> None:
    robot = _Robot()
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(action),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="session-1-shakedown-01",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert result.safety_events == (event,)
    assert robot.writes == []


def test_stale_camera_and_full_evidence_queue_abort_before_write() -> None:
    robot = _Robot()
    robot.read_ms = 101.0
    stale = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        _Recorder(),
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert stale.safety_events == ("stale_observation",)
    assert robot.writes == []

    robot.read_ms = 1.0
    saturated = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        _Recorder(capacity=False),
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert saturated.safety_events == ("safety_abort",)
    assert robot.writes == []


def test_frame_reservation_failure_is_typed_and_aborts_before_write() -> None:
    robot = _Robot()
    recorder = _Recorder(reservation_error=ValueError("bad front frame"))
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert result.safety_events == ("safety_abort",)
    assert robot.writes == []
    assert recorder.terminal[0]["actions_sent"] == 0


def test_inference_deadline_aborts_before_reservation_or_write() -> None:
    robot = _Robot()
    recorder = _Recorder()
    samples = iter((0.0, 0.0, 0.0, 0.0, 0.301, 0.301, 0.301, 0.301))
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: next(samples),
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert result.safety_events == ("deadline_miss",)
    assert robot.writes == []
    assert recorder.terminal[0]["actions_sent"] == 0


def test_operator_abort_is_checked_before_observation_and_write() -> None:
    robot = _Robot()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        _Recorder(),
        _Operator(abort=True),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert result.safety_events == ("operator_abort",)
    assert robot.writes == []


@pytest.mark.parametrize("event", ["collision", "intervention", "unexpected-signal"])
def test_safety_monitor_event_aborts_before_observation_or_write(event: str) -> None:
    robot = _Robot()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        _Recorder(),
        _Operator(),
        _Monitor(event),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    expected = event if event in {"collision", "intervention"} else "safety_abort"
    assert result.safety_events == (expected,)
    assert robot.writes == []


def test_observation_failure_is_typed_and_aborts_before_write() -> None:
    robot = _Robot()

    def fail_observation():
        raise OSError("camera disconnected")

    robot.capture_observation = fail_observation
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert result.safety_events == ("feedback_loss",)
    assert robot.writes == []
    assert "camera disconnected" in recorder.terminal[0]["detail"]


def test_failed_post_write_feedback_is_recorded_then_stops_the_trial() -> None:
    robot = _Robot()

    def send_with_bad_feedback(action):
        robot.writes.append(dict(action))
        before = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
        after = dict(before)
        after["Motor_0.pos"] = 101.0
        robot.last_receipt = SimpleNamespace(
            feedback_before=before,
            feedback_after=after,
            command_sequence=1,
            previous_feedback_received_ns=1,
            control_started_ns=2,
            prewrite_received_ns=3,
            write_started_ns=4,
            write_completed_ns=5,
            feedback_received_ns=6,
            minimum_observable_progress={joint: 0.01 for joint in JOINTS},
            poll_count=1,
        )
        return dict(action)

    robot.send_action = send_with_bad_feedback
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )
    assert result.safety_events == ("feedback_loss",)
    assert result.actions == len(robot.writes) == 1
    assert len(recorder.reservations[0].rows) == 1
    assert recorder.reservations[0].rows[0]["feedback_check"]["passed"] is False
    assert recorder.terminal[0]["actions_sent"] == 1


def test_feedback_read_exception_keeps_reserved_frames_and_sent_action() -> None:
    robot = _Robot()

    def send_then_fail(action):
        robot.writes.append(dict(action))
        before = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
        receipt = SimpleNamespace(
            sent=dict(action),
            feedback_before=before,
            feedback_after={},
            command_sequence=1,
            previous_feedback_received_ns=0,
            control_started_ns=2,
            prewrite_received_ns=3,
            write_started_ns=4,
            write_completed_ns=5,
            feedback_received_ns=0,
            feedback_failed_ns=6,
            feedback_error="OSError: monitor unavailable",
            minimum_observable_progress={joint: 0.01 for joint in JOINTS},
            poll_count=1,
        )
        robot.last_receipt = receipt
        error = RuntimeError("motor write completed, but feedback failed")
        error.action_receipt = receipt
        raise error

    robot.send_action = send_then_fail
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )

    assert result.safety_events == ("feedback_loss",)
    assert result.actions == len(robot.writes) == 1
    assert len(recorder.reservations) == 1
    reservation = recorder.reservations[0]
    assert reservation.cancelled is False
    assert len(reservation.rows) == 1
    row = reservation.rows[0]
    assert row["sent_action"] == robot.writes[0]
    assert row["feedback_action"] is None
    assert row["feedback_check"]["passed"] is False
    assert row["feedback_check"]["receipt_violations"] == ["feedback_read_failed"]
    assert recorder.terminal[0]["actions_sent"] == 1


def test_ambiguous_sdk_write_keeps_frames_without_claiming_a_sent_action() -> None:
    robot = _Robot()
    attempted = []

    def fail_inside_sdk_call(action):
        attempted.append(dict(action))
        before = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
        receipt = SimpleNamespace(
            proposed=dict(action),
            attempted=dict(action),
            feedback_before=before,
            attempt_sequence=1,
            previous_command_sequence=0,
            previous_feedback_received_ns=0,
            control_started_ns=2,
            prewrite_received_ns=3,
            write_started_ns=4,
            write_failed_ns=5,
            minimum_observable_progress={joint: 0.01 for joint in JOINTS},
            write_error="OSError: serial reply lost",
            write_outcome="unknown",
        )
        robot.last_receipt = receipt
        error = RuntimeError("physical write outcome is unknown")
        error.write_outcome_unknown = True
        error.action_receipt = receipt
        raise error

    robot.send_action = fail_inside_sdk_call
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )

    assert result.safety_events == ("feedback_loss",)
    assert result.actions == 0
    assert len(attempted) == 1
    reservation = recorder.reservations[0]
    assert reservation.cancelled is False
    assert len(reservation.rows) == 1
    row = reservation.rows[0]
    assert row["write_outcome"] == "unknown"
    assert row["attempted_action"] == attempted[0]
    assert row["sent_action"] is None
    assert row["feedback_action"] is None
    assert row["feedback_check"]["passed"] is False
    assert row["feedback_check"]["receipt_violations"] == [
        "sdk_write_outcome_unknown"
    ]
    assert recorder.terminal[0]["actions_sent"] == 0
    assert recorder.terminal[0]["action_attempts"] == 1
    assert recorder.terminal[0]["ambiguous_write_attempts"] == 1
    assert result.replans == 0
    assert result.inference_latency_ms == ()
    assert result.control_latency_ms == ()


def test_invalid_ambiguous_receipt_cannot_replace_known_attempted_action() -> None:
    robot = _Robot()

    def fail_inside_sdk_call(action):
        forged = dict(action)
        forged["Motor_0.pos"] += 0.5
        receipt = SimpleNamespace(
            proposed=forged,
            attempted=forged,
            feedback_before={
                key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS
            },
            attempt_sequence=1,
            previous_command_sequence=0,
            previous_feedback_received_ns=0,
            control_started_ns=2,
            prewrite_received_ns=3,
            write_started_ns=4,
            write_failed_ns=5,
            minimum_observable_progress={joint: 0.01 for joint in JOINTS},
            write_error="OSError: serial reply lost",
            write_outcome="unknown",
        )
        robot.last_receipt = receipt
        error = RuntimeError("physical write outcome is unknown")
        error.write_outcome_unknown = True
        error.action_receipt = receipt
        raise error

    robot.send_action = fail_inside_sdk_call
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )

    row = recorder.reservations[0].rows[0]
    assert result.actions == 0
    assert row["attempted_action"] == row["proposed_action"]
    assert row["attempted_action"]["Motor_0.pos"] == 0.0
    assert row["feedback_check"]["receipt_attempted_action"]["Motor_0.pos"] == 0.5
    assert "invalid_ambiguous_write_receipt" in row["feedback_check"][
        "receipt_violations"
    ]


def test_typed_confirmed_write_with_malformed_receipt_keeps_frames() -> None:
    robot = _Robot()

    def send_then_fail(action):
        robot.writes.append(dict(action))
        robot.last_receipt = SimpleNamespace(command_sequence="damaged")
        error = RuntimeError("feedback failed after confirmed write")
        error.write_outcome_confirmed = True
        error.action_receipt = robot.last_receipt
        raise error

    robot.send_action = send_then_fail
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )

    reservation = recorder.reservations[0]
    assert result.actions == 1
    assert reservation.cancelled is False
    assert reservation.rows[0]["sent_action"] == robot.writes[0]
    assert reservation.rows[0]["feedback_check"]["receipt_violations"] == [
        "invalid_post_write_failure_receipt"
    ]


@pytest.mark.parametrize(
    ("outcome", "signal"),
    [
        ("unknown", KeyboardInterrupt("operator stop")),
        ("confirmed", SystemExit(23)),
    ],
)
def test_process_control_after_sdk_boundary_keeps_frames_and_escapes(
    outcome: str,
    signal: BaseException,
) -> None:
    robot = _Robot()

    def interrupt_after_sdk_boundary(action):
        receipt = SimpleNamespace()
        robot.last_receipt = receipt
        signal.action_receipt = receipt
        if outcome == "unknown":
            signal.write_outcome_unknown = True
        else:
            signal.write_outcome_confirmed = True
        raise signal

    robot.send_action = interrupt_after_sdk_boundary
    recorder = _Recorder()
    with pytest.raises(type(signal)) as raised:
        run_control_trial(
            _permit(),
            _candidate(),
            robot,
            _Runtime(),
            recorder,
            _Operator(),
            _Monitor(),
            index=0,
            trial_id="trial",
            condition=SimpleNamespace(to_dict=lambda: {}),
            clock=lambda: 1.0,
            clock_ns=lambda: 1_000_000_000,
            sleep=lambda seconds: None,
        )

    assert raised.value is signal
    if isinstance(signal, SystemExit):
        assert raised.value.code == 23
    reservation = recorder.reservations[0]
    assert reservation.cancelled is False
    assert len(reservation.rows) == 1
    assert reservation.rows[0]["write_outcome"] == outcome
    if outcome == "unknown":
        assert reservation.rows[0]["sent_action"] is None
    else:
        assert reservation.rows[0]["sent_action"] is not None


def test_prewrite_process_control_cancels_reservation_and_escapes() -> None:
    robot = _Robot()
    signal = KeyboardInterrupt("operator stop before SDK write")

    def interrupt_before_sdk_boundary(_action):
        raise signal

    robot.send_action = interrupt_before_sdk_boundary
    recorder = _Recorder()
    with pytest.raises(KeyboardInterrupt) as raised:
        run_control_trial(
            _permit(),
            _candidate(),
            robot,
            _Runtime(),
            recorder,
            _Operator(),
            _Monitor(),
            index=0,
            trial_id="trial",
            condition=SimpleNamespace(to_dict=lambda: {}),
            clock=lambda: 1.0,
            clock_ns=lambda: 1_000_000_000,
            sleep=lambda seconds: None,
        )

    assert raised.value is signal
    assert recorder.reservations[0].cancelled is True
    assert recorder.reservations[0].rows == []


def test_definite_prewrite_failure_releases_reserved_frames() -> None:
    robot = _Robot()

    def fail_before_sdk_call(_action):
        raise OSError("prewrite monitor unavailable")

    robot.send_action = fail_before_sdk_call
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )

    assert result.safety_events == ("feedback_loss",)
    assert result.actions == 0
    reservation = recorder.reservations[0]
    assert reservation.cancelled is True
    assert reservation.rows == []
    assert recorder.terminal[0]["actions_sent"] == 0
    assert recorder.terminal[0]["action_attempts"] == 0
    assert recorder.terminal[0]["ambiguous_write_attempts"] == 0
    assert "before the SDK motor-write call" in recorder.terminal[0]["detail"]


def test_confirmed_write_with_invalid_receipt_keeps_reserved_evidence() -> None:
    robot = _Robot()

    def send_then_return_without_receipt(action):
        robot.writes.append(dict(action))
        robot.last_receipt = None
        return dict(action)

    robot.send_action = send_then_return_without_receipt
    recorder = _Recorder()
    result = run_control_trial(
        _permit(),
        _candidate(),
        robot,
        _Runtime(),
        recorder,
        _Operator(),
        _Monitor(),
        index=0,
        trial_id="trial",
        condition=SimpleNamespace(to_dict=lambda: {}),
        clock=lambda: 1.0,
        clock_ns=lambda: 1_000_000_000,
        sleep=lambda seconds: None,
    )

    assert result.safety_events == ("feedback_loss",)
    assert result.actions == len(robot.writes) == 1
    reservation = recorder.reservations[0]
    assert reservation.cancelled is False
    assert len(reservation.rows) == 1
    row = reservation.rows[0]
    assert row["sent_action"] == robot.writes[0]
    assert row["feedback_action"] is None
    assert row["feedback_check"]["passed"] is False
    assert row["feedback_check"]["receipt_violations"] == [
        "invalid_post_write_receipt"
    ]
    assert recorder.terminal[0]["actions_sent"] == 1
