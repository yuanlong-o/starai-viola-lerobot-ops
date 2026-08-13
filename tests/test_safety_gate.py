from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import viola_ops.safety as safety
from viola_ops.errors import SafetyGateError
from viola_ops.safety import (
    CANONICAL_TASK,
    GateRequest,
    JOINTS,
    MotionPermit,
    ResolvedSetup,
    authorize_motion,
    assert_permit_current,
    check_current_checkout,
    check_estop_freshness,
    check_phase_predecessors,
    check_session_shape,
    operator_challenge,
    validate_action,
)


def _protocol() -> dict:
    conditions = [
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
    order = [condition["condition_id"] for condition in conditions]
    return {
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
        "ordered_conditions": conditions,
        "schedule_order": order,
        "execution_order": order,
    }


def _session(now: datetime) -> dict:
    return {
        "schema_version": 1,
        "session_id": "session-1",
        "policy_bundle_id": "a" * 64,
        "policy_content_id": "a" * 64,
        "setup_hashes": {name: "b" * 64 for name in ("calibration", "camera", "robot", "reset")},
        "phase_permissions": ["hold", "shakedown", "scored"],
        "blockers": [],
        "operator": "operator",
        "task": CANONICAL_TASK,
        "session_inputs_binding": {
            "bundle_id": "d" * 64,
            "content_id": "d" * 64,
            "manifest_sha256": "e" * 64,
            "payload_sha256": "f" * 64,
            "setup_record_inventory_sha256": "0" * 64,
        },
        "source_session": {
            "sha256": "1" * 64,
            "created_at_utc": (now - timedelta(hours=2)).isoformat(),
            "benchmark_lineage_sha256": "2" * 64,
            "policy_bindings_sha256": "3" * 64,
            "policy_binding_sha256": "4" * 64,
            "physical_setup_binding_sha256": "5" * 64,
            "executor_binding_sha256": "6" * 64,
        },
        "executor": {
            "repository": "starai-viola-lerobot-ops",
            "repository_commit": "7" * 40,
            "entrypoint_sha256": "8" * 64,
            "attestation_sha256": "9" * 64,
            "reviewed_at_utc": (now - timedelta(hours=2)).isoformat(),
        },
        "estop": {
            "operator": "operator",
            "tested_at_utc": (now - timedelta(hours=1)).isoformat(),
            "attestation_sha256": "c" * 64,
        },
        "act_infrastructure_clearance": None,
        "trial_protocol": _protocol(),
    }


def _permit(now: datetime, *, forged: bool = False) -> MotionPermit:
    return MotionPermit(
        session_id="session-1",
        session_bundle_id="a" * 64,
        candidate_bundle_id="d" * 64,
        policy="act",
        phase="shakedown",
        trial="01",
        operator="operator",
        setup_hashes={name: "b" * 64 for name in ("calibration", "camera", "robot", "reset")},
        absolute_limits={joint: ((0.0, 100.0) if joint == "gripper" else (-100.0, 100.0)) for joint in JOINTS},
        max_step_deltas={joint: 4.0 for joint in JOINTS},
        speed_scale=0.25,
        issued_at=now,
        estop_tested_at=now - timedelta(hours=1),
        _nonce="nonce",
        _authority=object() if forged else safety._PERMIT_AUTHORITY,
    )


def test_session_shape_rejects_any_blocker() -> None:
    now = datetime.now(UTC)
    session = _session(now)
    check_session_shape(session)
    session["blockers"] = ["not reviewed"]
    with pytest.raises(SafetyGateError, match="contains blockers"):
        check_session_shape(session)


@pytest.mark.parametrize("blockers", [None, {}, (), [""], ["camera review pending"]])
def test_session_shape_requires_an_exact_empty_blocker_list(blockers: object) -> None:
    session = _session(datetime.now(UTC))
    session["blockers"] = blockers
    with pytest.raises(SafetyGateError, match="contains blockers"):
        check_session_shape(session)


def test_session_shape_rejects_a_missing_blocker_field() -> None:
    session = _session(datetime.now(UTC))
    del session["blockers"]
    with pytest.raises(SafetyGateError, match=r"missing=.*blockers"):
        check_session_shape(session)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("ordered_conditions", []),
        ("schedule_order", ["robustness_04"]),
        ("execution_order", ["nominal_01"] * 10),
    ],
)
def test_session_shape_rejects_noncanonical_trial_order(
    field: str, replacement: object
) -> None:
    session = _session(datetime.now(UTC))
    session["trial_protocol"][field] = replacement
    with pytest.raises(SafetyGateError, match="conditions|order"):
        check_session_shape(session)


@pytest.mark.parametrize("age", [timedelta(hours=24), timedelta(days=5)])
def test_estop_must_be_younger_than_24_hours(age: timedelta) -> None:
    now = datetime.now(UTC)
    session = _session(now)
    session["estop"]["tested_at_utc"] = (now - age).isoformat()
    with pytest.raises(SafetyGateError, match="24 hours"):
        check_estop_freshness(session, now)


def test_future_estop_is_rejected() -> None:
    now = datetime.now(UTC)
    session = _session(now)
    session["estop"]["tested_at_utc"] = (now + timedelta(seconds=1)).isoformat()
    with pytest.raises(SafetyGateError, match="future-dated"):
        check_estop_freshness(session, now)


def test_estop_must_belong_to_the_rollout_operator() -> None:
    now = datetime.now(UTC)
    session = _session(now)
    session["estop"]["operator"] = "someone-else"
    with pytest.raises(SafetyGateError, match="differs from tested E-stop owner"):
        check_estop_freshness(session, now)


def test_forged_and_expired_permits_are_rejected() -> None:
    now = datetime.now(UTC)
    with pytest.raises(SafetyGateError, match="issued by the motion gate"):
        assert_permit_current(_permit(now, forged=True), now=now)
    expired = _permit(now)
    object.__setattr__(expired, "estop_tested_at", now - timedelta(hours=24))
    with pytest.raises(SafetyGateError, match="expired"):
        assert_permit_current(expired, now=now)

    future = _permit(now)
    object.__setattr__(future, "estop_tested_at", now + timedelta(microseconds=1))
    with pytest.raises(SafetyGateError, match="future-dated"):
        assert_permit_current(future, now=now)


@pytest.mark.parametrize(
    "head,status,message",
    [
        ("b" * 40, "", "differs from reviewed executor"),
        ("a" * 40, "?? unreviewed.py", "clean Repo-A worktree"),
    ],
)
def test_checkout_gate_rejects_wrong_or_dirty_revision(
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    status: str,
    message: str,
) -> None:
    session = _session(datetime.now(UTC))
    session["executor"] = {"repository_commit": "a" * 40}

    def fake_git(_root, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return status
        raise AssertionError(arguments)

    monkeypatch.setattr(safety, "_git", fake_git)
    with pytest.raises(SafetyGateError, match=message):
        check_current_checkout(session, Path.cwd())


def test_shakedown_speed_scales_step_limit_without_clamping() -> None:
    now = datetime.now(UTC)
    permit = _permit(now)
    reference = {f"{joint}.pos": 0.0 for joint in JOINTS}
    reference["gripper.pos"] = 50.0
    action = dict(reference)
    action["Motor_0.pos"] = 1.0
    assert validate_action(action, reference, permit)["Motor_0.pos"] == 1.0
    action["Motor_0.pos"] = 1.0001
    with pytest.raises(SafetyGateError, match="per-step"):
        validate_action(action, reference, permit)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0", None])
def test_malformed_action_never_becomes_a_clamped_action(bad: object) -> None:
    now = datetime.now(UTC)
    permit = _permit(now)
    reference = {f"{joint}.pos": 0.0 for joint in JOINTS}
    reference["gripper.pos"] = 50.0
    action = dict(reference)
    action["Motor_0.pos"] = bad
    with pytest.raises(SafetyGateError):
        validate_action(action, reference, permit)


def test_operator_challenge_binds_session_phase_and_trial() -> None:
    assert operator_challenge("session-1", "scored", "robustness_02") == (
        "ARM session-1 scored robustness_02"
    )


@pytest.mark.parametrize("trial", ["", "../escape", "/tmp/escape", "line\nbreak"])
def test_operator_challenge_rejects_ambiguous_trial_tokens(trial: str) -> None:
    with pytest.raises(SafetyGateError, match="path-safe component"):
        operator_challenge("session-1", "hold", trial)


class _ChallengeStream:
    def __init__(self, response: str) -> None:
        self.response = response
        self.output = ""
        self.closed = False

    def write(self, value: str) -> int:
        self.output += value
        return len(value)

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        return self.response

    def close(self) -> None:
        self.closed = True


def _stub_authorization_dependencies(
    monkeypatch: pytest.MonkeyPatch, now: datetime
) -> tuple[object, object]:
    session_payload = _session(now)
    session_payload["policy_bundle_id"] = "d" * 64
    session_payload["policy_content_id"] = "d" * 64
    candidate_payload = {
        "policy": "act",
        "queue_actions": 10,
        "task": CANONICAL_TASK,
    }

    class StubBundle:
        def __init__(self, bundle_id: str, payload_name: str, payload: dict) -> None:
            self.bundle_id = bundle_id
            self.content_id = bundle_id
            self.payload_name = payload_name
            self.payload = payload
            self.manifest = {
                "lineage": {"policy": "act"},
                "payload": {"files": [{"path": payload_name}]},
                "artifacts": [],
            }

        def payload_file(self, name: str):
            assert name == self.payload_name
            return name

    session_bundle = StubBundle("a" * 64, "rollout_session.json", session_payload)
    candidate_bundle = StubBundle("d" * 64, "policy_candidate.json", candidate_payload)
    monkeypatch.setattr(
        safety,
        "_accepted_bundle",
        lambda _path, kind, _permission, receiver_role: (
            session_bundle if kind == "rollout_session" else candidate_bundle
        ),
    )
    monkeypatch.setattr(
        safety,
        "_canonical_object",
        lambda path: (
            session_payload if path == "rollout_session.json" else candidate_payload
        ),
    )
    monkeypatch.setattr(safety, "check_current_checkout", lambda *_args: None)
    monkeypatch.setattr(safety, "check_session_manifest_lineage", lambda *_args: None)
    monkeypatch.setattr(safety, "_check_act_clearance_attachment", lambda *_args: None)
    monkeypatch.setattr(safety, "require_active_canonical_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        safety,
        "resolve_reviewed_setup",
        lambda *_args, **_kwargs: ResolvedSetup(
            setup_id="setup-1",
            robot_port="/dev/reviewed-robot",
            camera_configs={"front": {}, "up": {}},
            calibration_path=Path("/reviewed/calibration.json"),
            reset_protocol_path=Path("/reviewed/reset.json"),
            executor_entrypoint=Path("/reviewed/execution.py"),
            setup_hashes=session_payload["setup_hashes"],
            absolute_limits={
                joint: ((0.0, 100.0) if joint == "gripper" else (-100.0, 100.0))
                for joint in JOINTS
            },
            max_step_deltas={joint: 4.0 for joint in JOINTS},
        ),
    )
    return session_bundle, candidate_bundle


def test_motion_gate_requires_the_exact_tty_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    _stub_authorization_dependencies(monkeypatch, now)
    request = GateRequest(
        session_bundle=Path("session"),
        candidate_bundle=Path("candidate"),
        phase="hold",
        trial="commissioning",
        repository_root=Path.cwd(),
        handoff_root=Path("handoffs"),
        now=now,
    )
    wrong = _ChallengeStream("ARM session-1 hold another-trial\n")
    with pytest.raises(SafetyGateError, match="did not match"):
        authorize_motion(request, input_stream=wrong, terminal_check=lambda _stream: True)
    assert "ARM session-1 hold commissioning" in wrong.output
    assert wrong.closed is False

    exact = _ChallengeStream("ARM session-1 hold commissioning\n")
    permit = authorize_motion(request, input_stream=exact, terminal_check=lambda _stream: True)
    assert permit.allows(session_id="session-1", phase="hold", trial="commissioning")
    assert exact.closed is False


def test_scored_gate_rejects_a_spliced_hold_shakedown_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Bundle:
        def __init__(self, name: str, bundle_id: str, lineage: dict) -> None:
            self.name = name
            self.bundle_id = bundle_id
            self.content_id = bundle_id
            self.manifest = {"lineage": lineage}

        def payload_file(self, name: str) -> str:
            assert name == "rollout_evidence.json"
            return self.name

    session = Bundle("session", "1" * 64, {"policy": "act"})
    candidate = Bundle("candidate", "2" * 64, {"policy": "act"})
    hold = Bundle(
        "hold",
        "3" * 64,
        {
            "session_id": "session-1",
            "rollout_session_bundle_id": session.bundle_id,
            "rollout_session_content_id": session.content_id,
            "policy_candidate_bundle_id": candidate.bundle_id,
            "policy_candidate_content_id": candidate.content_id,
            "policy": "act",
            "phase": "hold",
        },
    )
    shakedown = Bundle(
        "shakedown",
        "4" * 64,
        {
            "session_id": "session-1",
            "rollout_session_bundle_id": session.bundle_id,
            "rollout_session_content_id": session.content_id,
            "policy_candidate_bundle_id": candidate.bundle_id,
            "policy_candidate_content_id": candidate.content_id,
            "policy": "act",
            "phase": "shakedown",
            "hold_rollout_evidence_bundle_id": "9" * 64,
            "hold_rollout_evidence_content_id": "9" * 64,
        },
    )
    common = {
        "session_id": "session-1",
        "session_bundle_id": session.bundle_id,
        "session_content_id": session.content_id,
        "policy_bundle_id": candidate.bundle_id,
        "policy_content_id": candidate.content_id,
        "policy": "act",
        "status": "completed",
    }
    payloads = {
        "hold": {
            **common,
            "phase": "hold",
            "prior_phase_bundles": {"hold": None, "shakedown": None},
        },
        "shakedown": {
            **common,
            "phase": "shakedown",
            "prior_phase_bundles": {
                "hold": {"bundle_id": "9" * 64, "content_id": "9" * 64},
                "shakedown": None,
            },
        },
    }
    monkeypatch.setattr(
        safety,
        "_accepted_bundle",
        lambda path, *_args, **_kwargs: hold if path == Path("hold") else shakedown,
    )
    monkeypatch.setattr(safety, "_canonical_object", lambda name: payloads[name])
    monkeypatch.setattr(safety, "require_active_canonical_source", lambda *_args, **_kwargs: None)

    with pytest.raises(SafetyGateError, match="broken predecessor chain"):
        check_phase_predecessors(
            "scored",
            Path("hold"),
            Path("shakedown"),
            session=session,
            candidate=candidate,
            handoff_root=Path("handoffs"),
        )
