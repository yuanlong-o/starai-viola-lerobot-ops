from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import viola_ops.execute_ops as execute_ops
import viola_ops.policy_runtime as policy_runtime
import viola_ops.rollout_evidence as rollout_evidence
from viola_handoff import canonical_json_bytes
from viola_ops.errors import SafetyGateError, ValidationError
from viola_ops.execution import ACTION_KEYS, PhaseResult


class _StopBeforeHardware(RuntimeError):
    pass


def _permit() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        session_bundle_id="a" * 64,
        candidate_bundle_id="b" * 64,
        policy="act",
        phase="hold",
        trial="commissioning",
        speed_scale=1.0,
        operator="operator",
        setup_hashes={name: "c" * 64 for name in ("calibration", "camera", "robot", "reset")},
        absolute_limits={
            joint: ((0.0, 100.0) if joint == "gripper" else (-100.0, 100.0))
            for joint in (
                "Motor_0",
                "Motor_1",
                "Motor_2",
                "Motor_3",
                "Motor_4",
                "Motor_5",
                "gripper",
            )
        },
    )


def _identity() -> SimpleNamespace:
    return SimpleNamespace(
        role="pc_a",
        repository_commit="d" * 40,
        repository_clean=True,
    )


def _accepted_material() -> tuple[SimpleNamespace, SimpleNamespace]:
    session = SimpleNamespace(
        bundle_id="a" * 64,
        content_id="a" * 64,
        manifest={"experiment": "viola-eight-policy-v1"},
    )
    candidate_bundle = SimpleNamespace(bundle_id="b" * 64, content_id="b" * 64)
    candidate = SimpleNamespace(bundle=candidate_bundle)
    return session, candidate


def _stub_pre_hardware_checks(
    monkeypatch: pytest.MonkeyPatch, order: list[str]
) -> None:
    session, candidate = _accepted_material()

    def authorize(_request):
        order.append("gate")
        return _permit()

    monkeypatch.setattr(execute_ops, "authorize_motion", authorize)
    monkeypatch.setattr(
        execute_ops,
        "RuntimeIdentity",
        SimpleNamespace(capture=lambda **_kwargs: order.append("identity") or _identity()),
    )
    monkeypatch.setattr(
        execute_ops,
        "inspect_bundle",
        lambda *_args, **_kwargs: order.append("session-inspect") or session,
    )
    monkeypatch.setattr(
        policy_runtime,
        "inspect_candidate",
        lambda _path: order.append("candidate-inspect") or candidate,
    )


def _execute(tmp_path: Path, *, publisher):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return execute_ops.execute_command(
        Path("accepted-session"),
        candidate=Path("accepted-candidate"),
        phase="hold",
        trial="commissioning",
        repo_root=repo,
        handoff_root=tmp_path / "handoffs",
        evidence_root=tmp_path / "evidence",
        wandb_entity="test-entity",
        intent_publisher=publisher,
    )


def test_gate_failure_prevents_identity_intent_and_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def reject(_request):
        calls.append("gate")
        raise SafetyGateError("blockers remain")

    monkeypatch.setattr(execute_ops, "authorize_motion", reject)
    monkeypatch.setattr(
        execute_ops,
        "RuntimeIdentity",
        SimpleNamespace(capture=lambda **_kwargs: calls.append("identity")),
    )

    with pytest.raises(SafetyGateError, match="blockers remain"):
        _execute(
            tmp_path,
            publisher=lambda *_args, **_kwargs: calls.append("intent"),
        )
    assert calls == ["gate"]
    assert not (tmp_path / "evidence").exists()


def test_failed_online_intent_stops_before_hardware_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[tuple[str, int]] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    original_import = builtins.__import__

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append((name, level))
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", recording_import)

    def fail_intent(*_args, **_kwargs):
        order.append("intent")
        raise ValidationError("W&B unavailable")

    with pytest.raises(ValidationError, match="W&B unavailable"):
        _execute(tmp_path, publisher=fail_intent)

    assert order == ["gate", "identity", "session-inspect", "candidate-inspect", "intent"]
    imported_names = {name for name, _level in imports}
    assert "hardware" not in imported_names
    assert "viola_ops.hardware" not in imported_names


def test_finished_intent_receipt_exists_before_hardware_module_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)

    def finish_intent(run, **kwargs):
        if kwargs["job_type"] == "viola-policy-execution-intent":
            order.append("intent")
            assert kwargs["summary"] == {
                "operator_authorized": True,
                "hardware_imported": False,
                "hardware_connected": False,
            }
        else:
            order.append("failure")
            assert kwargs["job_type"] == "viola-policy-execution-failure"
            assert kwargs["summary"]["ready"] is False
        return run

    original_import = builtins.__import__

    def stop_on_hardware(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"hardware", "viola_ops.hardware"}:
            order.append("hardware-import")
            raise _StopBeforeHardware
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", stop_on_hardware)
    outcome = _execute(tmp_path, publisher=finish_intent)

    assert outcome.status == "execution_failed_not_ready"
    assert order == [
        "gate",
        "identity",
        "session-inspect",
        "candidate-inspect",
        "intent",
        "hardware-import",
        "failure",
    ]
    receipt_path = (
        tmp_path
        / "evidence"
        / "session-1"
        / "act"
        / "hold"
        / "commissioning"
        / "EXECUTION_INTENT_WANDB_SYNCED.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "operator_authorized_pre_hardware"
    assert receipt["binding"]["session_id"] == "session-1"
    assert receipt["binding"]["phase"] == "hold"
    assert receipt["wandb"]["url"].startswith("https://wandb.ai/")


def test_support_import_failure_after_intent_is_recoverable_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    original_import = builtins.__import__

    def fail_support_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        if name == "evidence" and level == 1:
            raise ImportError("evidence support unavailable")
        return original_import(name, globals, locals, fromlist, level)

    def publish(run, **kwargs):
        order.append(kwargs["job_type"])
        return run

    monkeypatch.setattr(builtins, "__import__", fail_support_import)
    outcome = _execute(tmp_path, publisher=publish)
    marker = execute_ops._load_execution_failure(
        outcome.material_root / "EXECUTION_FAILURE.json",
        permit=_permit(),
        identity=_identity(),
    )

    assert outcome.status == "execution_failed_not_ready"
    assert marker["stage"] == "support_import"
    assert marker["hardware_may_have_connected"] is False
    assert marker["motion_may_have_started"] is False
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports
    assert order[-2:] == [
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
    ]


def test_evidence_root_inside_checkout_is_rejected_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    called = False

    def gate(_request):
        nonlocal called
        called = True

    monkeypatch.setattr(execute_ops, "authorize_motion", gate)
    with pytest.raises(ValidationError, match="outside the Git worktree"):
        execute_ops.execute_command(
            Path("session"),
            candidate=Path("candidate"),
            phase="hold",
            trial="commissioning",
            repo_root=repo,
            handoff_root=tmp_path / "handoffs",
            evidence_root=repo / "evidence",
            wandb_entity="test-entity",
        )
    assert called is False


@pytest.mark.parametrize("trial", ["", "../escape", "/tmp/escape", "line\nbreak"])
def test_trial_must_be_one_safe_evidence_path_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trial: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    called = False

    def gate(_request):
        nonlocal called
        called = True

    monkeypatch.setattr(execute_ops, "authorize_motion", gate)
    with pytest.raises(ValidationError, match="path-safe component"):
        execute_ops.execute_command(
            Path("session"),
            candidate=Path("candidate"),
            phase="hold",
            trial=trial,
            repo_root=repo,
            handoff_root=tmp_path / "handoffs",
            evidence_root=tmp_path / "evidence",
            wandb_entity="test-entity",
        )
    assert called is False


def test_completed_local_result_retries_finalization_without_hardware_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    held = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
    result = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="hold",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=1.0,
        held_action=held,
        trials=(),
        terminal_event=None,
        terminal_reason=None,
    )
    material = (
        tmp_path / "evidence" / "session-1" / "act" / "hold" / "commissioning"
    )
    material.mkdir(parents=True)
    (material / "PHASE_RESULT.json").write_bytes(
        canonical_json_bytes(execute_ops._phase_result_payload(result))
    )

    def finish_intent(run, **_kwargs):
        order.append("intent")
        return run

    bundle = SimpleNamespace(path=tmp_path / "sealed", bundle_id="e" * 64)

    def seal(local_result, **kwargs):
        order.append("seal")
        assert local_result == result
        assert kwargs["evidence_factory"] is None
        return SimpleNamespace(bundle=bundle)

    monkeypatch.setattr(rollout_evidence, "seal_completed_phase", seal)
    original_import = builtins.__import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", record_import)
    outcome = _execute(tmp_path, publisher=finish_intent)

    assert outcome.bundle is bundle
    assert order == [
        "gate",
        "identity",
        "session-inspect",
        "candidate-inspect",
        "intent",
        "seal",
    ]
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports


def test_operator_setup_failure_is_terminal_non_ready_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)

    class FailedOperator:
        def __init__(self) -> None:
            raise RuntimeError("operator terminal failed\nwithout starting")

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", FailedOperator)

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    outcome = _execute(tmp_path, publisher=publish)
    material = outcome.material_root
    marker_path = material / "EXECUTION_FAILURE.json"
    marker = execute_ops._load_execution_failure(
        marker_path,
        permit=_permit(),
        identity=_identity(),
    )

    assert outcome.status == "execution_failed_not_ready"
    assert outcome.bundle is None
    assert marker["stage"] == "operator_setup"
    assert marker["error_type"] == "RuntimeError"
    assert marker["error_message"] == "operator terminal failed without starting"
    assert marker["hardware_may_have_connected"] is False
    assert marker["motion_may_have_started"] is False
    assert marker["ready"] is False
    assert marker["sealed"] is False
    assert marker["rerun_allowed"] is False
    assert jobs == [
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
    ]
    assert (material / "EXECUTION_FAILURE_WANDB_SYNCED.json").is_file()
    assert not (material / "PHASE_RESULT.json").exists()
    assert not list(tmp_path.rglob("READY"))


def test_failed_attempt_retry_republishes_stable_receipts_without_hardware_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    jobs: list[tuple[str, str]] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    material = tmp_path / "evidence" / "session-1" / "act" / "hold" / "commissioning"
    material.mkdir(parents=True)
    marker = execute_ops._execution_failure_payload(
        _permit(),
        identity=_identity(),
        stage="execution",
        error=RuntimeError("controller failed"),
    )
    (material / "EXECUTION_FAILURE.json").write_bytes(canonical_json_bytes(marker))

    def publish(run, **kwargs):
        jobs.append((kwargs["job_type"], run.run_id))
        return run

    class MustNotStart:
        def __init__(self) -> None:
            raise AssertionError("operator setup must not run during failure recovery")

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", MustNotStart)
    original_import = builtins.__import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", record_import)
    first = _execute(tmp_path, publisher=publish)
    receipt_path = material / "EXECUTION_FAILURE_WANDB_SYNCED.json"
    first_receipt = receipt_path.read_bytes()
    second = _execute(tmp_path, publisher=publish)

    assert first.status == second.status == "execution_failed_not_ready"
    assert first.bundle is second.bundle is None
    assert receipt_path.read_bytes() == first_receipt
    failure_runs = [run_id for job, run_id in jobs if job == "viola-policy-execution-failure"]
    assert len(failure_runs) == 2
    assert failure_runs[0] == failure_runs[1]
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports
    assert "evidence" not in imports
    assert "rollout_evidence" not in imports


@pytest.mark.parametrize(
    ("signal_type", "termination_kind"),
    [
        (KeyboardInterrupt, "keyboard_interrupt"),
        (SystemExit, "system_exit"),
    ],
)
def test_process_control_failure_is_recorded_then_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
    termination_kind: str,
) -> None:
    order: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)

    class InterruptedOperator:
        def __init__(self) -> None:
            raise signal_type("operator requested process stop")

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", InterruptedOperator)

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    with pytest.raises(signal_type, match="operator requested process stop"):
        _execute(tmp_path, publisher=publish)

    material = tmp_path / "evidence" / "session-1" / "act" / "hold" / "commissioning"
    marker = execute_ops._load_execution_failure(
        material / "EXECUTION_FAILURE.json",
        permit=_permit(),
        identity=_identity(),
    )
    assert marker["termination_kind"] == termination_kind
    assert marker["error_type"] == signal_type.__name__
    assert marker["ready"] is False
    assert jobs[-1] == "viola-policy-execution-failure"


def test_failure_payload_safely_normalizes_unprintable_exception_text() -> None:
    class CustomInterrupt(KeyboardInterrupt):
        def __str__(self) -> str:
            return "bad\ud800\x00message\ncontinued"

    marker = execute_ops._execution_failure_payload(
        _permit(),
        identity=_identity(),
        stage="execution",
        error=CustomInterrupt(),
    )

    assert marker["termination_kind"] == "keyboard_interrupt"
    assert marker["error_type"] == "KeyboardInterrupt"
    assert marker["error_message"] == "bad? message continued"
    canonical_json_bytes(marker)


def test_terminal_cleanup_cannot_mask_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)

    class OperatorWithBrokenClose:
        def close(self) -> None:
            raise RuntimeError("terminal cleanup failed")

    def interrupt_execution(*_args, **_kwargs):
        raise KeyboardInterrupt("stop now")

    import viola_ops.execution as execution

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", OperatorWithBrokenClose)
    monkeypatch.setattr(execution, "execute_phase", interrupt_execution)

    with pytest.raises(KeyboardInterrupt, match="stop now"):
        _execute(tmp_path, publisher=lambda run, **_kwargs: run)

    material = tmp_path / "evidence" / "session-1" / "act" / "hold" / "commissioning"
    marker = execute_ops._load_execution_failure(
        material / "EXECUTION_FAILURE.json",
        permit=_permit(),
        identity=_identity(),
    )
    assert marker["stage"] == "execution"
    assert marker["termination_kind"] == "keyboard_interrupt"
    assert marker["error_message"] == "stop now"
    assert marker["hardware_may_have_connected"] is True
    assert marker["motion_may_have_started"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"session_bundle_id": "e" * 64}, "session_bundle_id differs"),
        ({"repo_commit": "e" * 40}, "another repository revision"),
        ({"ready": True}, "must remain unsealed and not READY"),
        ({"stage": "finished"}, "unknown execution stage"),
        ({"termination_kind": "system_exit"}, "inconsistent system-exit fields"),
    ],
)
def test_failure_loader_rejects_tampered_terminal_marker(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    material = tmp_path / "session-1" / "act" / "hold" / "commissioning"
    material.mkdir(parents=True)
    marker = execute_ops._execution_failure_payload(
        _permit(),
        identity=_identity(),
        stage="operator_setup",
        error=RuntimeError("operator failed"),
    )
    marker.update(mutation)
    path = material / "EXECUTION_FAILURE.json"
    path.write_bytes(canonical_json_bytes(marker))

    with pytest.raises(ValidationError, match=message):
        execute_ops._load_execution_failure(path, permit=_permit(), identity=_identity())


def test_failure_loader_rejects_wrong_path_and_noncanonical_json(tmp_path: Path) -> None:
    marker = execute_ops._execution_failure_payload(
        _permit(),
        identity=_identity(),
        stage="operator_setup",
        error=RuntimeError("operator failed"),
    )
    wrong_material = tmp_path / "session-1" / "act" / "hold" / "different-trial"
    wrong_material.mkdir(parents=True)
    wrong_path = wrong_material / "EXECUTION_FAILURE.json"
    wrong_path.write_bytes(canonical_json_bytes(marker))
    with pytest.raises(ValidationError, match="material path differs"):
        execute_ops._load_execution_failure(
            wrong_path,
            permit=_permit(),
            identity=_identity(),
        )

    material = tmp_path / "session-1" / "act" / "hold" / "commissioning"
    material.mkdir(parents=True)
    path = material / "EXECUTION_FAILURE.json"
    path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    with pytest.raises(ValidationError, match="not canonical JSON"):
        execute_ops._load_execution_failure(path, permit=_permit(), identity=_identity())


def test_failure_loader_rejects_dirty_current_identity(tmp_path: Path) -> None:
    material = tmp_path / "session-1" / "act" / "hold" / "commissioning"
    material.mkdir(parents=True)
    marker = execute_ops._execution_failure_payload(
        _permit(),
        identity=_identity(),
        stage="operator_setup",
        error=RuntimeError("operator failed"),
    )
    path = material / "EXECUTION_FAILURE.json"
    path.write_bytes(canonical_json_bytes(marker))
    dirty_identity = SimpleNamespace(
        role="pc_a",
        repository_commit="d" * 40,
        repository_clean=False,
    )
    with pytest.raises(ValidationError, match="clean repository identity"):
        execute_ops._load_execution_failure(
            path,
            permit=_permit(),
            identity=dirty_identity,
        )


def test_local_phase_result_must_be_canonical_and_match_the_permit(tmp_path: Path) -> None:
    held = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
    result = PhaseResult(
        session_id="another-session",
        policy="act",
        phase="hold",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=1.0,
        held_action=held,
        trials=(),
        terminal_event=None,
        terminal_reason=None,
    )
    path = tmp_path / "PHASE_RESULT.json"
    path.write_bytes(canonical_json_bytes(execute_ops._phase_result_payload(result)))
    with pytest.raises(ValidationError, match="differs from the current motion permit"):
        execute_ops._load_phase_result(path, permit=_permit())

    path.write_text(json.dumps(execute_ops._phase_result_payload(result), indent=2))
    with pytest.raises(ValidationError, match="not canonical"):
        execute_ops._load_phase_result(path, permit=_permit())


def test_phase_result_round_trip_preserves_completed_hold(tmp_path: Path) -> None:
    held = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
    result = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="hold",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=1.0,
        held_action=held,
        trials=(),
        terminal_event=None,
        terminal_reason=None,
    )
    path = tmp_path / "PHASE_RESULT.json"
    path.write_bytes(canonical_json_bytes(execute_ops._phase_result_payload(result)))
    assert execute_ops._load_phase_result(path, permit=_permit()) == result


def test_phase_result_loader_rejects_wrong_trial_count_for_completed_motion(
    tmp_path: Path,
) -> None:
    permit = _permit()
    permit.phase = "shakedown"
    permit.speed_scale = 0.25
    value = {
        "schema_version": 1,
        "session_id": "session-1",
        "policy": "act",
        "phase": "shakedown",
        "started_at": "2026-08-13T01:00:00+00:00",
        "completed_at": "2026-08-13T01:00:01+00:00",
        "speed_scale": 0.25,
        "held_action": None,
        "terminal_event": None,
        "terminal_reason": None,
        "trials": [],
    }
    path = tmp_path / "PHASE_RESULT.json"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValidationError, match="completed shakedown requires exactly 2 trials"):
        execute_ops._load_phase_result(path, permit=permit)


def test_intent_retry_rejects_noncanonical_local_receipt(tmp_path: Path) -> None:
    material = tmp_path / "material"
    material.mkdir()
    session, candidate = _accepted_material()

    def publish(run, **_kwargs):
        return run

    receipt = execute_ops._publish_execution_intent(
        permit=_permit(),
        session=session,
        candidate=candidate.bundle,
        identity=_identity(),
        material=material,
        wandb_entity="test-entity",
        wandb_project="project",
        publisher=publish,
    )
    receipt.chmod(0o644)
    receipt.write_text(json.dumps(json.loads(receipt.read_text()), indent=2), encoding="utf-8")

    with pytest.raises(ValidationError, match="not canonical JSON"):
        execute_ops._publish_execution_intent(
            permit=_permit(),
            session=session,
            candidate=candidate.bundle,
            identity=_identity(),
            material=material,
            wandb_entity="test-entity",
            wandb_project="project",
            publisher=publish,
        )


def test_unsafe_publish_reloads_and_path_binds_marker(tmp_path: Path) -> None:
    permit = _permit()
    session, candidate = _accepted_material()
    material = tmp_path / "session-1" / "act" / "hold" / "commissioning"
    material.mkdir(parents=True)
    marker = {
        "schema_version": 1,
        "status": "unsafe_shakedown",
        "session_id": permit.session_id,
        "policy": permit.policy,
        "phase": permit.phase,
        "terminal_event": "feedback_loss",
        "terminal_reason": "feedback stopped",
        "ready": False,
        "blocker": "repo_b_rollout_trace_schema_mismatch",
        "note": "retained locally",
    }
    marker_path = material / "UNSAFE_NOT_SEALED.json"
    marker_path.write_bytes(canonical_json_bytes(marker))

    def publish(run, **_kwargs):
        return run

    execute_ops._publish_unsafe_outcome(
        marker=marker,
        marker_path=marker_path,
        permit=permit,
        session=session,
        candidate=candidate.bundle,
        identity=_identity(),
        material=material,
        wandb_entity="test-entity",
        wandb_project="project",
        publisher=publish,
    )

    receipt = material / "UNSAFE_WANDB_SYNCED.json"
    receipt.chmod(0o644)
    receipt.write_text(json.dumps(json.loads(receipt.read_text()), indent=2), encoding="utf-8")
    with pytest.raises(ValidationError, match="not canonical JSON"):
        execute_ops._publish_unsafe_outcome(
            marker=marker,
            marker_path=marker_path,
            permit=permit,
            session=session,
            candidate=candidate.bundle,
            identity=_identity(),
            material=material,
            wandb_entity="test-entity",
            wandb_project="project",
            publisher=publish,
        )

    with pytest.raises(ValidationError, match="outside its authorized material path"):
        execute_ops._publish_unsafe_outcome(
            marker=marker,
            marker_path=tmp_path / "elsewhere.json",
            permit=permit,
            session=session,
            candidate=candidate.bundle,
            identity=_identity(),
            material=material,
            wandb_entity="test-entity",
            wandb_project="project",
            publisher=publish,
        )
