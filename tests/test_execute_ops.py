from __future__ import annotations

import builtins
import fcntl
import json
import os
import pty
import select
import stat
import subprocess
import sys
import threading
import tty
from pathlib import Path
from types import SimpleNamespace

import pytest

import viola_ops.execute_ops as execute_ops
import viola_ops.policy_runtime as policy_runtime
import viola_ops.rollout_evidence as rollout_evidence
from viola_handoff import canonical_json_bytes
from viola_ops.errors import SafetyGateError, ValidationError
from viola_ops.execution import (
    ACTION_KEYS,
    PhaseResult,
    TrialOutcome,
    TrialResult,
    shakedown_conditions,
)


class _StopBeforeHardware(RuntimeError):
    pass


_TEST_ROBOT_PORT = f"/dev/viola-test-{os.getpid()}"


def _permit(
    *,
    trial: str = "commissioning",
    phase: str = "hold",
    robot_port: str = _TEST_ROBOT_PORT,
) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        session_bundle_id="a" * 64,
        candidate_bundle_id="b" * 64,
        policy="act",
        phase=phase,
        trial=trial,
        speed_scale=1.0,
        operator="operator",
        robot_port=robot_port,
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
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _accepted_material() -> tuple[SimpleNamespace, SimpleNamespace]:
    session = SimpleNamespace(
        bundle_id="a" * 64,
        content_id="a" * 64,
        manifest={"experiment": "viola-eight-policy-v1"},
    )
    candidate_bundle = SimpleNamespace(bundle_id="b" * 64, content_id="b" * 64)
    candidate = SimpleNamespace(bundle=candidate_bundle, runtime_binding={})
    return session, candidate


class _TrackedRuntimeSnapshot:
    def __init__(self, candidate: object) -> None:
        self.candidate = candidate
        self.active = False
        self.closed = False
        self.verify_calls = 0

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_args) -> None:
        self.active = False
        self.closed = True

    def verify(self) -> None:
        assert self.active
        self.verify_calls += 1


def _stub_pre_hardware_checks(
    monkeypatch: pytest.MonkeyPatch, order: list[str]
) -> None:
    session, candidate = _accepted_material()

    def authorize(_request):
        order.append("gate")
        return _permit(trial=_request.trial, phase=_request.phase)

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
    monkeypatch.setattr(
        execute_ops,
        "revalidate_motion",
        lambda *_args, **_kwargs: (
            order.append("final-revalidate") or session,
            candidate.bundle,
        ),
    )


def _execute(
    tmp_path: Path,
    *,
    publisher,
    trial: str = "commissioning",
    phase: str = "hold",
):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return execute_ops.execute_command(
        Path("accepted-session"),
        candidate=Path("accepted-candidate"),
        phase=phase,
        trial=trial,
        repo_root=repo,
        handoff_root=tmp_path / "handoffs",
        evidence_root=tmp_path / "evidence",
        wandb_entity="test-entity",
        intent_publisher=publisher,
    )


def test_operator_poll_buffers_partial_pty_input_without_blocking() -> None:
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    terminal = execute_ops._OperatorTerminal(os.ttyname(slave_fd))
    os.close(slave_fd)
    operator = execute_ops.InteractiveTrialOperator.__new__(
        execute_ops.InteractiveTrialOperator
    )
    operator._terminal = terminal
    operator._pending_event = None
    operator._input_buffer = bytearray()
    original_flags = fcntl.fcntl(terminal.fileno(), fcntl.F_GETFL)
    try:
        os.write(master_fd, b"COLL")
        assert operator.abort_requested() is False
        assert operator.event() is None
        assert bytes(operator._input_buffer) == b"COLL"
        assert fcntl.fcntl(terminal.fileno(), fcntl.F_GETFL) == original_flags

        os.write(master_fd, b"ISION\n")
        assert operator.event() == "collision"
        assert operator._input_buffer == bytearray()
        assert fcntl.fcntl(terminal.fileno(), fcntl.F_GETFL) == original_flags
    finally:
        operator.close()
        os.close(master_fd)


def test_trial_start_discards_typeahead_from_before_the_prompt() -> None:
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    terminal = execute_ops._OperatorTerminal(os.ttyname(slave_fd))
    os.close(slave_fd)
    operator = execute_ops.InteractiveTrialOperator.__new__(
        execute_ops.InteractiveTrialOperator
    )
    operator._terminal = terminal
    operator._pending_event = None
    operator._input_buffer = bytearray()
    completed = threading.Event()
    failure: list[BaseException] = []
    trial_id = "local-act-session-local_act-01"

    def prepare() -> None:
        try:
            operator.prepare_trial(
                "local-act-session",
                trial_id,
                {
                    "condition_id": "local_act_01",
                    "stratum": "local_act",
                    "blue_axis": None,
                    "blue_offset_mm": 0.0,
                    "red_axis": None,
                    "red_offset_mm": 0.0,
                },
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            failure.append(exc)
        finally:
            completed.set()

    try:
        # This line was typed before the START prompt and must be discarded.
        os.write(master_fd, f"START {trial_id}\n".encode())
        worker = threading.Thread(target=prepare)
        worker.start()
        readable, _, _ = select.select([master_fd], [], [], 2.0)
        assert readable
        assert b"Type exactly" in os.read(master_fd, 8192)
        assert completed.wait(0.1) is False

        os.write(master_fd, f"START {trial_id}\n".encode())
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert failure == []
    finally:
        operator.close()
        os.close(master_fd)


def test_trial_condition_prompt_is_plain_language() -> None:
    rendered = execute_ops._render_trial_condition(
        {
            "condition_id": "robustness_01",
            "stratum": "robustness",
            "blue_axis": "pad_x",
            "blue_offset_mm": -25.0,
            "red_axis": "pad_y",
            "red_offset_mm": 25.0,
        }
    )

    assert rendered.splitlines() == [
        "  Condition: robustness_01 (robustness)",
        "  Blue cube: -25 mm along pad_x",
        "  Red cube: +25 mm along pad_y",
    ]
    assert "{" not in rendered


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


def test_robot_execution_lease_contends_across_processes() -> None:
    script = (
        "import sys\n"
        "from viola_ops.execute_ops import _execution_lease\n"
        f"with _execution_lease({_TEST_ROBOT_PORT!r}):\n"
        "    print('lease-held', flush=True)\n"
        "    sys.stdin.read(1)\n"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    try:
        readable, _, _ = select.select([process.stdout], [], [], 5.0)
        if not readable:
            process.kill()
            process.wait(timeout=5.0)
            error = process.stderr.read() if process.stderr is not None else ""
            pytest.fail(f"lease holder did not become ready: {error}")
        assert process.stdout.readline().strip() == "lease-held"
        with pytest.raises(SafetyGateError, match="already holds the execution lease"):
            with execute_ops._execution_lease(_TEST_ROBOT_PORT):
                pytest.fail("a second process acquired the same robot lease")
    finally:
        if process.poll() is None:
            process.stdin.write("x")
            process.stdin.close()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def test_robot_execution_lease_treats_device_aliases_as_one_robot() -> None:
    with execute_ops._execution_lease("/dev/null"):
        with pytest.raises(SafetyGateError, match="already holds the execution lease"):
            with execute_ops._execution_lease("/dev/../dev/null"):
                pytest.fail("a device alias acquired a second robot lease")


def test_robot_execution_lease_survives_absent_to_present_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_port = "/dev/viola-hotplug-test"
    original_stat = execute_ops.os.stat
    original_realpath = execute_ops.os.path.realpath
    inspections = 0
    resolutions = 0

    def changing_realpath(path):
        nonlocal resolutions
        if os.fspath(path) != robot_port:
            return original_realpath(path)
        resolutions += 1
        return robot_port if resolutions == 1 else "/dev/ttyUSB0"

    def changing_stat(path, *args, **kwargs):
        nonlocal inspections
        if os.fspath(path) != robot_port:
            return original_stat(path, *args, **kwargs)
        inspections += 1
        if inspections == 1:
            raise FileNotFoundError(robot_port)
        return SimpleNamespace(
            st_mode=stat.S_IFCHR | 0o660,
            st_rdev=os.makedev(188, 0),
        )

    monkeypatch.setattr(execute_ops.os, "stat", changing_stat)
    monkeypatch.setattr(execute_ops.os.path, "realpath", changing_realpath)

    with execute_ops._execution_lease(robot_port):
        with pytest.raises(SafetyGateError, match="already holds the execution lease"):
            with execute_ops._execution_lease(robot_port):
                pytest.fail("a hot-plug transition acquired a second robot lease")

    # A failed multi-key acquisition released every key it briefly held.
    with execute_ops._execution_lease(robot_port):
        pass


@pytest.mark.parametrize(
    ("second_trial", "blocked_during_job"),
    [
        ("commissioning", "viola-policy-execution-intent"),
        ("different-trial", "viola-policy-execution-failure"),
    ],
)
def test_concurrent_execution_for_same_robot_never_reaches_a_second_publisher_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_trial: str,
    blocked_during_job: str,
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    entered = threading.Event()
    release = threading.Event()
    publisher_threads: list[str] = []
    hardware_threads: list[str] = []
    outcomes: list[object] = []
    failures: list[BaseException] = []

    class Operator:
        def close(self) -> None:
            pass

    def publish(run, **kwargs):
        publisher_threads.append(threading.current_thread().name)
        if kwargs["job_type"] == blocked_during_job:
            entered.set()
            if not release.wait(timeout=5.0):
                raise AssertionError("test did not release the first execution")
        return run

    def stop_without_hardware(*_args, **_kwargs):
        hardware_threads.append(threading.current_thread().name)
        raise RuntimeError("disconnected execution test stop")

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", Operator)
    import viola_ops.execution as execution

    monkeypatch.setattr(execution, "execute_phase", stop_without_hardware)

    def run_first() -> None:
        try:
            outcomes.append(_execute(tmp_path, publisher=publish))
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            failures.append(exc)

    first = threading.Thread(target=run_first, name="first-execution")
    first.start()
    try:
        assert entered.wait(timeout=5.0)
        with pytest.raises(SafetyGateError, match="already holds the execution lease"):
            _execute(tmp_path, publisher=publish, trial=second_trial)
    finally:
        release.set()
        first.join(timeout=5.0)

    assert not first.is_alive()
    assert failures == []
    assert len(outcomes) == 1
    assert getattr(outcomes[0], "status") == "execution_failed_not_ready"
    assert set(publisher_threads) == {"first-execution"}
    assert hardware_threads == ["first-execution"]
    if second_trial != "commissioning":
        second_material = (
            tmp_path / "evidence" / "session-1" / "act" / "hold" / second_trial
        )
        assert not second_material.exists()


def test_preexisting_empty_attempt_is_refused_before_intent_or_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    material = tmp_path / "evidence" / "session-1" / "act" / "hold" / "commissioning"
    material.mkdir(parents=True)

    with pytest.raises(ValidationError, match="incomplete live attempt"):
        _execute(
            tmp_path,
            publisher=lambda *_args, **_kwargs: order.append("intent"),
        )

    assert "intent" not in order
    assert list(material.iterdir()) == []


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
        "identity",
        "final-revalidate",
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


def test_final_revalidation_failure_records_and_recovers_without_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)

    def revoked(*_args, **_kwargs):
        order.append("final-revalidate")
        raise SafetyGateError("candidate was revoked after operator authorization")

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        order.append(kwargs["job_type"])
        return run

    original_import = builtins.__import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(execute_ops, "revalidate_motion", revoked)
    monkeypatch.setattr(builtins, "__import__", record_import)
    outcome = _execute(tmp_path, publisher=publish)

    assert outcome.status == "execution_failed_not_ready"
    assert jobs == [
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
    ]
    assert order[-4:] == [
        "viola-policy-execution-intent",
        "identity",
        "final-revalidate",
        "viola-policy-execution-failure",
    ]
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports
    marker = json.loads(
        (outcome.material_root / "EXECUTION_FAILURE.json").read_text(encoding="utf-8")
    )
    assert marker["stage"] == "final_revalidation"
    assert marker["hardware_may_have_connected"] is False
    receipt = outcome.material_root / "EXECUTION_FAILURE_WANDB_SYNCED.json"
    assert receipt.is_file()
    receipt_bytes = receipt.read_bytes()

    retry = _execute(tmp_path, publisher=publish)
    assert retry.status == "execution_failed_not_ready"
    assert jobs == [
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
    ]
    assert receipt.read_bytes() == receipt_bytes
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports


def test_post_start_callback_revalidates_and_blocks_before_motion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    checks = 0

    def revoke_after_start(*_args, **_kwargs):
        nonlocal checks
        checks += 1
        order.append(f"revalidate-{checks}")
        if checks == 3:
            raise SafetyGateError("session revoked after START")
        return _accepted_material()[0], _accepted_material()[1].bundle

    class Operator:
        def close(self) -> None:
            order.append("operator-close")

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    monkeypatch.setattr(execute_ops, "revalidate_motion", revoke_after_start)
    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", Operator)
    import viola_ops.execution as execution

    def simulate_successful_start(*_args, **kwargs):
        order.append("start")
        kwargs["revalidate_authority"]()
        order.append("motion")
        pytest.fail("motion reached after revocation")

    monkeypatch.setattr(
        execution,
        "execute_phase",
        simulate_successful_start,
    )

    outcome = _execute(tmp_path, publisher=publish)
    marker = json.loads(
        (outcome.material_root / "EXECUTION_FAILURE.json").read_text(encoding="utf-8")
    )
    assert marker["stage"] == "execution_revalidation"
    assert marker["hardware_may_have_connected"] is True
    assert marker["motion_may_have_started"] is True
    assert checks == 3
    assert order.index("start") < order.index("revalidate-3")
    assert "motion" not in order
    assert order[-1] == "operator-close"
    assert jobs == [
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
    ]


def test_post_disconnect_revocation_preserves_result_and_records_non_ready_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    checks = 0

    def revoke_after_disconnect(*_args, **_kwargs):
        nonlocal checks
        checks += 1
        if checks == 3:
            raise SafetyGateError("session was revoked while execution was active")
        session, candidate = _accepted_material()
        return session, candidate.bundle

    class Operator:
        def close(self) -> None:
            order.append("operator-close")

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

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    def disconnected_execution(*_args, **_kwargs):
        order.append("execution-disconnected")
        return result

    def must_not_seal(*_args, **_kwargs):
        pytest.fail("revoked authority reached READY finalization")

    monkeypatch.setattr(execute_ops, "revalidate_motion", revoke_after_disconnect)
    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", Operator)
    import viola_ops.execution as execution

    monkeypatch.setattr(execution, "execute_phase", disconnected_execution)
    monkeypatch.setattr(rollout_evidence, "seal_completed_phase", must_not_seal)

    first = _execute(tmp_path, publisher=publish)
    material = first.material_root
    result_bytes = (material / "PHASE_RESULT.json").read_bytes()
    marker = json.loads((material / "EXECUTION_FAILURE.json").read_bytes())

    assert first.status == "execution_failed_not_ready"
    assert first.bundle is None
    assert checks == 3
    assert marker["stage"] == "post_execution_revalidation"
    assert marker["ready"] is False
    assert marker["hardware_may_have_connected"] is True
    assert result_bytes == canonical_json_bytes(execute_ops._phase_result_payload(result))
    assert not list(tmp_path.rglob("READY"))

    # A retry republishes the terminal failure without importing or repeating
    # the hardware path, and it leaves the raw result byte-for-byte unchanged.
    second = _execute(tmp_path, publisher=publish)
    assert second.status == "execution_failed_not_ready"
    assert second.bundle is None
    assert checks == 3
    assert (material / "PHASE_RESULT.json").read_bytes() == result_bytes
    assert jobs == [
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
        "viola-policy-execution-intent",
        "viola-policy-execution-failure",
    ]


def test_only_post_disconnect_revalidation_allows_a_consumed_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    allow_consumed_calls: list[bool] = []

    def revalidate(*_args, **kwargs):
        allow_consumed_calls.append(kwargs.get("allow_consumed", False))
        session, candidate = _accepted_material()
        return session, candidate.bundle

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

    class Operator:
        def close(self) -> None:
            pass

    monkeypatch.setattr(execute_ops, "revalidate_motion", revalidate)
    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", Operator)
    import viola_ops.execution as execution

    monkeypatch.setattr(execution, "execute_phase", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        rollout_evidence,
        "seal_completed_phase",
        lambda *_args, **_kwargs: SimpleNamespace(
            bundle=SimpleNamespace(bundle_id="e" * 64)
        ),
    )

    _execute(tmp_path, publisher=lambda run, **_kwargs: run)

    assert allow_consumed_calls == [False, False, True]


def test_execution_revalidation_rejects_same_runtime_hashes_from_another_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    original = _accepted_material()[1]
    replacement_bundle = SimpleNamespace(
        bundle_id="e" * 64,
        content_id="e" * 64,
    )
    replacement = SimpleNamespace(bundle=replacement_bundle, runtime_binding={})
    inspections = 0

    def changing_candidate(_path: Path) -> SimpleNamespace:
        nonlocal inspections
        inspections += 1
        return original if inspections == 1 else replacement

    monkeypatch.setattr(policy_runtime, "inspect_candidate", changing_candidate)

    class Operator:
        def close(self) -> None:
            pass

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", Operator)
    outcome = _execute(tmp_path, publisher=lambda run, **_kwargs: run)

    marker = json.loads(
        (outcome.material_root / "EXECUTION_FAILURE.json").read_text(encoding="utf-8")
    )
    assert marker["stage"] == "execution_revalidation"
    assert "policy candidate runtime material" in marker["error_message"]
    assert outcome.bundle is None


def test_execute_uses_verifies_and_cleans_private_runtime_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    accepted = _accepted_material()[1]
    runtime_candidate = SimpleNamespace(
        bundle=accepted.bundle,
        bundle_id=accepted.bundle.bundle_id,
        policy="act",
        runtime_binding=accepted.runtime_binding,
    )
    snapshot = _TrackedRuntimeSnapshot(runtime_candidate)
    snapshotted: list[object] = []

    def make_snapshot(candidate: object) -> _TrackedRuntimeSnapshot:
        snapshotted.append(candidate)
        return snapshot

    monkeypatch.setattr(
        policy_runtime,
        "_snapshot_candidate_runtime",
        make_snapshot,
    )

    class Operator:
        def close(self) -> None:
            order.append("operator-close")

    monkeypatch.setattr(execute_ops, "InteractiveTrialOperator", Operator)
    import viola_ops.execution as execution

    def fail_after_start(_permit, candidate, **kwargs):
        assert candidate is runtime_candidate
        assert snapshot.active
        kwargs["revalidate_authority"]()
        assert snapshot.verify_calls == 2
        raise RuntimeError("disconnected execution stop")

    monkeypatch.setattr(execution, "execute_phase", fail_after_start)
    outcome = _execute(
        tmp_path,
        publisher=lambda run, **_kwargs: run,
        phase="shakedown",
        trial="supervised-shakedown",
    )

    assert outcome.status == "execution_failed_not_ready"
    assert snapshot.verify_calls == 2
    assert snapshot.closed
    assert not snapshot.active
    assert len(snapshotted) == 1


def test_replaced_evidence_root_cannot_redirect_failure_writes_into_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    captures = 0

    def capture(**_kwargs):
        nonlocal captures
        captures += 1
        if captures == 2:
            evidence = tmp_path / "evidence"
            evidence.rename(tmp_path / "preserved-evidence")
            evidence.symlink_to(tmp_path / "repo", target_is_directory=True)
        return _identity()

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    original_import = builtins.__import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        execute_ops,
        "RuntimeIdentity",
        SimpleNamespace(capture=capture),
    )
    monkeypatch.setattr(builtins, "__import__", record_import)
    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        _execute(tmp_path, publisher=publish)

    assert jobs == ["viola-policy-execution-intent"]
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports
    assert not list((tmp_path / "repo").rglob("EXECUTION_FAILURE.json"))


def test_same_path_attempt_replacement_is_rejected_before_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    captures = 0
    material = tmp_path / "evidence/session-1/act/hold/commissioning"
    preserved = material.with_name("commissioning-preserved")

    def capture(**_kwargs):
        nonlocal captures
        captures += 1
        if captures == 2:
            material.rename(preserved)
            material.mkdir()
        return _identity()

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    original_import = builtins.__import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        execute_ops,
        "RuntimeIdentity",
        SimpleNamespace(capture=capture),
    )
    monkeypatch.setattr(builtins, "__import__", record_import)

    with pytest.raises(SafetyGateError, match="live evidence path changed"):
        _execute(tmp_path, publisher=publish)

    assert jobs == ["viola-policy-execution-intent"]
    assert (preserved / "EXECUTION_INTENT_WANDB_SYNCED.json").is_file()
    assert list(material.iterdir()) == []
    assert not list(tmp_path.rglob("EXECUTION_FAILURE.json"))
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports


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


def test_handoff_root_that_can_dirty_checkout_is_rejected_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    called = False

    def gate(_request):
        nonlocal called
        called = True

    monkeypatch.setattr(execute_ops, "authorize_motion", gate)
    with pytest.raises(ValidationError, match="live handoff_root.*outside"):
        execute_ops.execute_command(
            Path("session"),
            candidate=Path("candidate"),
            phase="hold",
            trial="commissioning",
            repo_root=repo,
            handoff_root=repo / "handoffs",
            evidence_root=tmp_path / "evidence",
            wandb_entity="test-entity",
        )
    assert called is False


def test_symlinked_live_root_is_rejected_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    called = False

    def gate(_request):
        nonlocal called
        called = True

    monkeypatch.setattr(execute_ops, "authorize_motion", gate)
    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        execute_ops.execute_command(
            Path("session"),
            candidate=Path("candidate"),
            phase="hold",
            trial="commissioning",
            repo_root=repo,
            handoff_root=tmp_path / "handoffs",
            evidence_root=linked / "rollout",
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
        assert callable(kwargs["identity_capture"])
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
        "identity",
        "final-revalidate",
        "candidate-inspect",
        "seal",
    ]
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports


def test_unsafe_shakedown_result_seals_without_hardware_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    imports: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    permit = _permit(trial="supervised-shakedown")
    permit.phase = "shakedown"
    permit.speed_scale = 0.25
    condition = shakedown_conditions()[0].to_dict()
    trial = TrialResult(
        trial_id="session-1-shakedown-01",
        index=0,
        condition=condition,
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        duration_sec=1.0,
        actions=0,
        replans=0,
        inference_latency_ms=(),
        control_latency_ms=(),
        outcome=TrialOutcome(
            False,
            "safety_abort",
            "feedback_loss",
            None,
            None,
            None,
            0.0,
        ),
        safety_events=("feedback_loss",),
        trace_path="trials/trial-00.jsonl",
        front_video_path="videos/trial-00-front.mp4",
        up_video_path="videos/trial-00-up.mp4",
    )
    result = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event="feedback_loss",
        terminal_reason="motor write outcome was unknown",
    )
    output = tmp_path / "evidence"
    handoff = tmp_path / "handoffs"
    material = output / "session-1/act/shakedown/supervised-shakedown"
    (material / "motion_record/safety_traces").mkdir(parents=True)
    (material / "motion_record/videos").mkdir()
    (material / "PHASE_RESULT.json").write_bytes(
        canonical_json_bytes(execute_ops._phase_result_payload(result))
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    gate_request = SimpleNamespace(
        repository_root=repository,
        session_bundle=Path("accepted-session"),
        candidate_bundle=Path("accepted-candidate"),
        prior_hold_bundle=Path("accepted-hold"),
        prior_shakedown_bundle=None,
    )
    bundle = SimpleNamespace(path=tmp_path / "sealed", bundle_id="e" * 64)

    def seal_unsafe(local_result, **kwargs):
        order.append("seal-unsafe")
        assert local_result == result
        assert kwargs["evidence_factory"].root == material / "motion_record"
        assert kwargs["prior_hold"] is not None
        assert callable(kwargs["identity_capture"])
        return SimpleNamespace(bundle=bundle)

    monkeypatch.setattr(rollout_evidence, "seal_unsafe_phase", seal_unsafe)
    original_import = builtins.__import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", record_import)
    outcome = execute_ops._execute_authorized_command(
        gate_request,
        permit,
        output=output,
        handoff=handoff,
        evidence_root=output,
        handoff_root=handoff,
        wandb_entity="test-entity",
        wandb_project="project",
        intent_publisher=lambda run, **_kwargs: run,
    )

    assert outcome.status == "unsafe_shakedown"
    assert outcome.bundle is bundle
    assert "cannot authorize another motion phase" in outcome.note
    assert "seal-unsafe" in order
    assert "hardware" not in imports
    assert "viola_ops.hardware" not in imports
    assert not (material / "UNSAFE_NOT_SEALED.json").exists()


def test_zero_frame_unsafe_shakedown_records_non_ready_without_hardware_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    jobs: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    permit = _permit(trial="supervised-shakedown")
    permit.phase = "shakedown"
    permit.speed_scale = 0.25
    condition = shakedown_conditions()[0].to_dict()
    trial = TrialResult(
        trial_id="session-1-shakedown-01",
        index=0,
        condition=condition,
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        duration_sec=1.0,
        actions=0,
        replans=0,
        inference_latency_ms=(),
        control_latency_ms=(),
        outcome=TrialOutcome(
            False,
            "safety_abort",
            "operator_abort",
            None,
            None,
            None,
            0.0,
        ),
        safety_events=("operator_abort",),
        trace_path="trials/trial-00.jsonl",
        front_video_path="videos/trial-00-front.mp4",
        up_video_path="videos/trial-00-up.mp4",
    )
    result = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event="operator_abort",
        terminal_reason="operator stopped before observation capture",
    )
    output = tmp_path / "evidence"
    handoff = tmp_path / "handoffs"
    material = output / "session-1/act/shakedown/supervised-shakedown"
    (material / "motion_record/safety_traces").mkdir(parents=True)
    (material / "motion_record/videos").mkdir()
    (material / "PHASE_RESULT.json").write_bytes(
        canonical_json_bytes(execute_ops._phase_result_payload(result))
    )
    (material / "motion_record/safety_traces/trial-00.jsonl").write_bytes(
        canonical_json_bytes(
            {
                "trial": trial.trial_id,
                "condition": condition,
                "index": 0,
                "elapsed_s": 1.0,
                "event": "operator_abort",
                "detail": result.terminal_reason,
                "actions_sent": 0,
                "action_attempts": 0,
                "ambiguous_write_attempts": 0,
            }
        )
        + b"\n"
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    gate_request = SimpleNamespace(
        repository_root=repository,
        session_bundle=Path("accepted-session"),
        candidate_bundle=Path("accepted-candidate"),
        prior_hold_bundle=Path("accepted-hold"),
        prior_shakedown_bundle=None,
    )

    def publish(run, **kwargs):
        jobs.append(kwargs["job_type"])
        return run

    monkeypatch.setattr(
        rollout_evidence,
        "seal_unsafe_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            rollout_evidence.UnsafeTerminalVideoUnavailableError(
                "no transferable camera frame"
            )
        ),
    )
    first = execute_ops._execute_authorized_command(
        gate_request,
        permit,
        output=output,
        handoff=handoff,
        evidence_root=output,
        handoff_root=handoff,
        wandb_entity="test-entity",
        wandb_project="project",
        intent_publisher=publish,
    )
    second = execute_ops._execute_authorized_command(
        gate_request,
        permit,
        output=output,
        handoff=handoff,
        evidence_root=output,
        handoff_root=handoff,
        wandb_entity="test-entity",
        wandb_project="project",
        intent_publisher=publish,
    )

    assert first.status == second.status == "unsafe_shakedown"
    assert first.bundle is second.bundle is None
    marker = json.loads((material / "UNSAFE_NOT_SEALED.json").read_bytes())
    assert marker["blocker"] == "repo_b_unsafe_terminal_video_unavailable"
    assert marker["ready"] is False
    assert (material / "UNSAFE_WANDB_SYNCED.json").is_file()
    assert not list(handoff.rglob("READY"))
    assert jobs == [
        "viola-policy-execution-intent",
        "viola-policy-unsafe-outcome",
        "viola-policy-execution-intent",
        "viola-policy-unsafe-outcome",
    ]


def test_unsafe_shakedown_does_not_mask_other_finalizer_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_pre_hardware_checks(monkeypatch, order)
    permit = _permit(trial="supervised-shakedown")
    permit.phase = "shakedown"
    permit.speed_scale = 0.25
    result = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=0.25,
        held_action=None,
        trials=(),
        terminal_event="feedback_loss",
        terminal_reason="tampered evidence",
    )
    output = tmp_path / "evidence"
    handoff = tmp_path / "handoffs"
    material = output / "session-1/act/shakedown/supervised-shakedown"
    (material / "motion_record/safety_traces").mkdir(parents=True)
    (material / "motion_record/videos").mkdir()
    (material / "PHASE_RESULT.json").write_bytes(
        canonical_json_bytes(execute_ops._phase_result_payload(result))
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    gate_request = SimpleNamespace(
        repository_root=repository,
        session_bundle=Path("accepted-session"),
        candidate_bundle=Path("accepted-candidate"),
        prior_hold_bundle=Path("accepted-hold"),
        prior_shakedown_bundle=None,
    )
    monkeypatch.setattr(
        rollout_evidence,
        "seal_unsafe_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValidationError("schema tamper")),
    )
    monkeypatch.setattr(execute_ops, "_load_phase_result", lambda *_args, **_kwargs: result)

    with pytest.raises(ValidationError, match="schema tamper"):
        execute_ops._execute_authorized_command(
            gate_request,
            permit,
            output=output,
            handoff=handoff,
            evidence_root=output,
            handoff_root=handoff,
            wandb_entity="test-entity",
            wandb_project="project",
            intent_publisher=lambda run, **_kwargs: run,
        )
    assert not (material / "UNSAFE_NOT_SEALED.json").exists()


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


def test_phase_result_round_trip_preserves_zero_send_ambiguous_abort(
    tmp_path: Path,
) -> None:
    permit = _permit()
    permit.phase = "shakedown"
    permit.speed_scale = 0.25
    trial = TrialResult(
        trial_id="session-1-shakedown-01",
        index=0,
        condition=shakedown_conditions()[0].to_dict(),
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        duration_sec=1.0,
        actions=0,
        replans=0,
        inference_latency_ms=(),
        control_latency_ms=(),
        outcome=TrialOutcome(
            False,
            "safety_abort",
            "feedback_loss",
            None,
            None,
            None,
            0.0,
        ),
        safety_events=("feedback_loss",),
        trace_path="trials/trial-00.jsonl",
        front_video_path="videos/trial-00-front.mp4",
        up_video_path="videos/trial-00-up.mp4",
    )
    result = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at="2026-08-13T01:00:00+00:00",
        completed_at="2026-08-13T01:00:01+00:00",
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event="feedback_loss",
        terminal_reason="motor write outcome was unknown",
    )
    path = tmp_path / "PHASE_RESULT.json"
    path.write_bytes(canonical_json_bytes(execute_ops._phase_result_payload(result)))
    assert execute_ops._load_phase_result(path, permit=permit) == result


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
