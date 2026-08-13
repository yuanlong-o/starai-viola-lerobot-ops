from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import viola_ops.local_act_ops as ops
from viola_handoff import RuntimeIdentity
from viola_ops.dataset import RELEASE_ID
from viola_ops.errors import ValidationError
from viola_ops.execution import PhaseResult, TrialOutcome, TrialResult
from viola_ops.local_act import LEGACY_ACT_MODEL_SHA256
from viola_ops.local_act_ops import LocalActExecutionApi, load_local_act_setup


def _identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        role="pc_a",
        repository_commit="a" * 40,
        repository_clean=True,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _candidate() -> Any:
    return SimpleNamespace(
        checkpoint=Path("/reviewed/checkpoint"),
        dataset_root=Path("/reviewed/dataset"),
        checkpoint_inventory={"inventory_sha256": "c" * 64},
        dataset_inventory={"inventory_sha256": "d" * 64},
        dataset_metadata_inventory={"inventory_sha256": "e" * 64},
        model_sha256=LEGACY_ACT_MODEL_SHA256,
        dataset_release_id=RELEASE_ID,
        candidate_id="b" * 64,
        bundle_id="b" * 64,
        policy="act",
    )


def _permit(setup: Any, candidate: Any, *, trial: str) -> Any:
    session_binding_id = "1" * 64
    return SimpleNamespace(
        session_id=f"local-act-{session_binding_id[:24]}",
        session_bundle_id=session_binding_id,
        candidate_bundle_id=candidate.candidate_id,
        policy="act",
        phase="local_act",
        trial=trial,
        operator="operator-a",
        estop_tested_at=datetime.now(UTC),
        setup_id=setup.setup_id,
        setup_hashes=setup.setup_hashes,
        speed_scale=0.25,
        robot_port=setup.robot_port,
        local_act_duration_s=10.0,
    )


@contextmanager
def _lease(_robot_port: str):
    yield


@contextmanager
def _runtime_snapshot(candidate: Any):
    yield SimpleNamespace(candidate=candidate, verify=lambda: None)


class _Operator:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Evidence:
    def __init__(self, root: Path) -> None:
        (root / "safety_traces").mkdir(parents=True)
        (root / "videos").mkdir()
        (root / "safety_traces/trial-00.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "videos/trial-00-front.mp4").write_bytes(b"front")
        (root / "videos/trial-00-up.mp4").write_bytes(b"up")


def _completed_result(session_id: str, duration_s: float) -> PhaseResult:
    actions = round(duration_s * 30)
    trial = TrialResult(
        trial_id=f"{session_id}-local_act-01",
        index=0,
        condition={
            "condition_id": "local_act_01",
            "stratum": "local_act",
            "blue_axis": None,
            "blue_offset_mm": 0.0,
            "red_axis": None,
            "red_offset_mm": 0.0,
        },
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        duration_sec=duration_s,
        actions=actions,
        replans=(actions + 9) // 10,
        inference_latency_ms=(1.0,) * actions,
        control_latency_ms=(2.0,) * actions,
        outcome=TrialOutcome(False, "failure", "timeout", None, None, None, 0.0),
        safety_events=(),
        trace_path="trials/trial-00.jsonl",
        front_video_path="videos/trial-00-front.mp4",
        up_video_path="videos/trial-00-up.mp4",
    )
    return PhaseResult(
        session_id=session_id,
        policy="act",
        phase="local_act",
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event=None,
        terminal_reason=None,
    )


@pytest.fixture
def independent_runtime(monkeypatch: pytest.MonkeyPatch):
    repository = Path(__file__).resolve().parents[1]
    candidate = _candidate()
    identity = _identity()
    setup = load_local_act_setup(
        repository / "config/local_act_setup.json",
        repo_root=repository,
        identity=identity,
    )
    permit = _permit(setup, candidate, trial="act-smoke")
    monkeypatch.setattr(ops, "inspect_local_act_candidate", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(ops, "authorize_local_act_motion", lambda *_args, **_kwargs: permit)
    monkeypatch.setattr(ops, "revalidate_local_act_motion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ops, "snapshot_local_act_runtime", _runtime_snapshot)
    monkeypatch.setattr(ops, "_execution_lease", _lease)
    monkeypatch.setattr(ops, "_require_local_devices", lambda _setup: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return repository, candidate, identity, permit


def test_versioned_local_setup_reproduces_historical_rig() -> None:
    repository = Path(__file__).resolve().parents[1]
    setup = load_local_act_setup(
        repository / "config/local_act_setup.json",
        repo_root=repository,
        identity=_identity(),
    )

    assert setup.setup_id == "pc-a-local-act-existing-rig-v1"
    assert setup.robot_port.endswith("usb-0:10:1.0-port0")
    assert set(setup.camera_configs) == {"front", "up"}
    assert setup.max_step_deltas == {joint: 3.0 for joint in ops.JOINTS}
    assert setup.absolute_limits["Motor_0"] == (-100.0, 100.0)
    assert setup.absolute_limits["gripper"] == (0.0, 100.0)


def test_device_preflight_explains_current_dialout_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = SimpleNamespace(
        robot_port="/dev/null",
        camera_configs={
            "front": {"index_or_path": "/dev/zero"},
            "up": {"index_or_path": "/dev/full"},
        },
    )
    monkeypatch.setattr(
        ops.os,
        "access",
        lambda path, _mode: str(path) != "/dev/null",
    )

    with pytest.raises(ValidationError, match="sg dialout"):
        ops._require_local_devices(setup)


def test_local_act_requires_the_historical_gpu_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValidationError, match="CUDA_VISIBLE_DEVICES=0"):
        ops._require_local_act_cuda()


def test_online_intent_failure_prevents_hardware_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    independent_runtime: tuple[Path, Any, RuntimeIdentity, Any],
) -> None:
    repository, _candidate_value, identity, _permit_value = independent_runtime
    imported = False

    def reject_intent(*_args, **_kwargs):
        raise ValidationError("W&B unavailable")

    def hardware_api() -> LocalActExecutionApi:
        nonlocal imported
        imported = True
        raise AssertionError("hardware must stay unreachable")

    with pytest.raises(ValidationError, match="W&B unavailable"):
        ops.run_command(
            repo_root=repository,
            checkpoint=Path("/checkpoint"),
            dataset_root=Path("/dataset"),
            evidence_root=tmp_path / "evidence",
            operator="operator-a",
            trial="act-smoke",
            identity_capture=lambda **_kwargs: identity,
            intent_publisher=reject_intent,
            execution_api_factory=hardware_api,
        )

    assert imported is False
    assert not list(tmp_path.rglob("LOCAL_ACT_INTENT_WANDB_SYNCED.json"))


def test_repo_a_local_act_records_result_without_repo_b_ready(
    tmp_path: Path,
    independent_runtime: tuple[Path, Any, RuntimeIdentity, Any],
) -> None:
    repository, _candidate_value, identity, permit = independent_runtime
    published: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    operator = _Operator()
    authority_checks = 0

    def publish(run, *, job_type, config, summary, **_kwargs):
        published.append((job_type, dict(config), dict(summary)))
        return run

    def execute(*_args, revalidate_authority, local_act_duration_s, **_kwargs):
        nonlocal authority_checks
        revalidate_authority()
        authority_checks += 1
        return _completed_result(permit.session_id, local_act_duration_s)

    api = LocalActExecutionApi(
        execute_phase=execute,
        robot_factory=lambda _permit: None,
        evidence_factory=_Evidence,
        operator_factory=lambda: operator,
    )
    outcome = ops.run_command(
        repo_root=repository,
        checkpoint=Path("/checkpoint"),
        dataset_root=Path("/dataset"),
        evidence_root=tmp_path / "evidence",
        operator="operator-a",
        trial="act-smoke",
        duration_s=10.0,
        identity_capture=lambda **_kwargs: identity,
        intent_publisher=publish,
        execution_api_factory=lambda: api,
    )

    assert outcome.status == "completed"
    assert operator.closed is True
    assert authority_checks == 1
    assert [item[0] for item in published] == [
        "viola-local-act-intent",
        "viola-local-act-result",
    ]
    assert outcome.result_path.is_file()
    assert (outcome.material_root / "LOCAL_ACT_RESULT_WANDB_SYNCED.json").is_file()
    assert not list(outcome.material_root.rglob("READY.json"))
    assert not list(outcome.material_root.rglob("manifest.json"))
    assert "Repo A local-only" in outcome.render_text()


def test_execution_failure_is_immutable_and_not_ready(
    tmp_path: Path,
    independent_runtime: tuple[Path, Any, RuntimeIdentity, Any],
) -> None:
    repository, _candidate_value, identity, _permit_value = independent_runtime

    def publish(run, **_kwargs):
        return run

    def fail_execution(*_args, **_kwargs):
        raise RuntimeError("fake executor stopped")

    api = LocalActExecutionApi(
        execute_phase=fail_execution,
        robot_factory=lambda _permit: None,
        evidence_factory=_Evidence,
        operator_factory=_Operator,
    )
    with pytest.raises(RuntimeError, match="fake executor stopped"):
        ops.run_command(
            repo_root=repository,
            checkpoint=Path("/checkpoint"),
            dataset_root=Path("/dataset"),
            evidence_root=tmp_path / "evidence",
            operator="operator-a",
            trial="act-smoke",
            identity_capture=lambda **_kwargs: identity,
            intent_publisher=publish,
            execution_api_factory=lambda: api,
        )

    markers = list(tmp_path.rglob("LOCAL_ACT_FAILURE.json"))
    assert len(markers) == 1
    assert '"motion_may_have_started":true' in markers[0].read_text(encoding="utf-8")
    assert len(list(tmp_path.rglob("LOCAL_ACT_FAILURE_WANDB_SYNCED.json"))) == 1
    assert not list(tmp_path.rglob("READY.json"))


def test_failed_failure_upload_has_hardware_free_recovery(
    tmp_path: Path,
    independent_runtime: tuple[Path, Any, RuntimeIdentity, Any],
) -> None:
    repository, _candidate_value, identity, _permit_value = independent_runtime
    publications = 0

    def fail_second_publish(run, **_kwargs):
        nonlocal publications
        publications += 1
        if publications > 1:
            raise ValidationError("W&B outage")
        return run

    api = LocalActExecutionApi(
        execute_phase=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fake motion failure")
        ),
        robot_factory=lambda _permit: None,
        evidence_factory=_Evidence,
        operator_factory=_Operator,
    )
    with pytest.raises(RuntimeError, match="fake motion failure"):
        ops.run_command(
            repo_root=repository,
            checkpoint=Path("/checkpoint"),
            dataset_root=Path("/dataset"),
            evidence_root=tmp_path / "evidence",
            operator="operator-a",
            trial="act-smoke",
            identity_capture=lambda **_kwargs: identity,
            intent_publisher=fail_second_publish,
            execution_api_factory=lambda: api,
        )

    marker = next(tmp_path.rglob("LOCAL_ACT_FAILURE.json"))
    attempt = marker.parent
    assert not (attempt / "LOCAL_ACT_FAILURE_WANDB_SYNCED.json").exists()
    recovered = ops.recover_failure_command(
        attempt,
        repo_root=repository,
        identity_capture=lambda **_kwargs: identity,
        publisher=lambda run, **_kwargs: run,
    )

    assert recovered.status == "failure_recorded"
    assert recovered.receipt_path.is_file()
    assert not list(attempt.rglob("READY.json"))


def test_recovery_rejects_motion_added_after_absent_motion_marker(
    tmp_path: Path,
    independent_runtime: tuple[Path, Any, RuntimeIdentity, Any],
) -> None:
    repository, _candidate_value, identity, _permit_value = independent_runtime
    publications = 0

    def fail_after_intent(run, **_kwargs):
        nonlocal publications
        publications += 1
        if publications > 1:
            raise ValidationError("W&B outage")
        return run

    api = LocalActExecutionApi(
        execute_phase=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("failure before evidence allocation")
        ),
        robot_factory=lambda _permit: None,
        evidence_factory=lambda _root: object(),
        operator_factory=_Operator,
    )
    with pytest.raises(RuntimeError, match="before evidence allocation"):
        ops.run_command(
            repo_root=repository,
            checkpoint=Path("/checkpoint"),
            dataset_root=Path("/dataset"),
            evidence_root=tmp_path / "evidence",
            operator="operator-a",
            trial="act-no-motion-record",
            identity_capture=lambda **_kwargs: identity,
            intent_publisher=fail_after_intent,
            execution_api_factory=lambda: api,
        )

    marker = next(tmp_path.rglob("LOCAL_ACT_FAILURE.json"))
    value = json.loads(marker.read_text(encoding="utf-8"))
    assert value["partial_motion_inventory_error"] == "motion_record_not_created"
    (marker.parent / "motion_record").mkdir()
    with pytest.raises(ValidationError, match="motion evidence was absent"):
        ops.recover_failure_command(
            marker.parent,
            repo_root=repository,
            identity_capture=lambda **_kwargs: identity,
            publisher=lambda run, **_kwargs: run,
        )
