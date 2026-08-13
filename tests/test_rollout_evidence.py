from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import viola_ops.rollout_evidence as rollout_evidence
from viola_handoff import canonical_json_bytes
from viola_ops.errors import ValidationError
from viola_ops.execution import ACTION_KEYS, PhaseResult, TrialOutcome, TrialResult
from viola_ops.rollout_evidence import (
    _finalize_trial,
    _prior_references,
    seal_completed_phase,
)


REPO_B_REVISION = "6fcf6439191c47bdee40c00b509a9f22712ac22a"
LEGACY_TRACE_KEYS = {
    "trial",
    "condition",
    "index",
    "elapsed_ms",
    "observation_age_ms",
    "replan",
    "state",
    "front_sha256",
    "up_sha256",
    "proposed_action",
    "sent_action",
    "feedback_action",
    "inference_ms",
}


def _trial() -> tuple[PhaseResult, TrialResult]:
    started = datetime(2026, 8, 13, 1, 0, tzinfo=UTC)
    completed = started + timedelta(seconds=60)
    condition = {
        "condition_id": "shakedown_01",
        "stratum": "shakedown",
        "blue_axis": None,
        "blue_offset_mm": 0.0,
        "red_axis": None,
        "red_offset_mm": 0.0,
    }
    trial = TrialResult(
        trial_id="session-1-shakedown-01",
        index=0,
        condition=condition,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        duration_sec=60.0,
        actions=1,
        replans=1,
        inference_latency_ms=(2.0,),
        control_latency_ms=(4.0,),
        outcome=TrialOutcome(False, "failure", "timeout", None, None, None, 0.0),
        safety_events=(),
        trace_path="unused",
        front_video_path="unused",
        up_video_path="unused",
    )
    phase = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event=None,
        terminal_reason=None,
    )
    return phase, trial


def _materialize_rich_trace(root: Path, trial: TrialResult) -> None:
    (root / "safety_traces").mkdir(parents=True)
    (root / "videos").mkdir()
    state = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
    row = {
        "trial": trial.trial_id,
        "condition": trial.condition,
        "index": 0,
        "elapsed_s": 0.004,
        "observation_read_ms": 1.0,
        "camera_freshness": {"frame_age_ms": {"front": 2.0, "up": 3.0}},
        "replan": True,
        "state": state,
        "proposed_action": state,
        "sent_action": state,
        "feedback_action": state,
        "feedback_check": {"retained": True},
        "inference_ms": 2.0,
        "control_ms": 4.0,
        "deadline_ms": 300.0,
        "deadline_passed": True,
        "target_period_ms": 1000.0 / 30.0,
        "rate_tolerance_ms": 5.0,
        "limit_check": {"passed": True},
        "front_sha256": "a" * 64,
        "up_sha256": "b" * 64,
    }
    (root / "safety_traces" / "trial-00.jsonl").write_bytes(
        canonical_json_bytes(row) + b"\n"
    )
    (root / "videos" / "trial-00-front.mp4").write_bytes(b"front-video")
    (root / "videos" / "trial-00-up.mp4").write_bytes(b"up-video")


def test_compact_completed_trace_has_the_exact_repo_b_legacy_shape(tmp_path: Path) -> None:
    phase, trial = _trial()
    motion_root = tmp_path / "motion_record"
    _materialize_rich_trace(motion_root, trial)

    record = _finalize_trial(phase, trial, motion_root)
    trace_path = motion_root / record["trace"]["path"]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) == LEGACY_TRACE_KEYS
    assert rows[0]["trial"] == 0
    assert rows[0]["condition"] == trial.condition
    assert rows[0]["replan"] is True
    assert rows[0]["elapsed_ms"] == 4.0
    assert rows[0]["observation_age_ms"] == 3.0
    assert record["actions"] == record["replans"] == 1
    assert record["failure_code"] == "timeout"
    assert record["safety_events"] == []


def test_authoritative_repo_b_consumer_accepts_the_compact_trace_when_available(
    tmp_path: Path,
) -> None:
    configured = os.environ.get("VIOLA_REPO_B_ROOT")
    repo_b = (
        Path(configured)
        if configured
        else Path("/tmp/viola-repob-audit.0JrvNj/repo-b")
    )
    if not (repo_b / "src/viola_benchmark/rollout_evidence.py").is_file():
        pytest.skip("set VIOLA_REPO_B_ROOT to run the authoritative cross-repo consumer")
    head_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_b,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reviewed_tree = subprocess.run(
        ["git", "rev-parse", f"{REPO_B_REVISION}^{{tree}}"],
        cwd=repo_b,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head_tree != reviewed_tree:
        pytest.skip("Repo-B checkout tree differs from reviewed main revision")

    phase, trial = _trial()
    motion_root = tmp_path / "motion_record"
    _materialize_rich_trace(motion_root, trial)
    record = _finalize_trial(phase, trial, motion_root)
    trial_path = tmp_path / "trial.json"
    trial_path.write_bytes(canonical_json_bytes(record))
    trace_path = motion_root / record["trace"]["path"]
    script = """
import json
import sys
from pathlib import Path
from viola_benchmark.rollout_evidence import _validate_trace

trial = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
result = _validate_trace(Path(sys.argv[1]), index=0, trial=trial)
assert result['actions'] == 1
assert result['replans'] == 1
"""
    environment = dict(os.environ)
    repo_a_src = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join((str(repo_b / "src"), str(repo_a_src)))
    subprocess.run(
        [sys.executable, "-c", script, str(trace_path), str(trial_path)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_phase_predecessor_references_are_exact_and_fail_closed() -> None:
    hold = type("Bundle", (), {"bundle_id": "a" * 64, "content_id": "a" * 64})()
    shakedown = type("Bundle", (), {"bundle_id": "b" * 64, "content_id": "b" * 64})()
    assert _prior_references("hold", None, None) == {"hold": None, "shakedown": None}
    assert _prior_references("shakedown", hold, None) == {
        "hold": {"bundle_id": "a" * 64, "content_id": "a" * 64},
        "shakedown": None,
    }
    assert _prior_references("scored", hold, shakedown)["shakedown"] == {
        "bundle_id": "b" * 64,
        "content_id": "b" * 64,
    }
    with pytest.raises(ValidationError, match="requires completed hold"):
        _prior_references("shakedown", None, None)
    with pytest.raises(ValidationError, match="requires hold and shakedown"):
        _prior_references("scored", hold, None)


def test_unsafe_result_is_never_published_or_sealed() -> None:
    called = False

    def publisher(*_args, **_kwargs):
        nonlocal called
        called = True

    unsafe = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at="2026-08-13T00:00:00+00:00",
        completed_at="2026-08-13T00:00:01+00:00",
        speed_scale=0.25,
        held_action=None,
        trials=(),
        terminal_event="collision",
        terminal_reason="operator reported collision",
    )
    with pytest.raises(ValidationError, match="cannot be sealed"):
        seal_completed_phase(
            unsafe,
            session=None,
            candidate=None,
            identity=None,
            experiment="viola-eight-policy-v1",
            material_root=Path("unused"),
            handoff_root=Path("unused"),
            wandb_entity="unused",
            wandb_project="unused",
            evidence_factory=None,
            publisher=publisher,
        )
    assert called is False


def test_completed_hold_finalization_retries_without_replacing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_payload = {
        "session_id": "session-1",
        "policy_bundle_id": "b" * 64,
        "operator": "operator",
        "executor": {"repository_commit": "d" * 40},
    }
    session_path = tmp_path / "rollout_session.json"
    session_path.write_bytes(canonical_json_bytes(session_payload))

    class Session:
        bundle_id = "a" * 64
        content_id = "a" * 64
        manifest = {
            "producer": {"repository_commit": "c" * 40},
            "experiment": "viola-eight-policy-v1",
            "lineage": {
                "policy": "act",
                "session_manifest_sha256": "e" * 64,
            },
        }

        def payload_file(self, name: str) -> Path:
            assert name == "rollout_session.json"
            return session_path

    candidate_bundle = type(
        "CandidateBundle",
        (),
        {
            "bundle_id": "b" * 64,
            "content_id": "b" * 64,
            "manifest": {"experiment": "viola-eight-policy-v1"},
        },
    )()
    candidate = type(
        "Candidate",
        (),
        {
            "bundle": candidate_bundle,
            "bundle_id": "b" * 64,
            "content_id": "b" * 64,
            "payload": {
                "evaluation_sha256": "f" * 64,
                "checkpoint_inventory_sha256": "0" * 64,
                "dataset_release_id": "dataset-release-1",
            },
        },
    )()
    identity = type(
        "Identity",
        (),
        {
            "role": "pc_a",
            "repository_clean": True,
            "repository_commit": "d" * 40,
            "hostname": "pc-a",
            "python_version": "3.12.13",
        },
    )()
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
    material = tmp_path / "material"
    calls: list[str] = []

    def fail_publish(*_args, **_kwargs):
        calls.append("failed-publish")
        raise ValidationError("W&B outage")

    with pytest.raises(ValidationError, match="W&B outage"):
        seal_completed_phase(
            result,
            session=Session(),
            candidate=candidate,
            identity=identity,
            experiment="viola-eight-policy-v1",
            material_root=material,
            handoff_root=tmp_path / "handoffs",
            wandb_entity="entity",
            wandb_project="project",
            evidence_factory=None,
            publisher=fail_publish,
            handoff_logger=object(),
        )
    payload_before = (material / "payload" / "rollout_evidence.json").read_bytes()
    assert not (material / "payload" / "WANDB_SYNCED.json").exists()

    fake_bundle = object()
    monkeypatch.setattr(
        rollout_evidence,
        "seal_bundle",
        lambda request, evidence_logger: calls.append("seal") or fake_bundle,
    )

    def finish_publish(*_args, **_kwargs):
        calls.append("finished-publish")

    sealed = seal_completed_phase(
        result,
        session=Session(),
        candidate=candidate,
        identity=identity,
        experiment="viola-eight-policy-v1",
        material_root=material,
        handoff_root=tmp_path / "handoffs",
        wandb_entity="entity",
        wandb_project="project",
        evidence_factory=None,
        publisher=finish_publish,
        handoff_logger=object(),
    )
    assert sealed.bundle is fake_bundle
    assert sealed.payload_path.read_bytes() == payload_before
    sync_before = sealed.sync_path.read_bytes()

    again = seal_completed_phase(
        result,
        session=Session(),
        candidate=candidate,
        identity=identity,
        experiment="viola-eight-policy-v1",
        material_root=material,
        handoff_root=tmp_path / "handoffs",
        wandb_entity="entity",
        wandb_project="project",
        evidence_factory=None,
        publisher=finish_publish,
        handoff_logger=object(),
    )
    assert again.sync_path.read_bytes() == sync_before
    assert calls == [
        "failed-publish",
        "finished-publish",
        "seal",
        "finished-publish",
        "seal",
    ]
