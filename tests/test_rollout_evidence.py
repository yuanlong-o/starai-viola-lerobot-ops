from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import viola_ops.rollout_evidence as rollout_evidence
from viola_handoff import canonical_json_bytes
from viola_ops.errors import ValidationError
from viola_ops.evidence import PhaseEvidenceFactory
from viola_ops.execution import ACTION_KEYS, PhaseResult, TrialOutcome, TrialResult
from viola_ops.rollout_evidence import (
    UnsafeTerminalVideoUnavailableError,
    _finalize_unsafe_trials,
    _finalize_trial,
    _prior_references,
    seal_completed_phase,
    seal_unsafe_phase,
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
UNSAFE_TERMINAL_TRACE_KEYS = {
    "trial",
    "condition",
    "index",
    "elapsed_s",
    "event",
    "detail",
    "stage",
    "captured_observations",
    "actions_sent",
    "verified_actions",
    "proposed_action",
    "sent_action",
    "feedback_action",
    "feedback_check",
    "limit_check",
    "timing",
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


def _unsafe_material(tmp_path: Path) -> tuple[PhaseResult, Path]:
    """Create one hardware-inert ambiguous-write terminal with real videos."""

    import av
    import numpy as np

    started = datetime(2026, 8, 13, 2, 0, tzinfo=UTC)
    completed = started + timedelta(milliseconds=100)
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
        duration_sec=0.1,
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
    phase = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event="feedback_loss",
        terminal_reason="the public SDK write outcome is unknown",
    )

    motion = tmp_path / "motion_record"
    (motion / "safety_traces").mkdir(parents=True)
    (motion / "videos").mkdir()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for camera in ("front", "up"):
        path = motion / "videos" / f"trial-00-{camera}.mp4"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = 640
            stream.height = 480
            stream.pix_fmt = "yuv420p"
            image = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(image):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    state = {key: (50.0 if key == "gripper.pos" else 0.0) for key in ACTION_KEYS}
    joints = [key.removesuffix(".pos") for key in ACTION_KEYS]
    lower = {joint: (0.0 if joint == "gripper" else -100.0) for joint in joints}
    upper = {joint: 100.0 for joint in joints}
    reviewed = {joint: 1.0 for joint in joints}
    allowed = {joint: 0.25 for joint in joints}
    failed_write = {
        "trial": trial.trial_id,
        "condition": condition,
        "index": 0,
        "elapsed_s": 0.05,
        "observation_read_ms": 1.0,
        "camera_freshness": {
            "public_api": "OpenCVCamera.async_read",
            "arrival_clock": "time.perf_counter",
            "read_deadline_ms": 100.0,
            "frame_age_ms": {"front": 2.0, "up": 3.0},
            "interarrival_ms": {"front": None, "up": None},
        },
        "replan": True,
        "state": state,
        "proposed_action": state,
        "attempted_action": state,
        "sent_action": None,
        "write_outcome": "unknown",
        "feedback_action": None,
        "feedback_check": {
            "receipt_violations": ["sdk_write_outcome_unknown"],
            "passed": False,
        },
        "inference_ms": 2.0,
        "control_ms": 4.0,
        "deadline_ms": 300.0,
        "deadline_passed": True,
        "target_period_ms": 1000.0 / 30.0,
        "rate_tolerance_ms": 5.0,
        "period_ms": 50.0,
        "rate_passed": False,
        "limit_check": {
            "joint_order": joints,
            "reference_action": state,
            "lower": lower,
            "upper": upper,
            "reviewed_max_step_deltas": reviewed,
            "speed_scale": 0.25,
            "allowed_step_deltas": allowed,
            "absolute_violations": [],
            "rate_violations": [],
            "would_be_clamped": [],
            "passed": True,
            "proposed_action": state,
        },
        "front_sha256": "a" * 64,
        "up_sha256": "b" * 64,
    }
    terminal = {
        "trial": trial.trial_id,
        "condition": condition,
        "index": 1,
        "elapsed_s": 0.1,
        "event": "feedback_loss",
        "detail": "the public SDK write outcome is unknown",
        "actions_sent": 0,
        "action_attempts": 1,
        "ambiguous_write_attempts": 1,
    }
    (motion / "safety_traces" / "trial-00.jsonl").write_bytes(
        canonical_json_bytes(failed_write)
        + b"\n"
        + canonical_json_bytes(terminal)
        + b"\n"
    )
    return phase, motion


def _terminal_only_unsafe_material(tmp_path: Path) -> tuple[PhaseResult, Path]:
    """Exercise the production writer when motion stops before frame capture."""

    started = datetime(2026, 8, 13, 2, 0, tzinfo=UTC)
    completed = started + timedelta(milliseconds=100)
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
        duration_sec=0.1,
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
    phase = PhaseResult(
        session_id="session-1",
        policy="act",
        phase="shakedown",
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        speed_scale=0.25,
        held_action=None,
        trials=(trial,),
        terminal_event="operator_abort",
        terminal_reason="operator requested a stop before observation capture",
    )
    factory = PhaseEvidenceFactory(tmp_path / "motion_record")
    writer = factory.start_trial(trial.trial_id)
    writer.record_terminal(
        {
            "trial": trial.trial_id,
            "condition": condition,
            "index": 0,
            "elapsed_s": 0.1,
            "event": "operator_abort",
            "detail": "operator requested a stop before observation capture",
            "actions_sent": 0,
            "action_attempts": 0,
            "ambiguous_write_attempts": 0,
        }
    )
    writer.close()
    return phase, factory.root


def _reviewed_repo_b() -> Path | None:
    configured = os.environ.get("VIOLA_REPO_B_ROOT")
    candidates = [
        Path(configured) if configured else None,
        Path("/tmp/viola-repob-main.lOT9ZS"),
        Path("/tmp/viola-repob-audit.0JrvNj/repo-b"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        consumer = candidate / "src/viola_benchmark/rollout_evidence.py"
        if consumer.is_file():
            return candidate
    return None


def _require_reviewed_repo_b() -> Path:
    repo_b = _reviewed_repo_b()
    if repo_b is None:
        pytest.skip("set VIOLA_REPO_B_ROOT to run the authoritative Repo-B consumer")
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
    return repo_b


def test_unsafe_projection_preserves_the_ambiguous_raw_write(tmp_path: Path) -> None:
    phase, motion = _unsafe_material(tmp_path)
    raw_path = motion / "safety_traces/trial-00.jsonl"
    raw_before = raw_path.read_bytes()

    details = _finalize_unsafe_trials(phase, motion)

    assert raw_path.read_bytes() == raw_before
    assert details["failure_type"] == "feedback_loss"
    assert details["motion_aborted"] is True
    trial = details["trials"][0]
    assert trial["actions"] == trial["verified_actions"] == 0
    assert trial["captured_observations"] == 1
    assert trial["completed_control"] is False
    assert trial["safety_events"] == ["feedback_loss"]
    projected = [
        json.loads(line)
        for line in (motion / trial["trace"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(projected) == 1
    terminal = projected[0]
    assert set(terminal) == UNSAFE_TERMINAL_TRACE_KEYS
    assert terminal["event"] == "feedback_loss"
    assert terminal["stage"] == "write_feedback"
    assert terminal["proposed_action"] == [0.0] * 6 + [50.0]
    assert terminal["sent_action"] is None
    assert terminal["feedback_action"] is None
    assert terminal["limit_check"]["passed"] is True


def test_unsafe_projection_rejects_tampered_raw_frame_trial_identity(
    tmp_path: Path,
) -> None:
    phase, motion = _unsafe_material(tmp_path)
    raw_path = motion / "safety_traces/trial-00.jsonl"
    rows = [json.loads(line) for line in raw_path.read_bytes().splitlines()]
    rows[0]["trial"] = "another-session-shakedown-01"
    raw_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )

    with pytest.raises(ValidationError, match="frame row differs from its trial identity"):
        _finalize_unsafe_trials(phase, motion)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("actions_sent", 1),
        ("action_attempts", 2),
        ("ambiguous_write_attempts", 0),
    ),
)
def test_unsafe_projection_rejects_tampered_terminal_attempt_counters(
    tmp_path: Path,
    field: str,
    tampered: int,
) -> None:
    phase, motion = _unsafe_material(tmp_path)
    raw_path = motion / "safety_traces/trial-00.jsonl"
    rows = [json.loads(line) for line in raw_path.read_bytes().splitlines()]
    rows[-1][field] = tampered
    raw_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )

    with pytest.raises(
        ValidationError,
        match="terminal trace differs from its phase result or frame attempts",
    ):
        _finalize_unsafe_trials(phase, motion)


def test_terminal_only_writer_has_a_typed_nontransferable_video_failure(
    tmp_path: Path,
) -> None:
    phase, motion = _terminal_only_unsafe_material(tmp_path)
    raw_path = motion / "safety_traces/trial-00.jsonl"
    raw_before = raw_path.read_bytes()
    assert not (motion / "videos/trial-00-front.mp4").exists()
    assert not (motion / "videos/trial-00-up.mp4").exists()

    with pytest.raises(
        UnsafeTerminalVideoUnavailableError,
        match="before a transferable camera frame",
    ):
        _finalize_unsafe_trials(phase, motion)

    assert raw_path.read_bytes() == raw_before
    assert not (motion / "traces/trial-00.jsonl").exists()


def test_terminal_only_fallback_does_not_hide_trace_tampering(tmp_path: Path) -> None:
    phase, motion = _terminal_only_unsafe_material(tmp_path)
    raw_path = motion / "safety_traces/trial-00.jsonl"
    terminal = json.loads(raw_path.read_bytes())
    terminal["unexpected"] = "tamper"
    raw_path.write_bytes(canonical_json_bytes(terminal) + b"\n")

    with pytest.raises(ValidationError, match="unexpected shape") as captured:
        _finalize_unsafe_trials(phase, motion)

    assert type(captured.value) is ValidationError


def test_authoritative_repo_b_consumer_accepts_unsafe_ambiguous_write(
    tmp_path: Path,
) -> None:
    repo_b = _require_reviewed_repo_b()
    phase, motion = _unsafe_material(tmp_path)
    payload = {
        "phase": phase.phase,
        "session_id": phase.session_id,
        "policy": phase.policy,
        "started_at": phase.started_at,
        "completed_at": phase.completed_at,
        "details": _finalize_unsafe_trials(phase, motion),
    }
    payload_path = tmp_path / "unsafe-payload.json"
    payload_path.write_bytes(canonical_json_bytes(payload))
    script = """
import json
import sys
from pathlib import Path
from viola_benchmark.rollout_evidence import _validate_unsafe_motion_details

payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
trials, event = _validate_unsafe_motion_details(
    payload,
    session={},
    motion_root=Path(sys.argv[2]),
)
assert event == 'feedback_loss'
assert trials[-1]['completed_control'] is False
assert trials[-1]['captured_observations'] == 1
"""
    environment = dict(os.environ)
    repo_a_src = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(repo_b / "src"), str(repo_a_src)))
    subprocess.run(
        [sys.executable, "-c", script, str(payload_path), str(motion)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


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
    repo_b = _require_reviewed_repo_b()

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


def test_completed_finalizer_rejects_unsafe_result_before_publish() -> None:
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
    with pytest.raises(ValidationError, match="accepts only a completed"):
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


def test_unsafe_shakedown_seals_as_terminal_evidence_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase, motion = _unsafe_material(tmp_path)
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
        content_id = "1" * 64
        manifest = {
            "experiment": "viola-eight-policy-v1",
            "lineage": {
                "policy": "act",
                "session_manifest_sha256": "e" * 64,
            },
        }

        def payload_file(self, name: str) -> Path:
            assert name == "rollout_session.json"
            return session_path

    candidate_bundle = SimpleNamespace(
        bundle_id="b" * 64,
        content_id="2" * 64,
        manifest={"experiment": "viola-eight-policy-v1"},
    )
    candidate = SimpleNamespace(
        bundle=candidate_bundle,
        bundle_id=candidate_bundle.bundle_id,
        content_id=candidate_bundle.content_id,
        payload={
            "evaluation_sha256": "f" * 64,
            "checkpoint_inventory_sha256": "0" * 64,
            "dataset_release_id": "dataset-release-1",
        },
    )
    identity = SimpleNamespace(
        role="pc_a",
        repository_clean=True,
        repository_commit="d" * 40,
        hostname="pc-a",
        python_version="3.12.13",
    )
    hold = SimpleNamespace(bundle_id="c" * 64, content_id="3" * 64)
    sealed_bundle = SimpleNamespace(path=tmp_path / "sealed", bundle_id="4" * 64)
    requests = []
    published = []

    def capture_seal(request, *, evidence_logger):
        assert evidence_logger is not None
        requests.append(request)
        return sealed_bundle

    def publish(run, **kwargs):
        published.append((run, kwargs))
        return run

    monkeypatch.setattr(rollout_evidence, "seal_bundle", capture_seal)
    sealed = seal_unsafe_phase(
        phase,
        session=Session(),
        candidate=candidate,
        identity=identity,
        experiment="viola-eight-policy-v1",
        material_root=tmp_path,
        handoff_root=tmp_path / "handoffs",
        wandb_entity="entity",
        wandb_project="project",
        evidence_factory=PhaseEvidenceFactory.reopen_completed(motion),
        prior_hold=hold,
        publisher=publish,
        handoff_logger=object(),
    )

    assert sealed.bundle is sealed_bundle
    assert len(requests) == 1
    request = requests[0]
    assert request.kind == "rollout_evidence"
    assert request.permission == "evidence_only"
    assert request.artifact_roots == {"motion_record": motion}
    assert len(published) == 1
    _, publication = published[0]
    assert publication["job_type"] == "viola-policy-execute"
    assert set(publication["config"]) == {
        "session_bundle_id",
        "session_id",
        "policy_bundle_id",
        "policy",
        "experiment",
        "phase",
        "prior_phase_bundles",
        "repo",
        "operator",
    }
    payload = json.loads(sealed.payload_path.read_bytes())
    assert payload["status"] == "unsafe_shakedown"
    assert payload["prior_phase_bundles"] == {
        "hold": {"bundle_id": hold.bundle_id, "content_id": hold.content_id},
        "shakedown": None,
    }


def test_unsafe_scored_phase_stays_non_ready_before_publish(tmp_path: Path) -> None:
    phase, _motion = _unsafe_material(tmp_path)
    scored = replace(phase, phase="scored", speed_scale=1.0)
    called = False

    def publish(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(ValidationError, match="incompatible completed-shakedown"):
        seal_unsafe_phase(
            scored,
            session=None,
            candidate=None,
            identity=None,
            experiment="viola-eight-policy-v1",
            material_root=tmp_path,
            handoff_root=tmp_path / "handoffs",
            wandb_entity="entity",
            wandb_project="project",
            evidence_factory=None,
            publisher=publish,
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
