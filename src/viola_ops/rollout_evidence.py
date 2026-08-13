"""Finalize physical phases into Repo-B-compatible evidence.

The control loop keeps its lossless trace in ``safety_traces/``.  Completed
phases receive Repo B's compact compatibility view.  An aborted shakedown
receives Repo B's richer unsafe view.  Neither projection replaces the original
trace, so an operator and Repo B can always inspect what the live loop actually
recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viola_handoff import (
    EvidenceLogger,
    HandoffError,
    RuntimeIdentity,
    SealRequest,
    VerifiedBundle,
    WandbEvidenceLogger,
    canonical_json_bytes,
    seal_bundle,
)

from .errors import ValidationError
from .evidence import PhaseEvidenceFactory, read_trace
from .execution import (
    ACTION_KEYS,
    CONTROL_DEADLINE_MS,
    RATE_TOLERANCE_MS,
    SAFETY_EVENTS,
    TARGET_HZ,
    TRIAL_DURATION_S,
    PhaseResult,
    TrialResult,
)
from .jsonutil import read_json_object, sha256_file, write_canonical_json
from .policy_runtime import AcceptedPolicyCandidate
from .wandb_ops import WandbRunIdentity, planned_run, publish_finished_run


@dataclass(frozen=True, slots=True)
class SealedRolloutEvidence:
    bundle: VerifiedBundle
    payload_path: Path
    sync_path: Path
    material_root: Path
    motion_record: Path | None


class UnsafeTerminalVideoUnavailableError(ValidationError):
    """The terminal happened before either reviewed camera frame was retained."""


_JOINTS = tuple(key.removesuffix(".pos") for key in ACTION_KEYS)
_TARGET_PERIOD_MS = 1_000.0 / TARGET_HZ
_EXPECTED_ACTIONS = round(TRIAL_DURATION_S * TARGET_HZ)
_MINIMUM_ACTIONS = math.ceil(_EXPECTED_ACTIONS * 0.95)
_RAW_TERMINAL_FIELDS = {
    "trial",
    "condition",
    "index",
    "elapsed_s",
    "event",
    "detail",
    "actions_sent",
    "action_attempts",
    "ambiguous_write_attempts",
}
_FEEDBACK_FIELDS = {
    "freshness_source",
    "joint_order",
    "command_sequence",
    "previous_feedback_received_ns",
    "control_started_ns",
    "prewrite_received_ns",
    "write_started_ns",
    "write_completed_ns",
    "feedback_received_ns",
    "command_deadline_ns",
    "poll_count",
    "prewrite_action",
    "target_action",
    "feedback_action",
    "minimum_observable_progress",
    "material_joints",
    "signed_progress",
    "target_overshoot",
    "absolute_error",
    "allowed_error",
    "progress_violations",
    "tracking_violations",
    "overshoot_violations",
    "limit_violations",
    "receipt_violations",
    "violations",
    "passed",
}


def seal_completed_phase(
    result: PhaseResult,
    *,
    session: VerifiedBundle,
    candidate: AcceptedPolicyCandidate,
    identity: RuntimeIdentity,
    experiment: str,
    material_root: str | Path,
    handoff_root: str | Path,
    wandb_entity: str,
    wandb_project: str,
    evidence_factory: PhaseEvidenceFactory | None,
    prior_hold: VerifiedBundle | None = None,
    prior_shakedown: VerifiedBundle | None = None,
    publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
    handoff_logger: EvidenceLogger | None = None,
) -> SealedRolloutEvidence:
    """Publish and seal one completed hold, shakedown, or scored phase."""

    if result.status != "completed":
        raise ValidationError(
            "seal_completed_phase accepts only a completed physical phase; use the "
            "terminal-evidence path for an unsafe shakedown"
        )
    if result.session_id != _session_payload(session)["session_id"]:
        raise ValidationError("phase result belongs to another rollout session")
    if candidate.bundle_id != result_policy_bundle(session, result.policy):
        raise ValidationError("phase result belongs to another policy candidate")
    if identity.role != "pc_a" or identity.repository_clean is not True:
        raise ValidationError("rollout evidence requires a clean pc_a identity")
    # The session is produced by PC B; its producer commit is intentionally not
    # a Repo-A identity.  Always compare A's result to the executor commit that
    # PC B reviewed and embedded in the session.
    executor_commit = _session_payload(session)["executor"]["repository_commit"]
    if identity.repository_commit != executor_commit:
        raise ValidationError("rollout evidence commit differs from the reviewed executor")
    if {experiment, session.manifest["experiment"], candidate.bundle.manifest["experiment"]} != {
        experiment
    }:
        raise ValidationError("rollout evidence crosses experiment namespaces")

    prior = _prior_references(result.phase, prior_hold, prior_shakedown)
    root = Path(material_root).expanduser().resolve()
    payload_root = root / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)
    motion_record: Path | None = None
    if result.phase == "hold":
        if evidence_factory is not None or result.trials:
            raise ValidationError("hold evidence cannot contain motion artifacts")
        details: dict[str, Any] = {
            "held_action": dict(result.held_action or {}),
            "motion": False,
        }
    else:
        if evidence_factory is None or evidence_factory.root.parent != root:
            raise ValidationError("motion evidence factory does not belong to this material root")
        motion_record = evidence_factory.root
        details = _finalize_trials(result, motion_record)

    session_payload = _session_payload(session)
    candidate_payload = candidate.payload
    repo = {
        "commit": identity.repository_commit,
        "clean": True,
        "hostname": identity.hostname,
        "python": identity.python_version,
    }
    run_seed = hashlib.sha256(
        canonical_json_bytes(
            {
                "session": session.bundle_id,
                "policy": candidate.bundle_id,
                "phase": result.phase,
                "started_at": result.started_at,
                "repo": repo,
            }
        )
    ).hexdigest()
    wandb = planned_run(
        wandb_entity,
        wandb_project,
        f"execute-{result.policy}-{result.phase}-{run_seed[:12]}",
    )
    payload = {
        "schema_version": 1,
        "kind": "rollout_evidence",
        "session_id": result.session_id,
        "session_bundle_id": session.bundle_id,
        "session_content_id": session.content_id,
        "session_manifest_sha256": session.manifest["lineage"]["session_manifest_sha256"],
        "policy_bundle_id": candidate.bundle_id,
        "policy_content_id": candidate.content_id,
        "prior_phase_bundles": prior,
        "evaluation_sha256": candidate_payload["evaluation_sha256"],
        "checkpoint_inventory_sha256": candidate_payload["checkpoint_inventory_sha256"],
        "dataset_release_id": candidate_payload["dataset_release_id"],
        "policy": result.policy,
        "experiment": experiment,
        "phase": result.phase,
        "status": "completed",
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "details": details,
        "repo": repo,
        "wandb": wandb.binding(),
    }
    payload_path = write_canonical_json(payload_root / "rollout_evidence.json", payload)
    publisher(
        wandb,
        job_type="viola-policy-execute",
        config={
            "session_bundle_id": session.bundle_id,
            "session_id": result.session_id,
            "policy_bundle_id": candidate.bundle_id,
            "policy": result.policy,
            "experiment": experiment,
            "phase": result.phase,
            "prior_phase_bundles": prior,
            "repo": repo,
            "operator": session_payload["operator"],
        },
        summary={
            "status": "completed",
            "trial_count": len(result.trials),
            "task_successes": details.get("task_successes", 0),
            "safety_event_count": 0,
        },
    )
    sync = {
        "schema_version": 1,
        "operation": "policy_execute",
        "evidence_file": "rollout_evidence.json",
        "evidence_sha256": sha256_file(payload_path),
        "wandb": wandb.binding(),
        "binding": {
            name: payload[name]
            for name in (
                "session_id",
                "session_bundle_id",
                "policy_bundle_id",
                "policy",
                "experiment",
                "phase",
                "status",
                "prior_phase_bundles",
            )
        },
        "synced_at": datetime.now(UTC).isoformat(),
    }
    sync_path = payload_root / "WANDB_SYNCED.json"
    if sync_path.is_file():
        existing_sync = read_json_object(sync_path, label="rollout W&B sync receipt")
        expected_without_time = {key: value for key, value in sync.items() if key != "synced_at"}
        actual_without_time = {
            key: value for key, value in existing_sync.items() if key != "synced_at"
        }
        if expected_without_time != actual_without_time:
            raise ValidationError("existing rollout W&B sync receipt differs from this phase")
    else:
        write_canonical_json(sync_path, sync)
    lineage = _lineage(payload, wandb.run_id, prior)
    request = SealRequest(
        root=handoff_root,
        kind="rollout_evidence",
        experiment=experiment,
        subject=result.session_id,
        producer=identity,
        lineage=lineage,
        wandb_project=wandb_project,
        payload_dir=payload_root,
        artifact_roots={} if motion_record is None else {"motion_record": motion_record},
    )
    logger = handoff_logger or WandbEvidenceLogger(entity=wandb_entity)
    bundle = seal_bundle(request, evidence_logger=logger)
    return SealedRolloutEvidence(bundle, payload_path, sync_path, root, motion_record)


def seal_unsafe_phase(
    result: PhaseResult,
    *,
    session: VerifiedBundle,
    candidate: AcceptedPolicyCandidate,
    identity: RuntimeIdentity,
    experiment: str,
    material_root: str | Path,
    handoff_root: str | Path,
    wandb_entity: str,
    wandb_project: str,
    evidence_factory: PhaseEvidenceFactory,
    prior_hold: VerifiedBundle | None = None,
    prior_shakedown: VerifiedBundle | None = None,
    publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
    handoff_logger: EvidenceLogger | None = None,
) -> SealedRolloutEvidence:
    """Publish one aborted shakedown as terminal evidence.

    The resulting bundle has ``evidence_only`` permission.  The live gate also
    requires predecessor payloads to say ``status=completed``, so this READY
    bundle can be accepted and aggregated by Repo B but can never authorize a
    later motion phase.
    """

    if result.status != "unsafe_shakedown" or result.phase != "shakedown":
        raise ValidationError(
            "unsafe READY finalization currently supports shakedown only; Repo B 6fcf643 "
            "requires incompatible completed-shakedown predecessor schemas for scored "
            "terminal evidence"
        )
    if result.terminal_event not in SAFETY_EVENTS or not result.terminal_reason:
        raise ValidationError("unsafe rollout lacks its typed terminal event and reason")

    session_payload = _session_payload(session)
    if result.session_id != session_payload["session_id"]:
        raise ValidationError("unsafe phase result belongs to another rollout session")
    if candidate.bundle_id != result_policy_bundle(session, result.policy):
        raise ValidationError("unsafe phase result belongs to another policy candidate")
    if identity.role != "pc_a" or identity.repository_clean is not True:
        raise ValidationError("unsafe rollout evidence requires a clean pc_a identity")
    if identity.repository_commit != session_payload["executor"]["repository_commit"]:
        raise ValidationError("unsafe rollout commit differs from the reviewed executor")
    if {experiment, session.manifest["experiment"], candidate.bundle.manifest["experiment"]} != {
        experiment
    }:
        raise ValidationError("unsafe rollout evidence crosses experiment namespaces")

    prior = _prior_references(result.phase, prior_hold, prior_shakedown)
    root = Path(material_root).expanduser().resolve()
    motion_record = evidence_factory.root
    if motion_record.parent != root:
        raise ValidationError("unsafe motion evidence does not belong to this material root")
    details = _finalize_unsafe_trials(result, motion_record)

    repo = {
        "commit": identity.repository_commit,
        "clean": True,
        "hostname": identity.hostname,
        "python": identity.python_version,
    }
    run_seed = hashlib.sha256(
        canonical_json_bytes(
            {
                "session": session.bundle_id,
                "policy": candidate.bundle_id,
                "phase": result.phase,
                "started_at": result.started_at,
                "repo": repo,
            }
        )
    ).hexdigest()
    wandb = planned_run(
        wandb_entity,
        wandb_project,
        f"execute-{result.policy}-{result.phase}-{run_seed[:12]}",
    )
    candidate_payload = candidate.payload
    payload = {
        "schema_version": 1,
        "kind": "rollout_evidence",
        "session_id": result.session_id,
        "session_bundle_id": session.bundle_id,
        "session_content_id": session.content_id,
        "session_manifest_sha256": session.manifest["lineage"][
            "session_manifest_sha256"
        ],
        "policy_bundle_id": candidate.bundle_id,
        "policy_content_id": candidate.content_id,
        "prior_phase_bundles": prior,
        "evaluation_sha256": candidate_payload["evaluation_sha256"],
        "checkpoint_inventory_sha256": candidate_payload[
            "checkpoint_inventory_sha256"
        ],
        "dataset_release_id": candidate_payload["dataset_release_id"],
        "policy": result.policy,
        "experiment": experiment,
        "phase": result.phase,
        "status": "unsafe_shakedown",
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "details": details,
        "repo": repo,
        "wandb": wandb.binding(),
    }
    payload_root = root / "payload"
    payload_path = write_canonical_json(
        payload_root / "rollout_evidence.json", payload
    )
    publisher(
        wandb,
        job_type="viola-policy-execute",
        config={
            "session_bundle_id": session.bundle_id,
            "session_id": result.session_id,
            "policy_bundle_id": candidate.bundle_id,
            "policy": result.policy,
            "experiment": experiment,
            "phase": result.phase,
            "prior_phase_bundles": prior,
            "repo": repo,
            "operator": session_payload["operator"],
        },
        summary={
            "status": "unsafe_shakedown",
            "trial_count": len(result.trials),
            "task_successes": details["task_successes"],
            "safety_event_count": sum(details["safety_event_counts"].values()),
            "terminal_event": result.terminal_event,
        },
    )
    sync = {
        "schema_version": 1,
        "operation": "policy_execute",
        "evidence_file": "rollout_evidence.json",
        "evidence_sha256": sha256_file(payload_path),
        "wandb": wandb.binding(),
        "binding": {
            name: payload[name]
            for name in (
                "session_id",
                "session_bundle_id",
                "policy_bundle_id",
                "policy",
                "experiment",
                "phase",
                "status",
                "prior_phase_bundles",
            )
        },
        "synced_at": datetime.now(UTC).isoformat(),
    }
    sync_path = payload_root / "WANDB_SYNCED.json"
    if sync_path.is_file():
        existing_sync = read_json_object(sync_path, label="unsafe rollout W&B sync receipt")
        expected_without_time = {key: value for key, value in sync.items() if key != "synced_at"}
        actual_without_time = {
            key: value for key, value in existing_sync.items() if key != "synced_at"
        }
        if expected_without_time != actual_without_time:
            raise ValidationError("existing unsafe W&B sync receipt differs from this phase")
    else:
        write_canonical_json(sync_path, sync)
    request = SealRequest(
        root=handoff_root,
        kind="rollout_evidence",
        experiment=experiment,
        subject=result.session_id,
        producer=identity,
        lineage=_lineage(payload, wandb.run_id, prior),
        wandb_project=wandb_project,
        payload_dir=payload_root,
        artifact_roots={"motion_record": motion_record},
        permission="evidence_only",
    )
    logger = handoff_logger or WandbEvidenceLogger(entity=wandb_entity)
    bundle = seal_bundle(request, evidence_logger=logger)
    return SealedRolloutEvidence(bundle, payload_path, sync_path, root, motion_record)


def result_policy_bundle(session: VerifiedBundle, policy: str) -> str:
    payload = _session_payload(session)
    if session.manifest["lineage"].get("policy") != policy:
        raise ValidationError("rollout session policy differs from execution result")
    return str(payload["policy_bundle_id"])


def _finalize_trials(result: PhaseResult, motion_root: Path) -> dict[str, Any]:
    expected = 2 if result.phase == "shakedown" else 10
    if len(result.trials) != expected:
        raise ValidationError(f"completed {result.phase} requires exactly {expected} trials")
    (motion_root / "trials").mkdir(exist_ok=True)
    (motion_root / "traces").mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    counts = {name: 0 for name in SAFETY_EVENTS}
    for trial in result.trials:
        if trial.safety_events:
            raise ValidationError("completed phase cannot contain a safety event")
        records.append(_finalize_trial(result, trial, motion_root))
        for event in trial.safety_events:
            counts[event] += 1
    return {
        "session_id": result.session_id,
        "trials": records,
        "trial_count": len(records),
        "task_successes": sum(int(record["success"]) for record in records),
        "safety_event_counts": counts,
        "speed_scale": result.speed_scale,
    }


def _finalize_trial(
    phase: PhaseResult,
    trial: TrialResult,
    motion_root: Path,
) -> dict[str, Any]:
    trial.outcome.validate()
    if not 59.0 <= trial.duration_sec <= 61.0:
        raise ValidationError("completed trial must span the reviewed 60-second window")
    rich_path = motion_root / "safety_traces" / f"trial-{trial.index:02d}.jsonl"
    rows = read_trace(rich_path)
    if len(rows) != trial.actions or not rows:
        raise ValidationError("rich trace row count differs from sent actions")
    compact: list[dict[str, Any]] = []
    for action_index, row in enumerate(rows):
        if row.get("index") != action_index or "event" in row:
            raise ValidationError("completed trial rich trace contains a terminal or reordered row")
        camera = row["camera_freshness"]
        ages = [float(value) for value in camera["frame_age_ms"].values()]
        compact.append(
            {
                "trial": trial.index,
                "condition": dict(trial.condition),
                "index": action_index,
                "elapsed_ms": float(row["control_ms"]),
                "observation_age_ms": max(ages),
                "replan": bool(row["replan"]),
                "state": _ordered_action(row["state"]),
                "front_sha256": row["front_sha256"],
                "up_sha256": row["up_sha256"],
                "proposed_action": _ordered_action(row["proposed_action"]),
                "sent_action": _ordered_action(row["sent_action"]),
                "feedback_action": _ordered_action(row["feedback_action"]),
                "inference_ms": float(row["inference_ms"]),
            }
        )
    trace_path = motion_root / "traces" / f"trial-{trial.index:02d}.jsonl"
    _write_bytes_once(
        trace_path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in compact),
    )
    front = motion_root / "videos" / f"trial-{trial.index:02d}-front.mp4"
    up = motion_root / "videos" / f"trial-{trial.index:02d}-up.mp4"
    inference = [float(row["inference_ms"]) for row in compact]
    control = [float(row["elapsed_ms"]) for row in compact]
    events = list(trial.safety_events)
    counters = Counter(events)
    record = {
        "session_id": phase.session_id,
        "trial_id": trial.trial_id,
        "policy": phase.policy,
        "index": trial.index,
        "schedule_index": trial.index if phase.phase == "scored" else None,
        "execution_index": trial.index,
        "speed_scale": phase.speed_scale,
        "scored": phase.phase == "scored",
        "condition": dict(trial.condition),
        "started_at": trial.started_at,
        "completed_at": trial.completed_at,
        "duration_sec": trial.duration_sec,
        "completion_time_sec": trial.outcome.completion_time_sec,
        "blue_completed_sec": trial.outcome.blue_completed_sec,
        "red_completed_sec": trial.outcome.red_completed_sec,
        "stable_duration_sec": trial.outcome.stable_duration_sec,
        "actions": len(compact),
        "replans": sum(int(row["replan"]) for row in compact),
        "success": trial.outcome.success,
        "outcome": trial.outcome.outcome,
        "failure_code": trial.outcome.failure_code,
        "safety_events": events,
        "collisions": counters["collision"],
        "interventions": counters["intervention"],
        "safety_aborts": sum(
            counters[name]
            for name in (
                "safety_abort",
                "clamped_action",
                "stale_observation",
                "feedback_loss",
                "operator_abort",
            )
        ),
        "malformed_actions": counters["malformed_action"],
        "deadline_misses": counters["deadline_miss"],
        "recorded_at_utc": trial.completed_at,
        "inference_latency_ms": _latency(inference),
        "control_latency_ms": _latency(control),
        "trace": {
            "path": trace_path.relative_to(motion_root).as_posix(),
            "sha256": sha256_file(trace_path),
        },
        "videos": {
            "front": {
                "path": front.relative_to(motion_root).as_posix(),
                "sha256": sha256_file(front),
            },
            "up": {"path": up.relative_to(motion_root).as_posix(), "sha256": sha256_file(up)},
        },
    }
    write_canonical_json(motion_root / "trials" / f"trial-{trial.index:02d}.json", record)
    return record


def _finalize_unsafe_trials(result: PhaseResult, motion_root: Path) -> dict[str, Any]:
    """Derive Repo B's rich terminal view without changing the raw trace."""

    expected = 2 if result.phase == "shakedown" else 10
    if not 1 <= len(result.trials) <= expected:
        raise ValidationError(
            f"unsafe {result.phase} must retain one through {expected} attempted trials"
        )
    if result.trials[-1].safety_events != (result.terminal_event,):
        raise ValidationError("unsafe final trial differs from the phase terminal event")
    if any(trial.safety_events for trial in result.trials[:-1]):
        raise ValidationError("only the final unsafe trial may contain a safety event")

    (motion_root / "trials").mkdir(exist_ok=True)
    (motion_root / "traces").mkdir(exist_ok=True)
    records = [
        _finalize_rich_trial(
            result,
            trial,
            motion_root,
            terminal=trial is result.trials[-1],
        )
        for trial in result.trials
    ]
    counts = {name: 0 for name in SAFETY_EVENTS}
    for trial in result.trials:
        for event in trial.safety_events:
            counts[event] += 1
    return {
        "failure_type": result.terminal_event,
        "failure_reason": result.terminal_reason,
        "motion_aborted": True,
        "session_id": result.session_id,
        "trials": records,
        "trial_count": len(records),
        "task_successes": sum(int(record["success"]) for record in records),
        "safety_event_counts": counts,
        "speed_scale": result.speed_scale,
    }


def _finalize_rich_trial(
    phase: PhaseResult,
    trial: TrialResult,
    motion_root: Path,
    *,
    terminal: bool,
) -> dict[str, Any]:
    """Project one lossless control trace into Repo B's rich trial schema."""

    trial.outcome.validate()
    raw_path = motion_root / "safety_traces" / f"trial-{trial.index:02d}.jsonl"
    raw_rows = read_trace(raw_path)
    event_rows = [row for row in raw_rows if "event" in row]
    frame_rows = [row for row in raw_rows if "event" not in row]
    if terminal:
        if len(event_rows) != 1 or raw_rows[-1] is not event_rows[0]:
            raise ValidationError("unsafe trial must end with exactly one terminal trace row")
        if event_rows[0].get("event") != phase.terminal_event:
            raise ValidationError("raw terminal row differs from the phase safety event")
    elif event_rows:
        raise ValidationError("a completed pre-terminal trial contains a terminal trace row")

    normal_rows: list[dict[str, Any]] = []
    failed_frame: Mapping[str, Any] | None = None
    for attempt_index, raw in enumerate(frame_rows):
        _validate_raw_frame_identity(raw, trial=trial, attempt_index=attempt_index)
        if _is_verified_control_row(raw):
            projected = _project_normal_trace_row(
                raw,
                trial=trial,
                row_index=len(normal_rows),
                previous_elapsed=(
                    float(normal_rows[-1]["elapsed_s"]) if normal_rows else 0.0
                ),
            )
        else:
            if not terminal or failed_frame is not None:
                raise ValidationError("only the terminal frame may lack a verified write")
            failed_frame = raw
            continue
        if failed_frame is not None:
            raise ValidationError("a verified action follows the terminal write attempt")
        normal_rows.append(projected)

    if not terminal and failed_frame is not None:
        raise ValidationError("completed pre-terminal trial contains an unverified write")
    if terminal and len(frame_rows) - len(normal_rows) > 1:
        raise ValidationError("unsafe trial contains more than one unverified write attempt")
    if terminal:
        _validate_raw_terminal_trace(
            event_rows[0],
            phase=phase,
            trial=trial,
            frame_rows=frame_rows,
        )
    verified = len(normal_rows)
    captured = len(frame_rows)
    if not verified <= trial.actions <= captured <= verified + int(terminal):
        raise ValidationError(
            "unsafe trial action, captured-observation, and verified-action counts disagree"
        )
    if not terminal and (
        trial.actions != _EXPECTED_ACTIONS
        or verified != _EXPECTED_ACTIONS
        or not 59.0 <= trial.duration_sec <= 61.0
    ):
        raise ValidationError("completed pre-terminal trial lacks full safe control coverage")
    if terminal and captured == 0:
        _validate_terminal_only_trace(
            event_rows[0],
            phase=phase,
            trial=trial,
            motion_root=motion_root,
        )
        raise UnsafeTerminalVideoUnavailableError(
            "unsafe shakedown stopped before a transferable camera frame was retained"
        )

    videos = {
        camera: _video_metadata(
            motion_root / "videos" / f"trial-{trial.index:02d}-{camera}.mp4",
            relative=f"videos/trial-{trial.index:02d}-{camera}.mp4",
        )
        for camera in ("front", "up")
    }
    if (
        videos["front"]["frame_count"] != captured
        or videos["up"]["frame_count"] != captured
    ):
        raise ValidationError("unsafe trace and front/up video frame counts differ")

    trace_rows = list(normal_rows)
    if terminal:
        trace_rows.append(
            _project_terminal_trace_row(
                event_rows[0],
                failed_frame=failed_frame,
                trial=trial,
                verified_actions=len(normal_rows),
                captured_observations=captured,
                previous_elapsed=(
                    float(normal_rows[-1]["elapsed_s"]) if normal_rows else 0.0
                ),
            )
        )
    trace_path = motion_root / "traces" / f"trial-{trial.index:02d}.jsonl"
    _write_bytes_once(
        trace_path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in trace_rows),
    )

    inference = [float(row["inference_ms"]) for row in normal_rows]
    control = [float(row["control_ms"]) for row in normal_rows]
    deadline_misses = sum(int(not row["deadline_passed"]) for row in normal_rows)
    rate_misses = sum(int(not row["rate_passed"]) for row in normal_rows)
    if terminal and phase.terminal_event == "deadline_miss":
        deadline_misses = max(deadline_misses, 1)
    events = list(trial.safety_events)
    counters = Counter(events)
    record = {
        "index": trial.index,
        "condition": dict(trial.condition),
        "started_at": trial.started_at,
        "completed_at": trial.completed_at,
        "duration_sec": trial.duration_sec,
        "expected_duration_sec": TRIAL_DURATION_S,
        "actions": trial.actions,
        "verified_actions": verified,
        "captured_observations": captured,
        "expected_actions": _EXPECTED_ACTIONS,
        "minimum_actions": _MINIMUM_ACTIONS,
        "action_coverage": verified / _EXPECTED_ACTIONS,
        "required_action_coverage": 0.95,
        "replans": sum(int(row["replan"]) for row in normal_rows),
        "control_deadline_misses": deadline_misses,
        "rate_misses": rate_misses,
        "target_hz": TARGET_HZ,
        "target_period_ms": _TARGET_PERIOD_MS,
        "rate_tolerance_ms": RATE_TOLERANCE_MS,
        "completed_control": not terminal,
        "inference_latency_ms": _latency_or_zero(inference),
        "control_latency_ms": _latency_or_zero(control),
        "recorded_at_utc": trial.completed_at,
        "session_id": phase.session_id,
        "trial_id": trial.trial_id,
        "policy": phase.policy,
        "schedule_index": trial.index if phase.phase == "scored" else None,
        "execution_index": trial.index,
        "speed_scale": phase.speed_scale,
        "scored": phase.phase == "scored",
        "success": trial.outcome.success,
        "outcome": "success" if trial.outcome.success else "failure",
        "failure_code": trial.outcome.failure_code,
        "blue_completed_sec": trial.outcome.blue_completed_sec,
        "red_completed_sec": trial.outcome.red_completed_sec,
        "completion_time_sec": trial.outcome.completion_time_sec,
        "stable_duration_sec": trial.outcome.stable_duration_sec,
        "safety_events": events,
        "collisions": counters["collision"],
        "interventions": counters["intervention"],
        "safety_aborts": sum(
            counters[name]
            for name in (
                "safety_abort",
                "clamped_action",
                "stale_observation",
                "feedback_loss",
                "operator_abort",
            )
        ),
        "malformed_actions": counters["malformed_action"],
        "deadline_misses": counters["deadline_miss"],
        "trace": {
            "path": trace_path.relative_to(motion_root).as_posix(),
            "sha256": sha256_file(trace_path),
            "rows": len(trace_rows),
        },
        "videos": videos,
    }
    write_canonical_json(
        motion_root / "trials" / f"trial-{trial.index:02d}.json", record
    )
    return record


def _project_normal_trace_row(
    raw: Mapping[str, Any],
    *,
    trial: TrialResult,
    row_index: int,
    previous_elapsed: float,
) -> dict[str, Any]:
    """Return one verified action in Repo B's exact rich trace shape."""

    if (
        raw.get("index") != row_index
        or raw.get("condition") != trial.condition
        or raw.get("replan") is not (row_index % 10 == 0)
    ):
        raise ValidationError("control row is not a fully verified action")
    elapsed = _number(raw.get("elapsed_s"), "trace elapsed_s")
    control_ms = _number(raw.get("control_ms"), "control_ms")
    period_ms = _number(raw.get("period_ms"), "period_ms")
    if elapsed <= previous_elapsed or abs(period_ms - (elapsed - previous_elapsed) * 1_000) > 1e-6:
        raise ValidationError("control row timing is not monotonic and period-exact")
    deadline_passed = control_ms < CONTROL_DEADLINE_MS
    rate_passed = period_ms <= _TARGET_PERIOD_MS + RATE_TOLERANCE_MS
    if (
        raw.get("deadline_ms") != CONTROL_DEADLINE_MS
        or raw.get("deadline_passed") is not deadline_passed
        or raw.get("target_period_ms") != _TARGET_PERIOD_MS
        or raw.get("rate_tolerance_ms") != RATE_TOLERANCE_MS
        or raw.get("rate_passed") is not rate_passed
    ):
        raise ValidationError("control row timing receipt was not recomputed exactly")
    return {
        "trial": trial.index,
        "condition": dict(trial.condition),
        "index": row_index,
        "elapsed_s": elapsed,
        "observation_read_ms": _number(
            raw.get("observation_read_ms"), "observation_read_ms"
        ),
        "camera_freshness": _json_copy(raw.get("camera_freshness")),
        "replan": raw["replan"],
        "state": _ordered_action(_mapping(raw.get("state"), "trace state")),
        "front_sha256": _digest(raw.get("front_sha256"), "front frame"),
        "up_sha256": _digest(raw.get("up_sha256"), "up frame"),
        "proposed_action": _ordered_action(
            _mapping(raw.get("proposed_action"), "proposed action")
        ),
        "sent_action": _ordered_action(
            _mapping(raw.get("sent_action"), "sent action")
        ),
        "feedback_action": _ordered_action(
            _mapping(raw.get("feedback_action"), "feedback action")
        ),
        "feedback_check": _project_feedback_check(raw["feedback_check"]),
        "limit_check": _project_limit_check(raw["limit_check"]),
        "inference_ms": _number(raw.get("inference_ms"), "inference_ms"),
        "control_ms": control_ms,
        "deadline_ms": CONTROL_DEADLINE_MS,
        "deadline_passed": deadline_passed,
        "period_ms": period_ms,
        "target_period_ms": _TARGET_PERIOD_MS,
        "rate_tolerance_ms": RATE_TOLERANCE_MS,
        "rate_passed": rate_passed,
    }


def _is_verified_control_row(raw: Mapping[str, Any]) -> bool:
    """Identify a confirmed write with complete passing safety receipts."""

    feedback = raw.get("feedback_check")
    limits = raw.get("limit_check")
    return (
        raw.get("write_outcome") == "confirmed"
        and raw.get("sent_action") == raw.get("proposed_action")
        and isinstance(feedback, Mapping)
        and feedback.get("passed") is True
        and set(feedback) == _FEEDBACK_FIELDS
        and isinstance(limits, Mapping)
        and limits.get("passed") is True
    )


def _validate_raw_frame_identity(
    raw: Mapping[str, Any],
    *,
    trial: TrialResult,
    attempt_index: int,
) -> None:
    """Bind every retained camera frame to its exact production attempt."""

    if (
        raw.get("trial") != trial.trial_id
        or raw.get("condition") != trial.condition
        or raw.get("index") != attempt_index
    ):
        raise ValidationError("unsafe frame row differs from its trial identity")


def _validate_raw_terminal_trace(
    raw: Mapping[str, Any],
    *,
    phase: PhaseResult,
    trial: TrialResult,
    frame_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile the lossless terminal receipt with frames and PhaseResult."""

    if set(raw) != _RAW_TERMINAL_FIELDS:
        raise ValidationError("terminal trace has an unexpected shape")
    counters = {
        name: raw[name]
        for name in ("actions_sent", "action_attempts", "ambiguous_write_attempts")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters.values()
    ):
        raise ValidationError("terminal trace counters must be nonnegative integers")

    outcomes = [row.get("write_outcome") for row in frame_rows]
    if any(outcome not in {"confirmed", "unknown"} for outcome in outcomes):
        raise ValidationError("unsafe frame has an unknown motor-write outcome")
    expected_attempts = len(frame_rows)
    expected_actions = outcomes.count("confirmed")
    expected_ambiguous = outcomes.count("unknown")
    detail = raw["detail"]
    if (
        raw["trial"] != trial.trial_id
        or raw["condition"] != trial.condition
        or raw["index"] != expected_attempts
        or raw["event"] != phase.terminal_event
        or not isinstance(detail, str)
        or not detail
        or phase.terminal_reason != detail
        or counters
        != {
            "actions_sent": expected_actions,
            "action_attempts": expected_attempts,
            "ambiguous_write_attempts": expected_ambiguous,
        }
        or trial.actions != expected_actions
    ):
        raise ValidationError("terminal trace differs from its phase result or frame attempts")

    elapsed = _number(raw["elapsed_s"], "terminal elapsed_s")
    frame_elapsed = [
        _number(row.get("elapsed_s"), "frame elapsed_s") for row in frame_rows
    ]
    if (
        (frame_elapsed and elapsed < max(frame_elapsed))
        or abs(elapsed - trial.duration_sec) > 0.25
    ):
        raise ValidationError("terminal trace elapsed time differs from retained execution")


def _project_terminal_trace_row(
    raw_terminal: Mapping[str, Any],
    *,
    failed_frame: Mapping[str, Any] | None,
    trial: TrialResult,
    verified_actions: int,
    captured_observations: int,
    previous_elapsed: float,
) -> dict[str, Any]:
    """Describe the one terminal point without inventing a confirmed write."""

    proposed = sent = feedback = limit = feedback_check = None
    timing = _empty_terminal_timing()
    stage = {
        "stale_observation": "observation",
        "malformed_action": "inference",
        "clamped_action": "inference",
        "feedback_loss": "write_feedback",
        "safety_abort": "write_feedback",
        "deadline_miss": "write_feedback",
    }.get(str(raw_terminal.get("event")), "start")
    if failed_frame is not None:
        proposed = _optional_action(failed_frame.get("proposed_action"))
        if failed_frame.get("write_outcome") == "confirmed":
            sent = _optional_action(failed_frame.get("sent_action"))
        feedback = _optional_action(failed_frame.get("feedback_action"))
        raw_limit = failed_frame.get("limit_check")
        if isinstance(raw_limit, Mapping):
            limit = _project_limit_check(raw_limit)
        raw_feedback = failed_frame.get("feedback_check")
        if (
            isinstance(raw_feedback, Mapping)
            and set(raw_feedback) == _FEEDBACK_FIELDS
            and proposed is not None
            and feedback is not None
            and limit is not None
        ):
            feedback_check = _project_feedback_check(raw_feedback)
        timing = _terminal_timing(failed_frame)
        stage = "write_feedback"

    _number(raw_terminal.get("elapsed_s"), "terminal elapsed_s")
    # Repo B permits a 250 ms difference between terminal monotonic time and
    # the wall-derived duration.  Prefer the retained duration for a stable
    # retry and only move by one float when a zero-duration abort occurred.
    elapsed = max(
        trial.duration_sec,
        math.nextafter(previous_elapsed, math.inf),
        math.nextafter(0.0, math.inf),
    )
    return {
        "trial": trial.index,
        "condition": dict(trial.condition),
        "index": verified_actions,
        "elapsed_s": elapsed,
        "event": raw_terminal["event"],
        "detail": str(raw_terminal.get("detail") or "safety event"),
        "stage": stage,
        "captured_observations": captured_observations,
        "actions_sent": trial.actions,
        "verified_actions": verified_actions,
        "proposed_action": proposed,
        "sent_action": sent,
        "feedback_action": feedback,
        "feedback_check": feedback_check,
        "limit_check": limit,
        "timing": timing,
    }


def _validate_terminal_only_trace(
    raw: Mapping[str, Any],
    *,
    phase: PhaseResult,
    trial: TrialResult,
    motion_root: Path,
) -> None:
    """Prove this is the production writer's genuine pre-frame terminal shape."""

    if set(raw) != _RAW_TERMINAL_FIELDS:
        raise ValidationError("pre-frame terminal trace has an unexpected shape")
    elapsed = _number(raw["elapsed_s"], "pre-frame terminal elapsed_s")
    counts = {
        name: raw[name]
        for name in ("actions_sent", "action_attempts", "ambiguous_write_attempts")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValidationError("pre-frame terminal counters must be nonnegative integers")
    if (
        raw["trial"] != trial.trial_id
        or raw["condition"] != trial.condition
        or raw["index"] != 0
        or raw["event"] != phase.terminal_event
        or not isinstance(raw["detail"], str)
        or not raw["detail"]
        or counts != {
            "actions_sent": 0,
            "action_attempts": 0,
            "ambiguous_write_attempts": 0,
        }
        or trial.actions != 0
        or trial.replans != 0
        or trial.inference_latency_ms
        or trial.control_latency_ms
        or abs(elapsed - trial.duration_sec) > 0.25
    ):
        raise ValidationError("pre-frame terminal trace differs from its phase result")
    for camera in ("front", "up"):
        video = motion_root / "videos" / f"trial-{trial.index:02d}-{camera}.mp4"
        if video.exists() or video.is_symlink():
            raise ValidationError("pre-frame terminal unexpectedly contains a video file")
    payload_root = motion_root.parent / "payload"
    if payload_root.exists() or payload_root.is_symlink():
        raise ValidationError("pre-frame terminal contains partial sealed payload material")
    for relative in (
        f"traces/trial-{trial.index:02d}.jsonl",
        f"trials/trial-{trial.index:02d}.json",
    ):
        derived = motion_root / relative
        if derived.exists() or derived.is_symlink():
            raise ValidationError("pre-frame terminal contains partial derived evidence")


def _prior_references(
    phase: str,
    hold: VerifiedBundle | None,
    shakedown: VerifiedBundle | None,
) -> dict[str, Any]:
    if phase == "hold":
        if hold is not None or shakedown is not None:
            raise ValidationError("hold cannot bind predecessor evidence")
        return {"hold": None, "shakedown": None}
    if hold is None:
        raise ValidationError(f"{phase} requires completed hold evidence")
    hold_ref = {"bundle_id": hold.bundle_id, "content_id": hold.content_id}
    if phase == "shakedown":
        if shakedown is not None:
            raise ValidationError("shakedown cannot bind a shakedown predecessor")
        return {"hold": hold_ref, "shakedown": None}
    if phase != "scored" or shakedown is None:
        raise ValidationError("scored execution requires hold and shakedown evidence")
    return {
        "hold": hold_ref,
        "shakedown": {
            "bundle_id": shakedown.bundle_id,
            "content_id": shakedown.content_id,
        },
    }


def _lineage(
    payload: Mapping[str, Any], run_id: str, prior: Mapping[str, Any]
) -> dict[str, Any]:
    value = {
        "session_id": payload["session_id"],
        "rollout_session_bundle_id": payload["session_bundle_id"],
        "rollout_session_content_id": payload["session_content_id"],
        "session_manifest_sha256": payload["session_manifest_sha256"],
        "policy_candidate_bundle_id": payload["policy_bundle_id"],
        "policy_candidate_content_id": payload["policy_content_id"],
        "evaluation_sha256": payload["evaluation_sha256"],
        "checkpoint_inventory_sha256": payload["checkpoint_inventory_sha256"],
        "dataset_release_id": payload["dataset_release_id"],
        "policy": payload["policy"],
        "phase": payload["phase"],
        "repo_a_commit": payload["repo"]["commit"],
        "wandb_run_id": run_id,
    }
    if prior["hold"] is not None:
        value.update(
            {
                "hold_rollout_evidence_bundle_id": prior["hold"]["bundle_id"],
                "hold_rollout_evidence_content_id": prior["hold"]["content_id"],
            }
        )
    if prior["shakedown"] is not None:
        value.update(
            {
                "shakedown_rollout_evidence_bundle_id": prior["shakedown"]["bundle_id"],
                "shakedown_rollout_evidence_content_id": prior["shakedown"]["content_id"],
            }
        )
    return value


def _session_payload(bundle: VerifiedBundle) -> dict[str, Any]:
    try:
        value = json.loads(bundle.payload_file("rollout_session.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read rollout session payload: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("rollout session payload must be an object")
    return value


def _ordered_action(value: Mapping[str, Any]) -> list[float]:
    if set(value) != set(ACTION_KEYS):
        raise ValidationError("trace action does not name the seven ordered joints")
    result = [float(value[key]) for key in ACTION_KEYS]
    if any(not math.isfinite(item) for item in result):
        raise ValidationError("trace action contains a non-finite value")
    return result


def _ordered_joint_values(value: Any, label: str) -> list[float]:
    mapping = _mapping(value, label)
    if set(mapping) != set(_JOINTS):
        raise ValidationError(f"{label} does not name the seven ordered Viola joints")
    result = [float(mapping[joint]) for joint in _JOINTS]
    if any(not math.isfinite(item) for item in result):
        raise ValidationError(f"{label} contains a non-finite value")
    return result


def _project_limit_check(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert keyed internal limits to Repo B's ordered portable vectors."""

    expected = {
        "joint_order",
        "reference_action",
        "lower",
        "upper",
        "reviewed_max_step_deltas",
        "speed_scale",
        "allowed_step_deltas",
        "absolute_violations",
        "rate_violations",
        "would_be_clamped",
        "passed",
        "proposed_action",
    }
    if set(value) != expected:
        raise ValidationError("internal limit receipt has an unexpected shape")
    return {
        "joint_order": list(_JOINTS),
        "reference_action": _ordered_action(
            _mapping(value["reference_action"], "limit reference action")
        ),
        "lower": _ordered_joint_values(value["lower"], "lower limits"),
        "upper": _ordered_joint_values(value["upper"], "upper limits"),
        "reviewed_max_step_deltas": _ordered_joint_values(
            value["reviewed_max_step_deltas"], "reviewed step limits"
        ),
        "speed_scale": _number(value["speed_scale"], "limit speed_scale"),
        "allowed_step_deltas": _ordered_joint_values(
            value["allowed_step_deltas"], "allowed step limits"
        ),
        "absolute_violations": _string_list(
            value["absolute_violations"], "absolute violations"
        ),
        "rate_violations": _string_list(value["rate_violations"], "rate violations"),
        "would_be_clamped": _string_list(
            value["would_be_clamped"], "clamped joints"
        ),
        "passed": _boolean(value["passed"], "limit passed"),
    }


def _project_feedback_check(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one complete synchronous feedback receipt to ordered vectors."""

    if set(value) != _FEEDBACK_FIELDS:
        raise ValidationError("internal feedback receipt has an unexpected shape")
    return {
        "freshness_source": value["freshness_source"],
        "joint_order": list(_JOINTS),
        "command_sequence": value["command_sequence"],
        "previous_feedback_received_ns": value["previous_feedback_received_ns"],
        "control_started_ns": value["control_started_ns"],
        "prewrite_received_ns": value["prewrite_received_ns"],
        "write_started_ns": value["write_started_ns"],
        "write_completed_ns": value["write_completed_ns"],
        "feedback_received_ns": value["feedback_received_ns"],
        "command_deadline_ns": value["command_deadline_ns"],
        "poll_count": value["poll_count"],
        "prewrite_action": _ordered_action(
            _mapping(value["prewrite_action"], "feedback prewrite action")
        ),
        "target_action": _ordered_action(
            _mapping(value["target_action"], "feedback target action")
        ),
        "feedback_action": _ordered_action(
            _mapping(value["feedback_action"], "feedback action")
        ),
        "minimum_observable_progress": _ordered_joint_values(
            value["minimum_observable_progress"], "minimum observable progress"
        ),
        "material_joints": _string_list(value["material_joints"], "material joints"),
        "signed_progress": _ordered_joint_values(
            value["signed_progress"], "signed progress"
        ),
        "target_overshoot": _ordered_joint_values(
            value["target_overshoot"], "target overshoot"
        ),
        "absolute_error": _ordered_joint_values(
            value["absolute_error"], "absolute error"
        ),
        "allowed_error": _ordered_joint_values(value["allowed_error"], "allowed error"),
        "progress_violations": _string_list(
            value["progress_violations"], "progress violations"
        ),
        "tracking_violations": _string_list(
            value["tracking_violations"], "tracking violations"
        ),
        "overshoot_violations": _string_list(
            value["overshoot_violations"], "overshoot violations"
        ),
        "limit_violations": _string_list(value["limit_violations"], "limit violations"),
        "receipt_violations": _string_list(
            value["receipt_violations"], "receipt violations"
        ),
        "violations": _string_list(value["violations"], "feedback violations"),
        "passed": _boolean(value["passed"], "feedback passed"),
    }


def _optional_action(value: Any) -> list[float] | None:
    if value is None:
        return None
    return _ordered_action(_mapping(value, "terminal action"))


def _empty_terminal_timing() -> dict[str, Any]:
    return {
        "observation_read_ms": None,
        "inference_ms": None,
        "prewrite_ms": None,
        "control_ms": None,
        "period_ms": None,
        "deadline_ms": CONTROL_DEADLINE_MS,
        "deadline_passed": None,
        "target_period_ms": _TARGET_PERIOD_MS,
        "rate_tolerance_ms": RATE_TOLERANCE_MS,
        "rate_passed": None,
    }


def _terminal_timing(value: Mapping[str, Any]) -> dict[str, Any]:
    timing = _empty_terminal_timing()
    for name in ("observation_read_ms", "inference_ms", "control_ms", "period_ms"):
        if value.get(name) is not None:
            timing[name] = _number(value[name], f"terminal {name}")
    control = timing["control_ms"]
    period = timing["period_ms"]
    timing["deadline_passed"] = (
        None if control is None else control < CONTROL_DEADLINE_MS
    )
    timing["rate_passed"] = (
        None if period is None else period <= _TARGET_PERIOD_MS + RATE_TOLERANCE_MS
    )
    return timing


def _video_metadata(path: Path, *, relative: str) -> dict[str, Any]:
    """Fully decode one retained camera stream and report exact metadata."""

    try:
        import av

        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1 or len(container.streams.audio) != 0:
                raise ValidationError("rollout MP4 must contain one video stream and no audio")
            stream = streams[0]
            if stream.average_rate is None:
                raise ValidationError("rollout MP4 has no declared frame rate")
            fps = float(stream.average_rate)
            width, height = int(stream.width), int(stream.height)
            frame_count = 0
            previous_pts: int | None = None
            for frame in container.decode(video=0):
                if (int(frame.width), int(frame.height)) != (width, height):
                    raise ValidationError("rollout MP4 changes geometry within its stream")
                if frame.pts is not None:
                    if previous_pts is not None and frame.pts <= previous_pts:
                        raise ValidationError("rollout MP4 timestamps are not increasing")
                    previous_pts = frame.pts
                frame_count += 1
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"could not fully decode rollout video {path}: {exc}") from exc
    if (width, height) != (640, 480) or abs(fps - TARGET_HZ) > 1e-9:
        raise ValidationError("rollout MP4 must be 640x480 at 30 FPS")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps,
        "width": width,
        "height": height,
    }


def _latency_or_zero(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return _latency(values)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (HandoffError, TypeError, ValueError) as exc:
        raise ValidationError(f"evidence value is not canonical JSON data: {exc}") from exc


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValidationError(f"{label} must be finite and nonnegative")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{label} must be boolean")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValidationError(f"{label} must be a string list")
    return list(value)


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _latency(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValidationError("completed trial latency trace is empty")
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValidationError(f"refusing to replace rollout evidence {path}") from exc
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "SealedRolloutEvidence",
    "UnsafeTerminalVideoUnavailableError",
    "seal_completed_phase",
    "seal_unsafe_phase",
]
