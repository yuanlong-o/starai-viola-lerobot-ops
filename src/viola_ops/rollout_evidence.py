"""Finalize completed physical phases into Repo-B-compatible evidence.

The control loop keeps a richer safety trace in ``safety_traces/``.  Repo B's
merged completed-phase consumer still requires its older compact trace, so this
module derives that compatibility view without discarding the richer record.
Unsafe terminal traces are retained locally but are deliberately not sealed
until the cross-repository unsafe/completed schema mismatch is resolved.
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
    RuntimeIdentity,
    SealRequest,
    VerifiedBundle,
    WandbEvidenceLogger,
    canonical_json_bytes,
    seal_bundle,
)

from .errors import ValidationError
from .evidence import PhaseEvidenceFactory, read_trace
from .execution import ACTION_KEYS, SAFETY_EVENTS, PhaseResult, TrialResult
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
            "unsafe motion was retained locally but cannot be sealed: Repo B's merged "
            "completed/unsafe evidence schemas cannot yet validate the same rich trace"
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
        "trace": {"path": trace_path.relative_to(motion_root).as_posix(), "sha256": sha256_file(trace_path)},
        "videos": {
            "front": {"path": front.relative_to(motion_root).as_posix(), "sha256": sha256_file(front)},
            "up": {"path": up.relative_to(motion_root).as_posix(), "sha256": sha256_file(up)},
        },
    }
    write_canonical_json(motion_root / "trials" / f"trial-{trial.index:02d}.json", record)
    return record


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


__all__ = ["SealedRolloutEvidence", "seal_completed_phase"]
