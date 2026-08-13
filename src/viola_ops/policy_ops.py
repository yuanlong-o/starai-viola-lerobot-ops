"""High-level, operator-facing policy verification and shadow commands.

This module is deliberately orchestration, not a framework.  It joins the
readable policy runtime to the immutable handoff and online W&B boundaries.
Replay never imports a camera or robot package.  Live soak lazily imports only
LeRobot's public OpenCV camera API after every signed input has been checked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import viola_handoff

from .errors import ValidationError
from .jsonutil import (
    copy_regular_file,
    read_json_object,
    require_exact_keys,
    sha256_file,
    sha256_json,
    write_canonical_json,
)
from .policies import CANONICAL_TASK, JOINT_NAMES
from .publication_guard import (
    FileSnapshot,
    GuardedEvidenceLogger,
    PublicationSnapshot,
    TreeSnapshot,
    require_sealed_bundle_matches,
    snapshot_file,
    snapshot_tree,
)
from .policy_runtime import (
    AcceptedPolicyCandidate,
    PolicyRuntime,
    REPLAY_DATASET_ARTIFACT,
    RuntimeVerification,
    _snapshot_candidate_runtime,
    build_runtime_terminal_lineage,
    build_runtime_terminal_payload,
    build_sync_receipt,
    build_verification_payload,
    finite_action,
    inspect_candidate,
    load_lerobot_runtime,
    load_replay_dataset,
    load_verification_observation,
    verification_sync_binding,
    verify_policy,
)
from .session_inputs import load_reviewed_setup
from .shadow import (
    CameraFrame,
    CameraSource,
    ReviewedLimits,
    ShadowRun,
    build_shadow_lineage,
    build_shadow_payload,
    run_live_soak,
    run_replay,
    shadow_sync_binding,
    state_vector_sha256,
    write_trace,
)
from .wandb_ops import WandbRunIdentity, planned_run, publish_finished_run

DEFAULT_WANDB_PROJECT = "starai-viola-policy-benchmark"
_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_VERIFICATION_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "bundle_id",
    "content_id",
    "policy",
    "checkpoint",
    "queue_actions",
    "task",
    "feature_map",
    "runtime_binding",
    "sample_action",
    "latency",
    "failure_reason",
    "failure",
    "repo",
    "wandb",
    "verified_at",
}
_SYNC_FIELDS = {
    "schema_version",
    "operation",
    "evidence_file",
    "evidence_sha256",
    "wandb",
    "binding",
    "synced_at",
}
_FROZEN_STATE_FIELDS = {
    "schema_version",
    "state",
    "captured_at",
    "robot_connected_at_capture",
    "motor_disconnected_at",
}
_FROZEN_CAPTURE_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "setup_id",
    "frozen_state_file",
    "frozen_state_sha256",
    "state_sha256",
    "setup_hashes",
    "operator",
    "captured_at",
    "motor_disconnected_at",
    "repo",
    "wandb",
}


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    candidate: AcceptedPolicyCandidate
    result: RuntimeVerification
    root: Path
    verification_path: Path
    sync_path: Path
    terminal_path: Path | None
    terminal_bundle: viola_handoff.VerifiedBundle | None

    def render_text(self) -> str:
        """Summarize the result without exposing the wire-format dictionaries."""

        eligibility = "eligible" if self.result.eligible else "ineligible on PC A"
        lines = [
            f"Policy verification: {eligibility}",
            f"  Policy: {self.candidate.policy}",
            f"  Candidate: {self.candidate.bundle_id}",
            f"  Evidence: {self.verification_path}",
            f"  W&B receipt: {self.sync_path}",
        ]
        if self.result.latency is not None:
            lines.extend(
                [
                    f"  Warm-up actions: {self.result.latency['warmups']}",
                    f"  Timed actions: {self.result.latency['trials']}",
                    f"  p95 latency: {self.result.latency['p95_ms']:.3f} ms",
                ]
            )
        if not self.result.eligible:
            lines.append(f"  Reason: {self.result.failure_reason}")
        if self.terminal_bundle is not None:
            lines.append(f"  Terminal evidence bundle: {self.terminal_bundle.path}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ShadowEvidence:
    candidate: AcceptedPolicyCandidate
    run: ShadowRun
    root: Path
    payload_root: Path
    artifact_root: Path
    bundle: viola_handoff.VerifiedBundle

    def render_text(self) -> str:
        """Return a concise operator summary for replay or camera-only shadow."""

        outcome = (
            "passed"
            if self.run.status == "passed"
            else "unsafe; motion remains blocked"
        )
        return "\n".join(
            [
                f"Policy shadow: {outcome}",
                f"  Policy: {self.candidate.policy}",
                f"  Mode: {self.run.mode}",
                f"  Status: {self.run.status}",
                f"  Proposed actions: {self.run.summary['actions']}",
                f"  Logical duration: {self.run.summary['logical_seconds']:.3f} s",
                f"  Evidence bundle: {self.bundle.path}",
                "  Motion permission: none",
            ]
        )


_EvidenceRootSnapshot = TreeSnapshot
_InputFileSnapshot = FileSnapshot


def verify_command(
    bundle: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
    *,
    handoff_root: str | Path = viola_handoff.DEFAULT_HANDOFF_ROOT,
    runtime_factory: Callable[[AcceptedPolicyCandidate], PolicyRuntime] = load_lerobot_runtime,
    observation_loader: Callable[[AcceptedPolicyCandidate], Mapping[str, Any]] = (
        load_verification_observation
    ),
    publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
    producer_identity: viola_handoff.RuntimeIdentity | None = None,
    bundle_evidence_logger: viola_handoff.EvidenceLogger | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    attempt_id: str | None = None,
) -> VerificationEvidence:
    """Verify one accepted candidate and persist authoritative online evidence."""

    if not wandb_entity:
        raise ValidationError("policy verification requires --wandb-entity")
    candidate = inspect_candidate(bundle)
    identity = producer_identity or viola_handoff.RuntimeIdentity.capture(
        role="pc_a", repo_root=repo_root
    )
    _require_pc_a_identity(identity)
    _require_external_root(output_root, repo_root=repo_root, label="policy evidence")
    _require_external_root(handoff_root, repo_root=repo_root, label="handoff")
    _require_current_candidate(bundle, candidate, handoff_root=handoff_root)
    repo = _repo_identity(identity)
    attempt = _attempt_id(attempt_id)
    material = _material_root(
        output_root,
        candidate=candidate,
        commit=identity.repository_commit,
        operation="verification",
        attempt=attempt,
    )
    replay_source = _snapshot_replay_source(candidate)

    def validate_sources() -> None:
        _require_policy_sources(
            bundle,
            candidate,
            handoff_root=handoff_root,
            repo_root=repo_root,
            identity=identity,
            root_snapshots=(replay_source,),
        )

    with _snapshot_candidate_runtime(candidate) as runtime_inputs:
        observation = observation_loader(candidate)
        result = verify_policy(
            runtime_inputs.candidate,
            observation,
            runtime_factory=runtime_factory,
            clock=clock,
        )
        runtime_inputs.verify()
    validate_sources()
    seed = sha256_json(
        {
            "operation": "policy_verify",
            "bundle_id": candidate.bundle_id,
            "content_id": candidate.content_id,
            "repo_commit": identity.repository_commit,
            "attempt_id": attempt,
        }
    )
    wandb = planned_run(wandb_entity, wandb_project, f"verify-{seed[:16]}")
    verified_at = _utc(now()).isoformat()
    payload = build_verification_payload(
        candidate,
        result,
        repo=repo,
        wandb=wandb.binding(),
        verified_at=verified_at,
    )
    verification_path = write_canonical_json(material / "verification.json", payload)
    evidence_sha256 = sha256_file(verification_path)
    publication_snapshot = _snapshot_evidence_roots({"verification payload": material})
    validate_sources()
    publisher(
        wandb,
        job_type="viola-policy-verify",
        config=_verification_wandb_config(
            candidate,
            payload,
            evidence_sha256=evidence_sha256,
        ),
        summary=_verification_summary(result),
    )
    sync = build_sync_receipt(
        operation="policy_verify",
        evidence_file=verification_path.name,
        evidence_sha256=evidence_sha256,
        wandb=wandb.binding(),
        binding=verification_sync_binding(candidate, payload),
        synced_at=_utc(now()).isoformat(),
    )
    validate_sources()
    _require_evidence_snapshot(publication_snapshot)
    sync_path = write_canonical_json(material / "verification_WANDB_SYNCED.json", sync)
    final_snapshot = _expect_added_files(
        publication_snapshot,
        {sync_path: viola_handoff.canonical_json_bytes(sync)},
    )

    terminal_path: Path | None = None
    terminal_bundle: viola_handoff.VerifiedBundle | None = None
    if not result.eligible:
        terminal = build_runtime_terminal_payload(
            candidate,
            payload,
            verification_sha256=evidence_sha256,
        )
        terminal_path = write_canonical_json(material / "shadow_evidence.json", terminal)
        final_snapshot = _expect_added_files(
            final_snapshot,
            {terminal_path: viola_handoff.canonical_json_bytes(terminal)},
        )
        lineage = build_runtime_terminal_lineage(
            candidate,
            terminal,
            verification_sha256=evidence_sha256,
            verification_sync_sha256=sha256_file(sync_path),
        )
        request = viola_handoff.SealRequest(
            root=handoff_root,
            kind="shadow_evidence",
            experiment=candidate.bundle.manifest["experiment"],
            subject=f"{candidate.policy}:runtime-verification",
            producer=identity,
            lineage=lineage,
            wandb_project=wandb_project,
            payload_dir=material,
        )
        terminal_bundle = _seal_guarded(
            request,
            evidence_logger=bundle_evidence_logger,
            wandb_entity=wandb_entity,
            snapshots=final_snapshot,
            artifact_names={},
            validate_sources=validate_sources,
            repo_root=repo_root,
        )
    validate_sources()
    _require_evidence_snapshot(final_snapshot)
    return VerificationEvidence(
        candidate,
        result,
        material,
        verification_path,
        sync_path,
        terminal_path,
        terminal_bundle,
    )


def shadow_command(
    bundle: str | Path,
    mode: str,
    verification_path: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    handoff_root: str | Path,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
    setup_path: str | Path | None = None,
    frozen_state_path: str | Path | None = None,
    *,
    cameras: Mapping[str, CameraSource] | None = None,
    runtime_factory: Callable[[AcceptedPolicyCandidate], PolicyRuntime] = load_lerobot_runtime,
    replay_loader: Callable[[AcceptedPolicyCandidate], Any] = load_replay_dataset,
    publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
    producer_identity: viola_handoff.RuntimeIdentity | None = None,
    bundle_evidence_logger: viola_handoff.EvidenceLogger | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    attempt_id: str | None = None,
) -> ShadowEvidence:
    """Run replay or camera-only live soak, publish W&B, then seal evidence."""

    if mode not in {"replay", "live-soak"}:
        raise ValidationError("shadow mode must be 'replay' or 'live-soak'")
    if not wandb_entity:
        raise ValidationError("policy shadow requires --wandb-entity")
    candidate = inspect_candidate(bundle)
    identity = producer_identity or viola_handoff.RuntimeIdentity.capture(
        role="pc_a", repo_root=repo_root
    )
    _require_pc_a_identity(identity)
    _require_external_root(output_root, repo_root=repo_root, label="policy evidence")
    _require_external_root(handoff_root, repo_root=repo_root, label="handoff")
    _require_current_candidate(bundle, candidate, handoff_root=handoff_root)
    repo = _repo_identity(identity)
    verification_source, verification_sync_source = _verification_source_paths(
        verification_path
    )
    verification_inputs = _snapshot_input_files(
        {
            "runtime verification": verification_source,
            "runtime verification W&B receipt": verification_sync_source,
        }
    )
    verification, loaded_verification, loaded_verification_sync = _load_verification(
        candidate,
        verification_path,
        expected_repo=repo,
    )
    if (loaded_verification, loaded_verification_sync) != (
        verification_source,
        verification_sync_source,
    ):
        raise ValidationError("runtime verification paths changed while loading")
    _require_input_file_snapshots(verification_inputs)
    attempt = _attempt_id(attempt_id)
    material = _material_root(
        output_root,
        candidate=candidate,
        commit=identity.repository_commit,
        operation=mode,
        attempt=attempt,
    )
    payload_root = material / "payload"
    artifact_root = material / "shadow_record"
    payload_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    copied_verification = copy_regular_file(
        verification_source, payload_root / "verification.json"
    )
    copied_verification_sync = copy_regular_file(
        verification_sync_source,
        payload_root / "verification_WANDB_SYNCED.json",
    )
    if (
        sha256_file(copied_verification) != verification_inputs[0].sha256
        or sha256_file(copied_verification_sync) != verification_inputs[1].sha256
    ):
        raise ValidationError("copied runtime verification differs from reviewed bytes")
    _require_input_file_snapshots(verification_inputs)

    setup_hashes: dict[str, str] | None = None
    frozen_binding: dict[str, Any] | None = None
    videos: dict[str, dict[str, Any]] = {}
    frozen_files: tuple[Path, Path, Path] | None = None
    camera_owner: _OpenedCameras | None = None
    recorder: _VideoPairRecorder | None = None
    replay_source: _EvidenceRootSnapshot | None = None
    reviewed_roots: tuple[_EvidenceRootSnapshot, ...] = ()
    reviewed_inputs: tuple[_InputFileSnapshot, ...] = verification_inputs

    def validate_sources() -> None:
        root_snapshots = (
            (replay_source,) if replay_source is not None else reviewed_roots
        )
        _require_policy_sources(
            bundle,
            candidate,
            handoff_root=handoff_root,
            repo_root=repo_root,
            identity=identity,
            root_snapshots=root_snapshots,
            input_files=reviewed_inputs,
        )

    validate_sources()
    with _snapshot_candidate_runtime(candidate) as runtime_inputs:
        runtime_candidate = runtime_inputs.candidate
        runtime = runtime_factory(runtime_candidate)
        try:
            if mode == "replay":
                if setup_path is not None or frozen_state_path is not None or cameras is not None:
                    raise ValidationError(
                        "replay shadow does not accept setup, frozen state, or camera inputs"
                    )
                replay_source = _snapshot_replay_source(candidate)
                validate_sources()
                run = run_replay(
                    runtime_candidate,
                    runtime,
                    replay_loader(candidate),
                    clock=clock,
                )
                _require_evidence_snapshot((replay_source,))
            else:
                if setup_path is None or frozen_state_path is None:
                    raise ValidationError(
                        "live soak requires reviewed setup and frozen-state evidence"
                    )
                setup_source = _snapshot_input_files(
                    {"reviewed setup": Path(setup_path)}
                )
                frozen_source = _snapshot_evidence_roots(
                    {"frozen-state evidence": frozen_state_path}
                )
                reviewed_roots = frozen_source
                setup = load_reviewed_setup(setup_path, now=_utc(now()))
                _require_input_file_snapshots(setup_source)
                if setup.executor["commit"] != identity.repository_commit:
                    raise ValidationError("live setup was reviewed for another Repo-A commit")
                dependent_inputs = _snapshot_input_files(
                    {
                        "reviewed calibration": setup.calibration_path,
                        "reviewed reset protocol": setup.reset_protocol_path,
                        "reviewed executor": setup.executor_entrypoint,
                    }
                )
                # Reload after pinning dependent files.  This makes their
                # snapshotted bytes reproduce the hashes attested by the setup
                # JSON instead of adopting a change between load and snapshot.
                if load_reviewed_setup(setup_path, now=_utc(now())) != setup:
                    raise ValidationError("live setup changed while its files were pinned")
                _require_input_file_snapshots((*setup_source, *dependent_inputs))
                setup_hashes = _setup_hashes(setup)
                if (
                    dependent_inputs[0].sha256 != setup_hashes["calibration"]
                    or dependent_inputs[1].sha256 != setup_hashes["reset"]
                    or dependent_inputs[2].sha256
                    != setup.executor["entrypoint_sha256"]
                ):
                    raise ValidationError(
                        "reviewed setup files differ from their attested hashes"
                    )
                state, frozen_binding, frozen_files = _load_frozen_state(
                    frozen_state_path,
                    expected_setup_id=setup.setup_id,
                    expected_operator=setup.estop["operator"],
                    expected_setup_hashes=setup_hashes,
                    expected_repo=repo,
                )
                reviewed_inputs = (
                    *verification_inputs,
                    *setup_source,
                    *dependent_inputs,
                )
                _require_evidence_snapshot(frozen_source)
                for source in frozen_files:
                    copy_regular_file(source, payload_root / source.name)
                _require_evidence_snapshot(frozen_source)
                _require_input_file_snapshots(reviewed_inputs)
                validate_sources()

                active_cameras = cameras
                if active_cameras is None:
                    camera_owner = _OpenedCameras.open(setup.cameras, clock=clock)
                    active_cameras = camera_owner.sources
                recorder = _VideoPairRecorder(artifact_root / "videos")
                run = run_live_soak(
                    runtime_candidate,
                    runtime,
                    active_cameras,
                    frozen_state=state,
                    limits=_reviewed_limits(setup),
                    clock=clock,
                    sleep=sleep,
                    frame_sink=recorder.record,
                )
                recorder.close()
                _require_evidence_snapshot(frozen_source)
                _require_input_file_snapshots(reviewed_inputs)
                videos = {
                    name: _video_metadata(
                        artifact_root / "videos" / f"{name}.mp4",
                        name=name,
                        expected_frames=int(run.summary["captured_observations"]),
                        passed=run.status == "passed",
                    )
                    for name in ("front", "up")
                }
            runtime_inputs.verify()
        finally:
            if recorder is not None:
                recorder.close()
            if camera_owner is not None:
                camera_owner.close()

    trace_path = write_trace(artifact_root / "action_trace.jsonl", run)

    validate_sources()

    seed = sha256_json(
        {
            "operation": "policy_shadow",
            "mode": mode,
            "bundle_id": candidate.bundle_id,
            "content_id": candidate.content_id,
            "repo_commit": identity.repository_commit,
            "setup_hashes": setup_hashes,
            "attempt_id": attempt,
        }
    )
    wandb = planned_run(wandb_entity, wandb_project, f"shadow-{mode.replace('-', '_')}-{seed[:16]}")
    verification_sha256 = sha256_file(copied_verification)
    shadow_payload = build_shadow_payload(
        candidate,
        run,
        verification_payload=verification,
        verification_sha256=verification_sha256,
        repo=repo,
        wandb=wandb.binding(),
        setup_hashes=setup_hashes,
        frozen_state_capture=frozen_binding,
        videos=videos,
        completed_at=_utc(now()).isoformat(),
    )
    shadow_path = write_canonical_json(payload_root / "shadow_evidence.json", shadow_payload)
    shadow_sha256 = sha256_file(shadow_path)
    publication_snapshot = _snapshot_evidence_roots(
        {
            "shadow payload": payload_root,
            "shadow artifact": artifact_root,
        }
    )
    validate_sources()
    publisher(
        wandb,
        job_type="viola-policy-shadow",
        config=_shadow_wandb_config(
            candidate,
            shadow_payload,
            evidence_sha256=shadow_sha256,
        ),
        summary={
            "status": run.status,
            "mode": mode,
            "actions": run.summary["actions"],
            "replans": run.summary["replans"],
            "wall_seconds": run.summary["wall_seconds"],
            "deadline_misses": run.summary["deadline_misses"],
            "rate_misses": run.summary["rate_misses"],
            "stale_frames": run.summary["stale_frames"],
            "limit_violations": run.summary["limit_violations"],
            "malformed_actions": run.summary["malformed_actions"],
        },
    )
    shadow_sync = build_sync_receipt(
        operation="policy_shadow",
        evidence_file=shadow_path.name,
        evidence_sha256=shadow_sha256,
        wandb=wandb.binding(),
        binding=shadow_sync_binding(
            candidate,
            shadow_payload,
            verification_sha256=verification_sha256,
        ),
        synced_at=_utc(now()).isoformat(),
    )
    validate_sources()
    _require_evidence_snapshot(publication_snapshot)
    shadow_sync_path = write_canonical_json(
        payload_root / "shadow_WANDB_SYNCED.json", shadow_sync
    )
    final_snapshot = _expect_added_files(
        publication_snapshot,
        {shadow_sync_path: viola_handoff.canonical_json_bytes(shadow_sync)},
    )

    frozen_state_file = payload_root / "frozen_state.json"
    frozen_capture_file = payload_root / "frozen_state_capture.json"
    frozen_sync_file = payload_root / "frozen_state_capture_WANDB_SYNCED.json"
    lineage = build_shadow_lineage(
        candidate,
        shadow_payload,
        shadow_evidence_sha256=shadow_sha256,
        shadow_sync_sha256=sha256_file(shadow_sync_path),
        verification_sha256=verification_sha256,
        verification_sync_sha256=sha256_file(copied_verification_sync),
        frozen_state_sha256=(sha256_file(frozen_state_file) if frozen_files else None),
        frozen_state_capture_sha256=(
            sha256_file(frozen_capture_file) if frozen_files else None
        ),
        frozen_state_capture_sync_sha256=(
            sha256_file(frozen_sync_file) if frozen_files else None
        ),
    )
    request = viola_handoff.SealRequest(
        root=handoff_root,
        kind="shadow_evidence",
        experiment=candidate.bundle.manifest["experiment"],
        subject=f"{candidate.policy}:{mode}",
        producer=identity,
        lineage=lineage,
        wandb_project=wandb_project,
        payload_dir=payload_root,
        artifact_roots={"shadow_record": artifact_root},
    )
    sealed = _seal_guarded(
        request,
        evidence_logger=bundle_evidence_logger,
        wandb_entity=wandb_entity,
        snapshots=final_snapshot,
        artifact_names={"shadow artifact": "shadow_record"},
        validate_sources=validate_sources,
        repo_root=repo_root,
    )
    validate_sources()
    _require_evidence_snapshot(final_snapshot)
    return ShadowEvidence(candidate, run, material, payload_root, artifact_root, sealed)


def _load_verification(
    candidate: AcceptedPolicyCandidate,
    path: str | Path,
    *,
    expected_repo: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    source, sync_path = _verification_source_paths(path)
    value = _read_canonical_object(source, label="runtime verification")
    require_exact_keys(value, _VERIFICATION_FIELDS, label="runtime verification")
    expected = {
        "schema_version": 1,
        "kind": "pc_runtime_verification",
        "status": "eligible",
        "bundle_id": candidate.bundle_id,
        "content_id": candidate.content_id,
        "policy": candidate.policy,
        "checkpoint": "selected_checkpoint",
        "queue_actions": 10,
        "task": CANONICAL_TASK,
        "feature_map": dict(candidate.spec.feature_map),
        "runtime_binding": _json_copy(candidate.runtime_binding),
        "failure_reason": None,
        "failure": None,
        "repo": dict(expected_repo),
    }
    for field, required in expected.items():
        if value[field] != required:
            raise ValidationError(f"runtime verification differs from candidate/current {field}")
    finite_action(value["sample_action"], label="runtime verification sample action")
    latency = value["latency"]
    if (
        not isinstance(latency, Mapping)
        or latency.get("warmups") != 20
        or latency.get("trials") != 200
    ):
        raise ValidationError("runtime verification lacks the complete 20+200 latency benchmark")
    require_exact_keys(
        latency,
        {"warmups", "trials", "p50_ms", "p95_ms", "threshold_ms", "samples_ms"},
        label="runtime verification latency",
    )
    samples = latency.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != 200:
        raise ValidationError("runtime verification must retain all 200 latency samples")
    measured = [_nonnegative_number(item, "runtime latency sample") for item in samples]
    p50 = _percentile(measured, 0.50)
    p95 = _percentile(measured, 0.95)
    reported_p50 = _nonnegative_number(latency["p50_ms"], "runtime latency p50")
    reported_p95 = _nonnegative_number(latency["p95_ms"], "runtime latency p95")
    threshold = _nonnegative_number(latency["threshold_ms"], "runtime latency threshold")
    if (
        threshold != 300.0
        or not math.isclose(reported_p50, p50, rel_tol=0, abs_tol=1e-9)
        or not math.isclose(reported_p95, p95, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValidationError("runtime verification latency summary differs from its 200 samples")
    if p95 >= 300:
        raise ValidationError("runtime verification p95 must be below 300 ms")
    _wandb_identity(value["wandb"], "runtime verification")
    _utc_string(value["verified_at"], "runtime verification verified_at")

    sync = _read_canonical_object(sync_path, label="verification W&B sync receipt")
    require_exact_keys(sync, _SYNC_FIELDS, label="verification W&B sync receipt")
    if (
        sync["schema_version"] != 1
        or sync["operation"] != "policy_verify"
        or sync["evidence_file"] != source.name
        or sync["evidence_sha256"] != sha256_file(source)
        or sync["wandb"] != value["wandb"]
        or sync["binding"] != verification_sync_binding(candidate, value)
    ):
        raise ValidationError("verification W&B sync receipt is stale or mismatched")
    return value, source, sync_path


def _verification_source_paths(path: str | Path) -> tuple[Path, Path]:
    """Resolve the exact verification pair before either file is consumed."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source /= "verification.json"
    return source, source.with_name("verification_WANDB_SYNCED.json")


def _load_frozen_state(
    path: str | Path,
    *,
    expected_setup_id: str,
    expected_operator: str,
    expected_setup_hashes: Mapping[str, str],
    expected_repo: Mapping[str, Any],
) -> tuple[tuple[float, ...], dict[str, Any], tuple[Path, Path, Path]]:
    root = _existing_directory(path, label="frozen-state evidence")
    expected_names = {
        "frozen_state.json",
        "frozen_state_capture.json",
        "frozen_state_capture_WANDB_SYNCED.json",
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValidationError(
            "frozen-state evidence must contain exactly its state, capture, and W&B receipt"
        )
    state_path = root / "frozen_state.json"
    capture_path = root / "frozen_state_capture.json"
    sync_path = root / "frozen_state_capture_WANDB_SYNCED.json"
    for evidence_file in (state_path, capture_path, sync_path):
        mode = evidence_file.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValidationError(
                f"frozen-state evidence must be a regular nonsymlink file: {evidence_file}"
            )
    state = _read_canonical_object(state_path, label="frozen state")
    capture = _read_canonical_object(capture_path, label="frozen-state capture")
    sync = _read_canonical_object(sync_path, label="frozen-state W&B sync receipt")
    require_exact_keys(state, _FROZEN_STATE_FIELDS, label="frozen state")
    require_exact_keys(capture, _FROZEN_CAPTURE_FIELDS, label="frozen-state capture")
    require_exact_keys(sync, _SYNC_FIELDS, label="frozen-state W&B sync receipt")
    state_vector = finite_action(state.get("state"), label="frozen state")
    state_sha = state_vector_sha256(state_vector)
    captured_at = _utc_string(state.get("captured_at"), "frozen state captured_at")
    disconnected_at = _utc_string(
        state.get("motor_disconnected_at"), "frozen state motor_disconnected_at"
    )
    if (
        state.get("schema_version") != 1
        or state.get("robot_connected_at_capture") is not True
        or capture.get("schema_version") != 1
        or capture.get("kind") != "frozen_state_capture"
        or capture.get("status") != "captured_disconnected"
        or capture.get("frozen_state_file") != "frozen_state.json"
        or capture.get("frozen_state_sha256") != sha256_file(state_path)
        or capture.get("state_sha256") != state_sha
        or capture.get("setup_id") != expected_setup_id
        or capture.get("operator") != expected_operator
        or capture.get("setup_hashes") != dict(expected_setup_hashes)
        or capture.get("repo") != dict(expected_repo)
        or capture.get("captured_at") != state["captured_at"]
        or capture.get("motor_disconnected_at") != state["motor_disconnected_at"]
        or not isinstance(capture.get("wandb"), Mapping)
    ):
        raise ValidationError("frozen-state evidence differs from the reviewed live setup")
    _wandb_identity(capture["wandb"], "frozen-state capture")
    if disconnected_at <= captured_at:
        raise ValidationError("frozen-state disconnection must follow capture")
    synced_at = _utc_string(sync.get("synced_at"), "frozen-state W&B synced_at")
    if synced_at < disconnected_at:
        raise ValidationError("frozen-state W&B sync predates serial disconnection")
    expected_sync_binding = {
        "setup_id": capture["setup_id"],
        "state_sha256": state_sha,
        "status": "captured_disconnected",
        "repo_commit": expected_repo["commit"],
        "setup_hashes": dict(expected_setup_hashes),
        "frozen_state_sha256": capture["frozen_state_sha256"],
    }
    if (
        sync.get("schema_version") != 1
        or sync.get("operation") != "frozen_state_capture"
        or sync.get("evidence_file") != capture_path.name
        or sync.get("evidence_sha256") != sha256_file(capture_path)
        or sync.get("wandb") != capture.get("wandb")
        or sync.get("binding") != expected_sync_binding
    ):
        raise ValidationError("frozen-state W&B sync receipt is stale or mismatched")
    binding = {
        "setup_id": capture["setup_id"],
        "evidence_sha256": sha256_file(capture_path),
        "frozen_state_sha256": sha256_file(state_path),
        "state_sha256": state_sha,
        "wandb": capture["wandb"],
    }
    return state_vector, binding, (state_path, capture_path, sync_path)


def _existing_directory(path: str | Path, *, label: str) -> Path:
    """Return one existing directory without following symlink components."""

    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValidationError(f"{label} does not exist: {candidate}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink path is forbidden for {label}: {current}")
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ValidationError(f"{label} is not a directory: {candidate}")
    return candidate


def _verification_wandb_config(
    candidate: AcceptedPolicyCandidate,
    payload: Mapping[str, Any],
    *,
    evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "operation": "policy_verify",
        "evidence_sha256": evidence_sha256,
        "policy_candidate_bundle_id": candidate.bundle_id,
        "policy_candidate_content_id": candidate.content_id,
        "policy": candidate.policy,
        "mode": "runtime-verification",
        "status": payload["status"],
        "repo_commit": payload["repo"]["commit"],
        "runtime_binding": _json_copy(candidate.runtime_binding),
        "trace_file_sha256": None,
        "action_trace_sha256": None,
        "frozen_state_sha256": None,
        "setup_hashes": None,
    }


def _shadow_wandb_config(
    candidate: AcceptedPolicyCandidate,
    payload: Mapping[str, Any],
    *,
    evidence_sha256: str,
) -> dict[str, Any]:
    frozen_capture = payload["frozen_state_capture"]
    return {
        "operation": "policy_shadow",
        "evidence_sha256": evidence_sha256,
        "policy_candidate_bundle_id": candidate.bundle_id,
        "policy_candidate_content_id": candidate.content_id,
        "policy": candidate.policy,
        "mode": payload["summary"]["mode"],
        "status": payload["status"],
        "repo_commit": payload["repo"]["commit"],
        "runtime_binding": _json_copy(candidate.runtime_binding),
        "verification_sha256": payload["verification"]["sha256"],
        "trace_file_sha256": payload["trace_file_sha256"],
        "action_trace_sha256": payload["summary"]["action_trace_sha256"],
        "frozen_state_sha256": payload["summary"]["frozen_state_sha256"],
        "frozen_state_file_sha256": (
            None if frozen_capture is None else frozen_capture["frozen_state_sha256"]
        ),
        "setup_hashes": payload["setup_hashes"],
    }


def _verification_summary(result: RuntimeVerification) -> dict[str, Any]:
    latency = result.latency or {}
    return {
        "status": result.status,
        "action_dimensions": None if result.sample_action is None else len(result.sample_action),
        "latency_p50_ms": latency.get("p50_ms"),
        "latency_p95_ms": latency.get("p95_ms"),
        "latency_trials": latency.get("trials", 0),
        "failure_code": None if result.failure is None else result.failure["code"],
    }


def _setup_hashes(setup: Any) -> dict[str, str]:
    robot = {
        "robot_port": setup.robot_port,
        "joint_limits": setup.joint_limits,
        "max_step_deltas": setup.max_step_deltas,
        "speed_scale": setup.speed_scale,
    }
    return {
        "calibration": sha256_file(setup.calibration_path),
        "camera": sha256_json(setup.cameras),
        "robot": sha256_json(robot),
        "reset": sha256_file(setup.reset_protocol_path),
    }


def _reviewed_limits(setup: Any) -> ReviewedLimits:
    return ReviewedLimits(
        lower=tuple(setup.joint_limits[name][0] for name in JOINT_NAMES),
        upper=tuple(setup.joint_limits[name][1] for name in JOINT_NAMES),
        max_step_deltas=tuple(setup.max_step_deltas[name] for name in JOINT_NAMES),
        speed_scale=setup.speed_scale,
    )


class _OpenCVCameraSource:
    def __init__(self, camera: Any, clock: Callable[[], float]) -> None:
        self.camera = camera
        self.clock = clock

    def read(self, *, deadline_ms: float) -> CameraFrame:
        image = self.camera.async_read(timeout_ms=deadline_ms)
        return CameraFrame(image=image, arrived_at=self.clock())


@dataclass(slots=True)
class _OpenedCameras:
    sources: Mapping[str, CameraSource]
    cameras: tuple[Any, ...]

    @classmethod
    def open(
        cls,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        clock: Callable[[], float],
    ) -> "_OpenedCameras":
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cameras: list[Any] = []
        sources: dict[str, CameraSource] = {}
        try:
            for name in ("front", "up"):
                item = configs[name]
                config = OpenCVCameraConfig(
                    index_or_path=Path(item["index_or_path"]),
                    fps=item["fps"],
                    width=item["width"],
                    height=item["height"],
                )
                camera = OpenCVCamera(config)
                cameras.append(camera)
                camera.connect(warmup=True)
                sources[name] = _OpenCVCameraSource(camera, clock)
        except BaseException:
            for camera in reversed(cameras):
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except BaseException:
                    pass
            raise
        return cls(sources, tuple(cameras))

    def close(self) -> None:
        errors: list[BaseException] = []
        for camera in reversed(self.cameras):
            try:
                camera.disconnect()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise ValidationError(f"camera disconnect failed: {errors[0]}") from errors[0]


class _VideoPairRecorder:
    """Bounded background encoder; queue saturation makes the soak fail."""

    def __init__(self, root: Path, *, queue_size: int = 64) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.paths = {name: root / f"{name}.mp4" for name in ("front", "up")}
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._failure: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def record(self, front: Any, up: Any) -> None:
        if self._closed or self._failure is not None:
            raise ValidationError(f"shadow video recorder failed: {self._failure}")
        import numpy as np

        pair = (np.ascontiguousarray(front).copy(), np.ascontiguousarray(up).copy())
        for name, image in zip(("front", "up"), pair, strict=True):
            if image.shape != (480, 640, 3):
                raise ValidationError(f"{name} frame must be 480x640x3")
        try:
            self._queue.put_nowait(pair)
        except queue.Full as exc:
            raise ValidationError(
                "shadow video queue saturated; evidence would be incomplete"
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=5.0)
        except queue.Full as exc:
            raise ValidationError("shadow video queue did not drain") from exc
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            raise ValidationError("shadow video encoder did not stop within 60 seconds")
        if self._failure is not None:
            raise ValidationError(
                f"shadow video encoder failed: {self._failure}"
            ) from self._failure

    def _worker(self) -> None:
        containers: dict[str, Any] = {}
        streams: dict[str, Any] = {}
        try:
            import av

            for name, path in self.paths.items():
                container = av.open(str(path), mode="w")
                stream = container.add_stream("libx264", rate=30)
                stream.width = 640
                stream.height = 480
                stream.pix_fmt = "yuv420p"
                containers[name] = container
                streams[name] = stream
            while True:
                pair = self._queue.get()
                if pair is None:
                    break
                for name, image in zip(("front", "up"), pair, strict=True):
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in streams[name].encode(frame):
                        containers[name].mux(packet)
            for name in ("front", "up"):
                for packet in streams[name].encode():
                    containers[name].mux(packet)
        except BaseException as exc:
            self._failure = exc
        finally:
            for container in containers.values():
                try:
                    container.close()
                except BaseException as exc:
                    if self._failure is None:
                        self._failure = exc


def _video_metadata(
    path: Path,
    *,
    name: str,
    expected_frames: int,
    passed: bool,
) -> dict[str, Any]:
    """Fully decode one live-soak video, including a typed early terminal."""

    import av

    frames = 0
    width = height = 0
    fps = 0.0
    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            for frame in container.decode(stream):
                frames += 1
                width, height = frame.width, frame.height
    except Exception as exc:
        raise ValidationError(f"could not fully decode {name} shadow video: {exc}") from exc
    if (
        frames != expected_frames
        or (passed and frames < 9_000)
        or fps != 30.0
        or (width, height) != (640, 480)
    ):
        raise ValidationError(
            f"{name} video must contain exactly {expected_frames} captured "
            "640x480 frames at 30 fps"
        )
    return {
        "path": f"videos/{name}.mp4",
        "sha256": sha256_file(path),
        "frame_count": frames,
        "fps": 30.0,
        "duration_s": frames / 30.0,
        "width": width,
        "height": height,
    }


def _material_root(
    output_root: str | Path,
    *,
    candidate: AcceptedPolicyCandidate,
    commit: str,
    operation: str,
    attempt: str,
) -> Path:
    parent = (
        _safe_output_directory(output_root)
        / candidate.policy
        / candidate.bundle_id
        / commit
        / operation
    )
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / attempt
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise ValidationError(
            f"policy attempt already exists and will not be reused: {root}"
        ) from exc
    return root


def _attempt_id(value: str | None) -> str:
    """Return a fresh path-safe ID; injection is reserved for deterministic tests."""

    attempt = uuid.uuid4().hex if value is None else value
    if not isinstance(attempt, str) or not _ATTEMPT_ID_RE.fullmatch(attempt):
        raise ValidationError(
            "policy attempt_id must be 1-64 path-safe letters, digits, '.', '_', or '-'"
        )
    return attempt


def _safe_output_directory(path: str | Path) -> Path:
    """Create an output root without following an existing symlink component."""

    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink output path is forbidden: {current}")
        if not stat.S_ISDIR(mode):
            raise ValidationError(f"output path component is not a directory: {current}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _require_external_root(
    path: str | Path,
    *,
    repo_root: str | Path,
    label: str,
) -> None:
    """Keep generated evidence out of the reviewed Git worktree."""

    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    repository = Path(os.path.abspath(os.path.expanduser(os.fspath(repo_root))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValidationError(f"cannot inspect {label} root {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink path is forbidden for {label} root: {current}")
        if not stat.S_ISDIR(mode):
            raise ValidationError(f"{label} root component is not a directory: {current}")
    try:
        candidate.relative_to(repository)
    except ValueError:
        return
    raise ValidationError(f"{label} root must be outside the Repo-A worktree: {candidate}")


def _require_current_identity(
    repo_root: str | Path,
    expected: viola_handoff.RuntimeIdentity,
) -> None:
    """Fail if Repo A or its required runtime changed during evidence production."""

    current = viola_handoff.RuntimeIdentity.capture(role="pc_a", repo_root=repo_root)
    _require_pc_a_identity(current)
    if current != expected:
        raise ValidationError(
            "Repo-A revision or runtime identity changed during policy evidence production"
        )


def _require_current_candidate(
    bundle: str | Path,
    expected: AcceptedPolicyCandidate,
    *,
    handoff_root: str | Path,
) -> None:
    """Reinspect local bytes and prove the canonical candidate is still active."""

    current = inspect_candidate(bundle)
    if (
        current.bundle_id != expected.bundle_id
        or current.content_id != expected.content_id
        or dict(current.runtime_binding) != dict(expected.runtime_binding)
    ):
        raise ValidationError("accepted policy candidate changed during evidence production")
    try:
        viola_handoff.require_active_canonical_source(
            current.bundle,
            handoff_root=handoff_root,
            expected_kind="policy_candidate",
            required_permission="disconnected_only",
        )
    except viola_handoff.HandoffError as exc:
        raise ValidationError(
            f"canonical policy candidate is unavailable, changed, or revoked: {exc}"
        ) from exc


def _snapshot_evidence_roots(
    roots: Mapping[str, str | Path],
) -> tuple[_EvidenceRootSnapshot, ...]:
    """Capture exact directory identities and inventories without following links."""

    return tuple(snapshot_tree(path, label=label) for label, path in roots.items())


def _require_evidence_snapshot(
    expected: tuple[_EvidenceRootSnapshot, ...],
) -> None:
    """Fail if a publication callback changed or replaced evidence material."""

    current = _snapshot_evidence_roots(
        {snapshot.label: snapshot.path for snapshot in expected}
    )
    if current != expected:
        raise ValidationError("policy evidence changed during online publication or sealing")


def _snapshot_replay_source(
    candidate: AcceptedPolicyCandidate,
) -> _EvidenceRootSnapshot:
    """Pin the accepted replay artifact without copying its large video payload."""

    snapshots = _snapshot_evidence_roots(
        {"accepted replay dataset": candidate.replay_dataset}
    )
    snapshot = snapshots[0]
    artifact = next(
        (
            item
            for item in candidate.bundle.manifest["artifacts"]
            if item["name"] == REPLAY_DATASET_ARTIFACT
        ),
        None,
    )
    if artifact is None:
        raise ValidationError("accepted policy candidate lacks its replay dataset artifact")
    expected = {
        key: artifact[key]
        for key in ("directories", "files", "file_count", "byte_count", "inventory_sha256")
    }
    if dict(snapshot.inventory) != expected:
        raise ValidationError("accepted replay dataset differs from signed candidate bytes")
    return snapshot


def _snapshot_input_files(
    files: Mapping[str, str | Path],
) -> tuple[_InputFileSnapshot, ...]:
    """Hash reviewed files while retaining their exact filesystem identity."""

    return tuple(snapshot_file(path, label=label) for label, path in files.items())


def _require_input_file_snapshots(expected: tuple[_InputFileSnapshot, ...]) -> None:
    current = _snapshot_input_files(
        {snapshot.label: snapshot.path for snapshot in expected}
    )
    if current != expected:
        raise ValidationError("reviewed policy input changed during shadow production")


def _expect_added_files(
    before: tuple[_EvidenceRootSnapshot, ...],
    added: Mapping[Path, bytes],
) -> tuple[_EvidenceRootSnapshot, ...]:
    """Permit only named immutable files to join an already-pinned evidence tree."""

    after = _snapshot_evidence_roots(
        {snapshot.label: snapshot.path for snapshot in before}
    )
    additions_by_root: dict[Path, dict[str, dict[str, Any]]] = {
        snapshot.path: {} for snapshot in before
    }
    for path, expected_bytes in added.items():
        resolved = Path(os.path.abspath(os.fspath(path)))
        owner = next(
            (snapshot.path for snapshot in before if resolved.is_relative_to(snapshot.path)),
            None,
        )
        if owner is None:
            raise ValidationError(f"added policy evidence is outside its pinned roots: {resolved}")
        relative = resolved.relative_to(owner).as_posix()
        additions_by_root[owner][relative] = {
            "path": relative,
            "sha256": hashlib.sha256(expected_bytes).hexdigest(),
            "size_bytes": len(expected_bytes),
        }

    for old, new in zip(before, after, strict=True):
        if (old.label, old.path, old.device, old.inode) != (
            new.label,
            new.path,
            new.device,
            new.inode,
        ):
            raise ValidationError("policy evidence root changed before handoff sealing")
        if old.inventory["directories"] != new.inventory["directories"]:
            raise ValidationError("policy evidence layout changed before handoff sealing")
        old_files = {item["path"]: item for item in old.inventory["files"]}
        new_files = {item["path"]: item for item in new.inventory["files"]}
        expected_paths = set(old_files) | set(additions_by_root[old.path])
        if set(new_files) != expected_paths:
            raise ValidationError("unexpected policy evidence changed before handoff sealing")
        for path, entry in old_files.items():
            if new_files[path] != entry:
                raise ValidationError("policy evidence bytes changed before handoff sealing")
        for path, entry in additions_by_root[old.path].items():
            if new_files[path] != entry:
                raise ValidationError("added policy evidence differs from its exact bytes")
    return after


def _require_policy_sources(
    bundle: str | Path,
    candidate: AcceptedPolicyCandidate,
    *,
    handoff_root: str | Path,
    repo_root: str | Path,
    identity: viola_handoff.RuntimeIdentity,
    root_snapshots: tuple[_EvidenceRootSnapshot, ...] = (),
    input_files: tuple[_InputFileSnapshot, ...] = (),
) -> None:
    """Recheck every mutable source that remains authoritative for this result."""

    _require_current_candidate(bundle, candidate, handoff_root=handoff_root)
    _require_current_identity(repo_root, identity)
    _require_evidence_snapshot(root_snapshots)
    _require_input_file_snapshots(input_files)


def _seal_guarded(
    request: viola_handoff.SealRequest,
    *,
    evidence_logger: viola_handoff.EvidenceLogger | None,
    wandb_entity: str,
    snapshots: tuple[_EvidenceRootSnapshot, ...],
    artifact_names: Mapping[str, str],
    validate_sources: Callable[[], None],
    repo_root: str | Path,
) -> viola_handoff.VerifiedBundle:
    """Seal only the exact evidence retained through the handoff W&B call."""

    payload = next(
        snapshot for snapshot in snapshots if snapshot.label not in artifact_names
    )
    publication = PublicationSnapshot(
        payload=payload,
        artifacts=tuple(
            (artifact_names[snapshot.label], snapshot)
            for snapshot in snapshots
            if snapshot.label in artifact_names
        ),
    )
    logger = GuardedEvidenceLogger(
        delegate=evidence_logger or viola_handoff.WandbEvidenceLogger(entity=wandb_entity),
        snapshot=publication,
        identity=request.producer,
        identity_capture=lambda: viola_handoff.RuntimeIdentity.capture(
            role="pc_a", repo_root=repo_root
        ),
        operation="policy evidence",
        source_validator=validate_sources,
    )
    validate_sources()
    _require_evidence_snapshot(snapshots)
    bundle = viola_handoff.seal_bundle(request, evidence_logger=logger)
    require_sealed_bundle_matches(
        bundle,
        request,
        publication,
        operation="policy evidence",
        permission="evidence_only",
        consumer_role="pc_b",
    )
    validate_sources()
    _require_evidence_snapshot(snapshots)
    return bundle


def _repo_identity(identity: viola_handoff.RuntimeIdentity) -> dict[str, Any]:
    return {
        "commit": identity.repository_commit,
        "clean": identity.repository_clean,
        "hostname": identity.hostname,
        "python": identity.python_version,
    }


def _require_pc_a_identity(identity: viola_handoff.RuntimeIdentity) -> None:
    if identity.role != "pc_a" or identity.repository_clean is not True:
        raise ValidationError("policy evidence requires a clean pc_a runtime identity")
    if not isinstance(identity.repository_commit, str) or not _GIT_SHA_RE.fullmatch(
        identity.repository_commit
    ):
        raise ValidationError("policy evidence requires a full lowercase Git commit SHA")
    if (
        not isinstance(identity.hostname, str)
        or not identity.hostname
        or not isinstance(identity.python_version, str)
        or identity.python_version.split(".")[:2] != ["3", "12"]
        or identity.lerobot_version != "0.6.1"
        or identity.conda_environment != "lerobot"
    ):
        raise ValidationError(
            "policy evidence requires a complete Python 3.12 / LeRobot 0.6.1 "
            "lerobot-environment identity"
        )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("evidence clock must be timezone-aware")
    return value.astimezone(UTC)


def _utc_string(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a UTC offset")
    return parsed.astimezone(UTC)


def _wandb_identity(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"run_id", "url"}:
        raise ValidationError(f"{label} W&B identity must contain only run_id and url")
    run_id = value["run_id"]
    url = value["url"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValidationError(f"{label} W&B run_id must be a nonempty string")
    if not isinstance(url, str) or not url.strip():
        raise ValidationError(f"{label} W&B url must be a nonempty string")
    parsed = urlsplit(url)
    parts = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc != "wandb.ai"
        or parsed.query
        or parsed.fragment
        or len(parts) != 5
        or parts[0] != ""
        or not parts[1]
        or not parts[2]
        or parts[3] != "runs"
        or parts[4] != run_id
    ):
        raise ValidationError(f"{label} W&B url must canonically identify run_id {run_id!r}")
    return {"run_id": run_id, "url": url}


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(viola_handoff.canonical_json_bytes(dict(value)))


def _read_canonical_object(path: Path, *, label: str) -> dict[str, Any]:
    value = read_json_object(path, label=label)
    if path.read_bytes() != viola_handoff.canonical_json_bytes(value):
        raise ValidationError(f"{label} must use canonical JSON")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValidationError(f"{label} must be finite and nonnegative")
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "ShadowEvidence",
    "VerificationEvidence",
    "shadow_command",
    "verify_command",
]
