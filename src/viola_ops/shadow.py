"""Disconnected replay and camera-only live-soak orchestration.

The two loops share a policy runtime but deliberately do not share a generic
framework: their timing, reset, limit, and evidence rules are different enough
that two short, linear functions are easier to audit.  No robot API is imported
or accepted by this module.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from viola_handoff import canonical_json_bytes

from .errors import ValidationError
from .policies import CANONICAL_TASK, JOINT_NAMES
from .policy_runtime import (
    SHADOW_SCHEMA_SHA256,
    AcceptedPolicyCandidate,
    PolicyRuntime,
    finite_action,
    prepare_observation,
)

TARGET_HZ = 30.0
TARGET_PERIOD_MS = 1_000.0 / TARGET_HZ
RATE_TOLERANCE_MS = 5.0
DEADLINE_MS = 300.0
MAXIMUM_FRAME_AGE_MS = 100.0
MINIMUM_ACTIONS = 9_000
MINIMUM_SECONDS = 300.0
CANONICAL_LOWER = (-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, 0.0)
CANONICAL_UPPER = (100.0,) * 7
REQUIREMENTS = {
    "minimum_actions": MINIMUM_ACTIONS,
    "minimum_seconds": MINIMUM_SECONDS,
    "target_hz": TARGET_HZ,
    "rate_tolerance_ms": RATE_TOLERANCE_MS,
    "deadline_ms": DEADLINE_MS,
    "maximum_frame_age_ms": MAXIMUM_FRAME_AGE_MS,
    "action_dimensions": 7,
    "recorded_cameras": ["front", "up"],
    "replay_limit_scope": "canonical_device_domain_only",
    "live_limit_scope": "reviewed_setup_absolute_and_per_step",
    "live_rate_reference": "frozen_then_previous_proposed_action",
}


class CameraSource(Protocol):
    """Injected camera source; production adapters may wrap ``async_read``."""

    def read(self, *, deadline_ms: float) -> "CameraFrame":
        """Return an image and the monotonic time at which it arrived."""


@dataclass(frozen=True, slots=True)
class CameraFrame:
    image: Any
    arrived_at: float


@dataclass(frozen=True, slots=True)
class ReviewedLimits:
    """The exact reviewed seven-axis limits used for a live soak."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_step_deltas: tuple[float, ...]
    speed_scale: float = 1.0

    def __post_init__(self) -> None:
        lower = finite_action(self.lower, label="reviewed lower limits")
        upper = finite_action(self.upper, label="reviewed upper limits")
        deltas = finite_action(self.max_step_deltas, label="reviewed max step deltas")
        if self.speed_scale != 1.0:
            raise ValidationError("live shadow requires reviewed speed_scale=1.0")
        for index, (minimum, maximum, canonical_min, canonical_max, delta) in enumerate(
            zip(lower, upper, CANONICAL_LOWER, CANONICAL_UPPER, deltas, strict=True)
        ):
            if not canonical_min <= minimum < maximum <= canonical_max:
                raise ValidationError(f"reviewed joint {index} bounds exceed the device domain")
            if delta <= 0 or delta > maximum - minimum:
                raise ValidationError(f"reviewed joint {index} max step delta is invalid")


@dataclass(frozen=True, slots=True)
class ShadowRun:
    """A complete passed run or a typed, positively evidenced unsafe terminal."""

    mode: str
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    failure: Mapping[str, Any] | None

    @property
    def status(self) -> str:
        return "passed" if self.summary["completed"] else "unsafe_shadow"

    def trace_bytes(self) -> bytes:
        return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in self.rows)

    @property
    def trace_file_sha256(self) -> str:
        return hashlib.sha256(self.trace_bytes()).hexdigest()


def run_replay(
    candidate: AcceptedPolicyCandidate,
    runtime: PolicyRuntime,
    frames: Sequence[Mapping[str, Any]],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> ShadowRun:
    """Replay the complete signed held-out cycle until 9,000 proposed actions."""

    if len(frames) != 6_079:
        raise ValidationError(f"replay requires all 6079 held-out frames, found {len(frames)}")

    rows: list[dict[str, Any]] = []
    counters = _empty_counters()
    semantic = hashlib.sha256()
    runtime.reset()
    runtime_resets = 1
    started = clock()
    repeat = 0
    while counters["actions"] < MINIMUM_ACTIONS:
        previous_episode: int | None = None
        local_frame = -1
        seen_episodes: list[int] = []
        for raw in frames:
            if counters["actions"] >= MINIMUM_ACTIONS:
                break
            episode = _scalar_integer(raw.get("episode_index"), "replay episode_index")
            if episode != previous_episode:
                if previous_episode is not None:
                    seen_episodes.append(previous_episode)
                previous_episode = episode
                local_frame = 0
                runtime.reset()
                runtime_resets += 1
            else:
                local_frame += 1

            observation = _canonical_observation(raw)
            prepared = prepare_observation(candidate.spec, observation)
            inference_started = clock()
            try:
                action = finite_action(runtime.infer(prepared))
            except Exception as exc:
                row = {
                    "repeat": repeat,
                    "episode": episode,
                    "frame": local_frame,
                    "event": "malformed_action",
                    "detail": str(exc),
                    "limit_scope": "canonical_device_domain_only",
                }
                rows.append(row)
                semantic.update(canonical_json_bytes(row))
                counters["malformed_actions"] += 1
                return _finish_replay(rows, counters, semantic, runtime_resets, started, clock)
            latency_ms = (clock() - inference_started) * 1_000.0
            _nonnegative_finite(latency_ms, "replay inference latency")
            limit_check = canonical_limit_check(action)
            row = {
                "repeat": repeat,
                "episode": episode,
                "frame": local_frame,
                "action": list(action),
                "state_sha256": value_digest(observation["observation.state"]),
                "front_sha256": value_digest(observation["observation.images.front"]),
                "up_sha256": value_digest(observation["observation.images.up"]),
                "latency_ms": latency_ms,
                "limit_check": limit_check,
                "replan": local_frame % 10 == 0,
            }
            rows.append(row)
            semantic.update(
                canonical_json_bytes({key: value for key, value in row.items() if key != "latency_ms"})
            )
            counters["actions"] += 1
            counters["captured_observations"] += 1
            counters["replans"] += int(row["replan"])
            counters["deadline_misses"] += int(latency_ms >= DEADLINE_MS)
            counters["limit_violations"] += int(not limit_check["passed"])
            if latency_ms >= DEADLINE_MS or not limit_check["passed"]:
                return _finish_replay(rows, counters, semantic, runtime_resets, started, clock)

        if previous_episode is not None:
            seen_episodes.append(previous_episode)
        if counters["actions"] >= MINIMUM_ACTIONS:
            return _finish_replay(rows, counters, semantic, runtime_resets, started, clock)
        if tuple(seen_episodes) != tuple(range(27, 34)):
            raise ValidationError(
                "replay frames must contain ordered held-out episodes 27 through 33"
            )
        repeat += 1

    return _finish_replay(rows, counters, semantic, runtime_resets, started, clock)


def run_live_soak(
    candidate: AcceptedPolicyCandidate,
    runtime: PolicyRuntime,
    cameras: Mapping[str, CameraSource],
    *,
    frozen_state: Sequence[float],
    limits: ReviewedLimits,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    frame_sink: Callable[[Any, Any], None] | None = None,
) -> ShadowRun:
    """Run both cameras and policy for 300 seconds with no robot connection."""

    if set(cameras) != {"front", "up"}:
        raise ValidationError("live soak requires injected front and up camera sources")
    state = finite_action(frozen_state, label="frozen state")
    rows: list[dict[str, Any]] = []
    counters = _empty_counters()
    semantic = hashlib.sha256()
    previous_arrival: dict[str, float | None] = {"front": None, "up": None}
    previous_action = state
    runtime.reset()
    started = clock()
    next_tick = started + 1.0 / TARGET_HZ

    while counters["actions"] < MINIMUM_ACTIONS:
        delay = next_tick - clock()
        if delay > 0:
            sleep(delay)
        iteration_started = clock()
        elapsed_s = iteration_started - started
        index = counters["actions"]

        try:
            samples = {
                name: cameras[name].read(deadline_ms=MAXIMUM_FRAME_AGE_MS)
                for name in ("front", "up")
            }
        except Exception as exc:
            now = clock()
            row = {
                "index": index,
                "elapsed_s": max(now - started, 0.0),
                "event": "stale_observation",
                "source_blocker": "camera_read",
                "detail": f"{type(exc).__name__}: {exc}",
                "total_loop_latency_ms": max((now - iteration_started) * 1_000.0, 0.0),
            }
            rows.append(row)
            semantic.update(canonical_json_bytes(row))
            counters["stale_frames"] += 1
            break

        observed_at = clock()
        observation_read_ms = (observed_at - iteration_started) * 1_000.0
        frame_ages = {
            name: max((observed_at - sample.arrived_at) * 1_000.0, 0.0)
            for name, sample in samples.items()
        }
        interarrival = {
            name: (
                None
                if previous_arrival[name] is None
                else max((sample.arrived_at - previous_arrival[name]) * 1_000.0, 0.0)
            )
            for name, sample in samples.items()
        }
        freshness = camera_freshness(frame_ages, interarrival)
        stale = [name for name in ("front", "up") if frame_ages[name] > MAXIMUM_FRAME_AGE_MS]
        if stale:
            now = clock()
            row = {
                "index": index,
                "elapsed_s": max(now - started, 0.0),
                "event": "stale_observation",
                "frame_ages_ms": frame_ages,
                "stale_cameras": stale,
                "observation_read_ms": observation_read_ms,
                "total_loop_latency_ms": max((now - iteration_started) * 1_000.0, 0.0),
                "camera_freshness": freshness,
            }
            rows.append(row)
            semantic.update(canonical_json_bytes(row))
            counters["stale_frames"] += 1
            counters["captured_observations"] += 1
            break
        previous_arrival = {name: sample.arrived_at for name, sample in samples.items()}
        if frame_sink is not None:
            frame_sink(samples["front"].image, samples["up"].image)

        observation = {
            "observation.state": state,
            "observation.images.front": samples["front"].image,
            "observation.images.up": samples["up"].image,
            "task": CANONICAL_TASK,
        }
        prepared = prepare_observation(candidate.spec, observation)
        inference_started = clock()
        try:
            action = finite_action(runtime.infer(prepared))
        except Exception as exc:
            now = clock()
            row = {
                "index": index,
                "elapsed_s": max(now - started, 0.0),
                "event": "malformed_action",
                "detail": str(exc),
                "observation_read_ms": observation_read_ms,
                "total_loop_latency_ms": max((now - iteration_started) * 1_000.0, 0.0),
                "frame_ages_ms": frame_ages,
                "camera_freshness": freshness,
            }
            rows.append(row)
            semantic.update(canonical_json_bytes(row))
            counters["malformed_actions"] += 1
            counters["captured_observations"] += 1
            break
        inference_finished = clock()
        latency_ms = (inference_finished - inference_started) * 1_000.0
        total_loop_latency_ms = (inference_finished - iteration_started) * 1_000.0
        _nonnegative_finite(latency_ms, "live inference latency")
        _nonnegative_finite(total_loop_latency_ms, "live total loop latency")
        elapsed_s = inference_finished - started
        actual_period_ms = (
            elapsed_s * 1_000.0
            if not rows
            else (elapsed_s - float(rows[-1]["elapsed_s"])) * 1_000.0
        )
        deadline_passed = total_loop_latency_ms < DEADLINE_MS
        rate_passed = actual_period_ms <= TARGET_PERIOD_MS + RATE_TOLERANCE_MS
        reference = "frozen_state" if index == 0 else "previous_proposed_action"
        limit_check = deployment_limit_check(
            action,
            reference_action=previous_action,
            reference=reference,
            limits=limits,
        )
        row = {
            "index": index,
            "elapsed_s": elapsed_s,
            "action": list(action),
            "state_sha256": value_digest(state),
            "front_sha256": value_digest(samples["front"].image),
            "up_sha256": value_digest(samples["up"].image),
            "latency_ms": latency_ms,
            "observation_read_ms": observation_read_ms,
            "total_loop_latency_ms": total_loop_latency_ms,
            "deadline_ms": DEADLINE_MS,
            "deadline_passed": deadline_passed,
            "target_period_ms": TARGET_PERIOD_MS,
            "rate_passed": rate_passed,
            "frame_ages_ms": frame_ages,
            "camera_freshness": freshness,
            "limit_check": limit_check,
            "replan": index % 10 == 0,
        }
        rows.append(row)
        semantic.update(canonical_json_bytes(row))
        counters["actions"] += 1
        counters["captured_observations"] += 1
        counters["replans"] += int(row["replan"])
        counters["deadline_misses"] += int(not deadline_passed)
        counters["rate_misses"] += int(not rate_passed)
        counters["limit_violations"] += int(not limit_check["passed"])
        if limit_check["passed"]:
            previous_action = action
        if not deadline_passed or not rate_passed or not limit_check["passed"]:
            break
        next_tick += 1.0 / TARGET_HZ

    wall_seconds = float(rows[-1]["elapsed_s"]) if rows else 0.0
    return _finish_shadow(
        "live-soak",
        rows,
        counters,
        semantic,
        runtime_resets=1,
        wall_seconds=wall_seconds,
        frozen_state_sha256=state_vector_sha256(state),
    )


def canonical_limit_check(action: Sequence[float]) -> dict[str, Any]:
    proposed = finite_action(action)
    violations = [
        joint
        for joint, value, lower, upper in zip(
            JOINT_NAMES, proposed, CANONICAL_LOWER, CANONICAL_UPPER, strict=True
        )
        if value < lower or value > upper
    ]
    return {
        "scope": "canonical_device_domain_only",
        "joint_order": list(JOINT_NAMES),
        "lower": list(CANONICAL_LOWER),
        "upper": list(CANONICAL_UPPER),
        "absolute_violations": violations,
        "rate_limits_reviewed": False,
        "passed": not violations,
    }


def deployment_limit_check(
    action: Sequence[float],
    *,
    reference_action: Sequence[float],
    reference: str,
    limits: ReviewedLimits,
) -> dict[str, Any]:
    proposed = finite_action(action)
    previous = finite_action(reference_action, label="limit reference action")
    allowed = tuple(delta * limits.speed_scale for delta in limits.max_step_deltas)
    absolute = [
        joint
        for joint, value, lower, upper in zip(
            JOINT_NAMES, proposed, limits.lower, limits.upper, strict=True
        )
        if value < lower or value > upper
    ]
    rate = [
        joint
        for joint, value, old, delta in zip(
            JOINT_NAMES, proposed, previous, allowed, strict=True
        )
        if abs(value - old) > delta + 1e-9
    ]
    clamped = [joint for joint in JOINT_NAMES if joint in set(absolute) | set(rate)]
    return {
        "scope": "reviewed_setup_absolute_and_per_step",
        "reference": reference,
        "joint_order": list(JOINT_NAMES),
        "reference_action": list(previous),
        "lower": list(limits.lower),
        "upper": list(limits.upper),
        "reviewed_max_step_deltas": list(limits.max_step_deltas),
        "speed_scale": limits.speed_scale,
        "allowed_step_deltas": list(allowed),
        "absolute_violations": absolute,
        "rate_violations": rate,
        "would_be_clamped": clamped,
        "passed": not clamped,
    }


def camera_freshness(
    ages: Mapping[str, float],
    interarrival: Mapping[str, float | None],
) -> dict[str, Any]:
    return {
        "public_api": "OpenCVCamera.async_read",
        "arrival_clock": "time.perf_counter",
        "read_deadline_ms": MAXIMUM_FRAME_AGE_MS,
        "frame_age_ms": dict(ages),
        "interarrival_ms": dict(interarrival),
    }


def build_shadow_payload(
    candidate: AcceptedPolicyCandidate,
    run: ShadowRun,
    *,
    verification_payload: Mapping[str, Any],
    verification_sha256: str,
    repo: Mapping[str, Any],
    wandb: Mapping[str, str],
    setup_hashes: Mapping[str, str] | None = None,
    frozen_state_capture: Mapping[str, Any] | None = None,
    videos: Mapping[str, Mapping[str, Any]] | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Render the exact ``shadow_evidence.json`` object consumed by Repo B."""

    if verification_payload.get("status") != "eligible":
        raise ValidationError("shadow evidence requires eligible runtime verification")
    if verification_payload.get("bundle_id") != candidate.bundle_id:
        raise ValidationError("verification belongs to a different policy candidate")
    verification_wandb = verification_payload.get("wandb")
    if not isinstance(verification_wandb, Mapping):
        raise ValidationError("verification lacks its W&B identity")
    if run.mode == "replay":
        if setup_hashes is not None or frozen_state_capture is not None or videos:
            raise ValidationError("replay shadow cannot bind setup, frozen state, or videos")
    elif run.mode == "live-soak":
        if setup_hashes is None or frozen_state_capture is None or set(videos or {}) != {
            "front",
            "up",
        }:
            raise ValidationError("live soak requires setup, frozen capture, and both videos")
    else:
        raise ValidationError(f"unsupported shadow mode {run.mode!r}")

    return {
        "schema_version": 1,
        "kind": "shadow_evidence",
        "status": run.status,
        "bundle_id": candidate.bundle_id,
        "content_id": candidate.content_id,
        "policy": candidate.policy,
        "setup_hashes": None if setup_hashes is None else dict(setup_hashes),
        "frozen_state_capture": (
            None if frozen_state_capture is None else dict(frozen_state_capture)
        ),
        "runtime_binding": _json_copy(candidate.runtime_binding),
        "failure": None if run.failure is None else _json_copy(run.failure),
        "verification": {
            "sha256": verification_sha256,
            "run_id": verification_wandb["run_id"],
            "url": verification_wandb["url"],
        },
        "summary": _json_copy(run.summary),
        "trace": "action_trace.jsonl",
        "trace_file_sha256": run.trace_file_sha256,
        "videos": {name: dict(value) for name, value in (videos or {}).items()},
        "repo": dict(repo),
        "wandb": dict(wandb),
        "completed_at": completed_at or datetime.now(UTC).isoformat(),
    }


def shadow_sync_binding(
    candidate: AcceptedPolicyCandidate,
    payload: Mapping[str, Any],
    *,
    verification_sha256: str,
) -> dict[str, Any]:
    """Return Repo B's exact immutable binding for a shadow W&B run."""

    return {
        "bundle_id": candidate.bundle_id,
        "content_id": candidate.content_id,
        "policy": candidate.policy,
        "mode": payload["summary"]["mode"],
        "status": payload["status"],
        "repo_commit": payload["repo"]["commit"],
        "runtime_binding": _json_copy(candidate.runtime_binding),
        "verification_sha256": verification_sha256,
        "trace_file_sha256": payload["trace_file_sha256"],
        "action_trace_sha256": payload["summary"]["action_trace_sha256"],
        "frozen_state_sha256": payload["summary"]["frozen_state_sha256"],
        "setup_hashes": payload["setup_hashes"],
    }


def build_shadow_lineage(
    candidate: AcceptedPolicyCandidate,
    payload: Mapping[str, Any],
    *,
    shadow_evidence_sha256: str,
    shadow_sync_sha256: str,
    verification_sha256: str,
    verification_sync_sha256: str,
    frozen_state_sha256: str | None = None,
    frozen_state_capture_sha256: str | None = None,
    frozen_state_capture_sync_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the exact manifest lineage consumed by Repo B."""

    videos = payload["videos"]
    return {
        "policy_candidate_bundle_id": candidate.bundle_id,
        "policy_candidate_content_id": candidate.content_id,
        "policy": candidate.policy,
        "mode": payload["summary"]["mode"],
        "repo_a_commit": payload["repo"]["commit"],
        "wandb_run_id": payload["wandb"]["run_id"],
        "status": payload["status"],
        "runtime_binding": _json_copy(candidate.runtime_binding),
        "failure_code": None if payload["failure"] is None else payload["failure"]["code"],
        "shadow_schema_sha256": SHADOW_SCHEMA_SHA256,
        "shadow_evidence_sha256": shadow_evidence_sha256,
        "shadow_sync_sha256": shadow_sync_sha256,
        "verification_sha256": verification_sha256,
        "verification_sync_sha256": verification_sync_sha256,
        "video_sha256": {
            name: value["sha256"] for name, value in sorted(videos.items())
        },
        "frozen_state_sha256": frozen_state_sha256,
        "frozen_state_capture_sha256": frozen_state_capture_sha256,
        "frozen_state_capture_sync_sha256": frozen_state_capture_sync_sha256,
    }


def write_trace(path: str | Path, run: ShadowRun) -> Path:
    """Write canonical JSON lines to a caller-owned staging directory."""

    target = Path(path)
    target.write_bytes(run.trace_bytes())
    return target


def value_digest(value: Any) -> str:
    """Hash tensor/array input exactly as Repo B's shadow verifier does."""

    try:
        array = value.detach().cpu().contiguous().numpy()
    except AttributeError:
        try:
            import numpy as np

            array = np.ascontiguousarray(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError("shadow input cannot be serialized for hashing") from exc
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def state_vector_sha256(value: Sequence[float]) -> str:
    """Hash a frozen state in the canonical form bound by capture evidence."""

    state = finite_action(value, label="frozen state")
    return hashlib.sha256(canonical_json_bytes(list(state))).hexdigest()


def _canonical_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("task") != CANONICAL_TASK:
        raise ValidationError("replay frame task differs from the benchmark task")
    required = (
        "observation.state",
        "observation.images.front",
        "observation.images.up",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValidationError(f"replay frame is missing {missing}")
    return {key: raw[key] for key in required} | {"task": CANONICAL_TASK}


def _finish_replay(
    rows: list[dict[str, Any]],
    counters: dict[str, int],
    semantic: Any,
    runtime_resets: int,
    started: float,
    clock: Callable[[], float],
) -> ShadowRun:
    return _finish_shadow(
        "replay",
        rows,
        counters,
        semantic,
        runtime_resets=runtime_resets,
        wall_seconds=max(clock() - started, 0.0),
        frozen_state_sha256=None,
    )


def _finish_shadow(
    mode: str,
    rows: list[dict[str, Any]],
    counters: dict[str, int],
    semantic: Any,
    *,
    runtime_resets: int,
    wall_seconds: float,
    frozen_state_sha256: str | None,
) -> ShadowRun:
    completed = (
        counters["actions"] >= MINIMUM_ACTIONS
        and counters["actions"] / TARGET_HZ >= MINIMUM_SECONDS
        and (mode != "live-soak" or wall_seconds >= MINIMUM_SECONDS)
        and all(
            counters[name] == 0
            for name in (
                "deadline_misses",
                "rate_misses",
                "stale_frames",
                "limit_violations",
                "malformed_actions",
            )
        )
    )
    summary = {
        "mode": mode,
        **counters,
        "runtime_resets": runtime_resets,
        "logical_seconds": counters["actions"] / TARGET_HZ,
        "wall_seconds": wall_seconds,
        "action_trace_sha256": semantic.hexdigest(),
        "frozen_state_sha256": frozen_state_sha256,
        "completed": completed,
        "requirements": REQUIREMENTS,
    }
    failure = None if completed else _shadow_failure(rows, counters)
    if not completed and failure is None:
        raise ValidationError("incomplete shadow run has no typed positive safety signal")
    return ShadowRun(mode, tuple(rows), summary, failure)


def _shadow_failure(
    rows: Sequence[Mapping[str, Any]], counters: Mapping[str, int]
) -> dict[str, Any] | None:
    choices = (
        ("deadline_miss", "deadline_misses"),
        ("rate_miss", "rate_misses"),
        ("stale_observation", "stale_frames"),
        ("limit_violation", "limit_violations"),
        ("malformed_action", "malformed_actions"),
    )
    for code, counter in choices:
        if counters[counter] > 0:
            return {
                "code": code,
                "recoverable": False,
                "evidence": {
                    "counter": counter,
                    "count": counters[counter],
                    "trace_row_sha256": hashlib.sha256(
                        canonical_json_bytes(dict(rows[-1]))
                    ).hexdigest(),
                },
            }
    return None


def _empty_counters() -> dict[str, int]:
    return {
        "actions": 0,
        "captured_observations": 0,
        "replans": 0,
        "deadline_misses": 0,
        "rate_misses": 0,
        "stale_frames": 0,
        "limit_violations": 0,
        "malformed_actions": 0,
    }


def _scalar_integer(value: Any, label: str) -> int:
    try:
        value = value.item()
    except AttributeError:
        pass
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a nonnegative integer")
    return value


def _nonnegative_finite(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValidationError(f"{label} must be finite and nonnegative")
    return value


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(canonical_json_bytes(dict(value)))


__all__ = [
    "CANONICAL_LOWER",
    "CANONICAL_UPPER",
    "DEADLINE_MS",
    "MAXIMUM_FRAME_AGE_MS",
    "MINIMUM_ACTIONS",
    "MINIMUM_SECONDS",
    "REQUIREMENTS",
    "TARGET_HZ",
    "TARGET_PERIOD_MS",
    "CameraFrame",
    "CameraSource",
    "ReviewedLimits",
    "ShadowRun",
    "build_shadow_payload",
    "build_shadow_lineage",
    "camera_freshness",
    "canonical_limit_check",
    "deployment_limit_check",
    "run_live_soak",
    "run_replay",
    "shadow_sync_binding",
    "state_vector_sha256",
    "value_digest",
    "write_trace",
]
