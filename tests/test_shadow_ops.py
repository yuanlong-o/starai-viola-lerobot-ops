from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from viola_ops.policies import CANONICAL_TASK, POLICY_TOKENS, get_policy_spec
from viola_ops.policy_runtime import AcceptedPolicyCandidate
from viola_ops.shadow import (
    CameraFrame,
    ReviewedLimits,
    build_shadow_payload,
    run_live_soak,
    run_replay,
    state_vector_sha256,
    value_digest,
)


class IncrementingClock:
    def __init__(self, step: float = 0.0001) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        result = self.value
        self.value += self.step
        return result


class ScheduledClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0.0)


class FakeRuntime:
    def __init__(self, action: Any = (0.0,) * 7) -> None:
        self.action = action
        self.resets = 0
        self.calls = 0
        self.observation_keys: set[str] = set()

    def reset(self) -> None:
        self.resets += 1

    def infer(self, observation: Any) -> Any:
        self.calls += 1
        self.observation_keys = set(observation)
        return self.action


class FailingRuntime(FakeRuntime):
    def infer(self, observation: Any) -> Any:
        super().infer(observation)
        raise RuntimeError("runtime disconnected internally")


class FakeCamera:
    def __init__(self, clock: ScheduledClock, marker: int) -> None:
        self.clock = clock
        self.image = np.full((1, 1, 3), marker, dtype=np.uint8)
        self.reads = 0

    def read(self, *, deadline_ms: float) -> CameraFrame:
        assert deadline_ms == 100.0
        self.reads += 1
        return CameraFrame(self.image, self.clock.value)


def _candidate(token: str = "act") -> AcceptedPolicyCandidate:
    spec = get_policy_spec(token)
    digest = "a" * 64
    return AcceptedPolicyCandidate(
        bundle=SimpleNamespace(bundle_id="b" * 64, content_id="c" * 64),
        spec=spec,
        payload=MappingProxyType({"policy": token}),
        replay_manifest=MappingProxyType({"dataset_repo_id": "test/replay"}),
        checkpoint=Path("/accepted/checkpoint"),
        replay_dataset=Path("/accepted/replay"),
        dependencies=MappingProxyType({}),
        runtime_binding=MappingProxyType(
            {
                "checkpoint_inventory_sha256": digest,
                "checkpoint_config_sha256": digest,
                "processor_sha256": {"preprocessor": digest, "postprocessor": digest},
                "dependency_inventory_sha256": {},
            }
        ),
    )


def _replay_frames() -> list[dict[str, Any]]:
    lengths = (868, 868, 868, 868, 869, 869, 869)
    state = np.zeros(7, dtype=np.float32)
    front = np.zeros((1, 1, 3), dtype=np.uint8)
    up = np.ones((1, 1, 3), dtype=np.uint8)
    return [
        {
            "episode_index": episode,
            "task": CANONICAL_TASK,
            "observation.state": state,
            "observation.images.front": front,
            "observation.images.up": up,
        }
        for episode, length in zip(range(27, 34), lengths, strict=True)
        for _ in range(length)
    ]


def _limits() -> ReviewedLimits:
    return ReviewedLimits(
        lower=(-90.0, -90.0, -90.0, -90.0, -90.0, -90.0, 0.0),
        upper=(90.0,) * 7,
        max_step_deltas=(10.0,) * 7,
    )


def test_full_replay_meets_repo_b_minimums_and_resets_at_episode_boundaries() -> None:
    runtime = FakeRuntime()
    run = run_replay(
        _candidate(),
        runtime,
        _replay_frames(),
        clock=IncrementingClock(),
    )
    assert run.status == "passed"
    assert run.summary["actions"] == 9_000
    assert run.summary["captured_observations"] == 9_000
    assert run.summary["logical_seconds"] == 300.0
    assert run.summary["runtime_resets"] == 12
    assert run.summary["replans"] >= 900
    assert run.failure is None
    assert len(run.rows) == 9_000
    assert run.rows[0]["replan"] is True
    assert run.rows[10]["replan"] is True
    assert run.rows[6_079]["repeat"] == 1
    assert run.trace_file_sha256 == __import__("hashlib").sha256(run.trace_bytes()).hexdigest()


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_shadow_exercises_every_policy_and_types_malformed_actions(token: str) -> None:
    runtime = FakeRuntime([0.0] * 6 + [float("nan")])
    run = run_replay(
        _candidate(token),
        runtime,
        _replay_frames(),
        clock=IncrementingClock(),
    )
    assert run.status == "unsafe_shadow"
    assert run.failure is not None
    assert run.failure["code"] == "malformed_action"
    assert run.summary["malformed_actions"] == 1
    if token == "vqbet":
        assert "observation.images.up" not in runtime.observation_keys
    else:
        assert "observation.images.up" in runtime.observation_keys


def test_replay_runtime_exception_becomes_typed_unsafe_evidence() -> None:
    run = run_replay(
        _candidate(),
        FailingRuntime(),
        _replay_frames(),
        clock=IncrementingClock(),
    )
    assert run.status == "unsafe_shadow"
    assert run.failure is not None
    assert run.failure["code"] == "malformed_action"
    assert run.rows[-1]["event"] == "malformed_action"
    assert "runtime disconnected internally" in run.rows[-1]["detail"]


def test_live_soak_runtime_exception_becomes_typed_unsafe_evidence() -> None:
    clock = ScheduledClock()
    run = run_live_soak(
        _candidate(),
        FailingRuntime(),
        {"front": FakeCamera(clock, 1), "up": FakeCamera(clock, 2)},
        frozen_state=(0.0,) * 7,
        limits=_limits(),
        clock=clock,
        sleep=clock.sleep,
    )
    assert run.status == "unsafe_shadow"
    assert run.failure is not None
    assert run.failure["code"] == "malformed_action"
    assert run.rows[-1]["event"] == "malformed_action"


def test_live_soak_records_both_cameras_with_frozen_state_and_no_robot() -> None:
    clock = ScheduledClock()
    cameras = {"front": FakeCamera(clock, 1), "up": FakeCamera(clock, 2)}
    runtime = FakeRuntime()
    state = (0.0,) * 7
    run = run_live_soak(
        _candidate("vqbet"),
        runtime,
        cameras,
        frozen_state=state,
        limits=_limits(),
        clock=clock,
        sleep=clock.sleep,
    )
    assert run.status == "passed"
    assert run.summary["actions"] == 9_000
    assert run.summary["wall_seconds"] >= 300.0
    assert run.summary["frozen_state_sha256"] == state_vector_sha256(state)
    assert cameras["front"].reads == cameras["up"].reads == 9_000
    assert runtime.resets == 1
    assert "observation.images.up" not in runtime.observation_keys
    assert all(row["up_sha256"] for row in run.rows)


def test_live_soak_stops_on_a_positive_limit_violation() -> None:
    clock = ScheduledClock()
    run = run_live_soak(
        _candidate(),
        FakeRuntime((50.0,) * 7),
        {"front": FakeCamera(clock, 1), "up": FakeCamera(clock, 2)},
        frozen_state=(0.0,) * 7,
        limits=_limits(),
        clock=clock,
        sleep=clock.sleep,
    )
    assert run.status == "unsafe_shadow"
    assert run.summary["limit_violations"] == 1
    assert run.failure is not None
    assert run.failure["code"] == "limit_violation"
    assert len(run.rows) == 1


def test_shadow_payload_has_exact_repo_b_schema_shape() -> None:
    candidate = _candidate()
    run = run_replay(
        candidate,
        FakeRuntime(),
        _replay_frames(),
        clock=IncrementingClock(),
    )
    verification = {
        "status": "eligible",
        "bundle_id": candidate.bundle_id,
        "wandb": {
            "run_id": "verify-act",
            "url": "https://wandb.ai/test/project/runs/verify-act",
        },
    }
    payload = build_shadow_payload(
        candidate,
        run,
        verification_payload=verification,
        verification_sha256="d" * 64,
        repo={
            "commit": "e" * 40,
            "clean": True,
            "hostname": "pc-a",
            "python": "3.12.13",
        },
        wandb={
            "run_id": "replay-act",
            "url": "https://wandb.ai/test/project/runs/replay-act",
        },
        completed_at="2026-08-13T00:00:00+00:00",
    )
    assert set(payload) == {
        "schema_version",
        "kind",
        "status",
        "bundle_id",
        "content_id",
        "policy",
        "setup_hashes",
        "frozen_state_capture",
        "runtime_binding",
        "failure",
        "verification",
        "summary",
        "trace",
        "trace_file_sha256",
        "videos",
        "repo",
        "wandb",
        "completed_at",
    }
    schema_text = (
        Path(__file__).parents[1] / "contracts" / "shadow_evidence.schema.json"
    ).read_text()
    assert '"kind": {"const": "shadow_evidence"}' in schema_text
    assert '"minimum_actions": 9000' in schema_text
    assert payload["trace"] == "action_trace.jsonl"
    assert payload["trace_file_sha256"] == run.trace_file_sha256
