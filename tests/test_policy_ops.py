from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np

from viola_handoff import RuntimeIdentity, inspect_bundle
from viola_ops.policies import CANONICAL_TASK, get_policy_spec
from viola_ops.policy_ops import ShadowEvidence, shadow_command, verify_command
from viola_ops.policy_runtime import AcceptedPolicyCandidate


class FakeEvidence:
    def record(self, *, project: str, run_id: str, event: str, metadata: Any) -> str:
        del event, metadata
        return f"https://wandb.ai/test/{project}/runs/{run_id}"


class FakeRuntime:
    def __init__(self, action: Any = (0.0,) * 7) -> None:
        self.action = action

    def reset(self) -> None:
        return None

    def infer(self, observation: Any) -> Any:
        del observation
        return self.action


class FastClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        result = self.value
        self.value += 0.001
        return result


def _candidate(token: str = "act") -> AcceptedPolicyCandidate:
    spec = get_policy_spec(token)
    digest = "a" * 64
    bundle = SimpleNamespace(
        bundle_id="b" * 64,
        content_id="c" * 64,
        manifest={"experiment": "viola-test"},
    )
    return AcceptedPolicyCandidate(
        bundle=bundle,
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


def _identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        role="pc_a",
        repository_commit="d" * 40,
        repository_clean=True,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _observation() -> dict[str, Any]:
    return {
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.images.front": np.zeros((1, 1, 3), dtype=np.uint8),
        "observation.images.up": np.ones((1, 1, 3), dtype=np.uint8),
        "task": CANONICAL_TASK,
    }


def _frames() -> list[dict[str, Any]]:
    lengths = (868, 868, 868, 868, 869, 869, 869)
    return [
        {"episode_index": episode, **_observation()}
        for episode, length in zip(range(27, 34), lengths, strict=True)
        for _ in range(length)
    ]


def _publisher_calls() -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def publish(identity: Any, **kwargs: Any) -> Any:
        calls.append({"identity": identity, **kwargs})
        return identity

    return calls, publish


def test_verify_command_records_online_sidecar_without_hardware(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate("pi0_fast")
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    calls, publisher = _publisher_calls()
    evidence = verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        tmp_path,
        wandb_entity="test",
        runtime_factory=lambda _candidate: FakeRuntime(),
        observation_loader=lambda _candidate: _observation(),
        publisher=publisher,
        producer_identity=_identity(),
        bundle_evidence_logger=FakeEvidence(),
        clock=FastClock(),
    )
    assert evidence.result.eligible
    assert evidence.verification_path.is_file()
    assert evidence.sync_path.is_file()
    assert evidence.terminal_bundle is None
    assert calls[0]["config"]["operation"] == "policy_verify"
    assert calls[0]["config"]["policy"] == "pi0_fast"
    assert calls[0]["summary"]["latency_trials"] == 200


def test_verify_command_seals_a_typed_runtime_terminal(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    _calls, publisher = _publisher_calls()
    evidence = verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        tmp_path,
        handoff_root=tmp_path / "handoffs",
        wandb_entity="test",
        runtime_factory=lambda _candidate: FakeRuntime([0.0] * 6 + [float("nan")]),
        observation_loader=lambda _candidate: _observation(),
        publisher=publisher,
        producer_identity=_identity(),
        bundle_evidence_logger=FakeEvidence(),
    )
    assert evidence.result.status == "ineligible_pc_runtime"
    assert evidence.terminal_path is not None
    assert evidence.terminal_bundle is not None
    verified = inspect_bundle(evidence.terminal_bundle.path, verify_artifacts=True)
    assert verified.kind == "shadow_evidence"
    assert not verified.manifest["artifacts"]
    assert {item["path"] for item in verified.manifest["payload"]["files"]} == {
        "shadow_evidence.json",
        "verification.json",
        "verification_WANDB_SYNCED.json",
    }


def test_replay_shadow_command_publishes_and_seals_repo_b_shape(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    calls, publisher = _publisher_calls()
    verification = verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        tmp_path,
        wandb_entity="test",
        runtime_factory=lambda _candidate: FakeRuntime(),
        observation_loader=lambda _candidate: _observation(),
        publisher=publisher,
        producer_identity=_identity(),
        clock=FastClock(),
    )
    result = shadow_command(
        "/accepted/candidate",
        "replay",
        verification.verification_path,
        tmp_path / "output",
        tmp_path,
        tmp_path / "handoffs",
        wandb_entity="test",
        runtime_factory=lambda _candidate: FakeRuntime(),
        replay_loader=lambda _candidate: _frames(),
        publisher=publisher,
        producer_identity=_identity(),
        bundle_evidence_logger=FakeEvidence(),
        clock=FastClock(),
    )
    assert result.run.status == "passed"
    verified = inspect_bundle(result.bundle.path, verify_artifacts=True)
    assert {item["path"] for item in verified.manifest["payload"]["files"]} == {
        "shadow_WANDB_SYNCED.json",
        "shadow_evidence.json",
        "verification.json",
        "verification_WANDB_SYNCED.json",
    }
    assert {item["name"] for item in verified.manifest["artifacts"]} == {"shadow_record"}
    assert {item["path"] for item in verified.manifest["artifacts"][0]["files"]} == {
        "action_trace.jsonl"
    }
    assert calls[-1]["config"]["operation"] == "policy_shadow"
    assert calls[-1]["config"]["mode"] == "replay"


def test_unsafe_shadow_summary_never_says_passed(tmp_path: Path) -> None:
    run = SimpleNamespace(
        status="unsafe_shadow",
        mode="replay",
        summary={"actions": 1, "logical_seconds": 1 / 30},
    )
    evidence = ShadowEvidence(
        candidate=_candidate(),
        run=run,
        root=tmp_path,
        payload_root=tmp_path / "payload",
        artifact_root=tmp_path / "record",
        bundle=SimpleNamespace(path=tmp_path / "bundle"),
    )
    rendered = evidence.render_text()
    assert "Policy shadow: unsafe; motion remains blocked" in rendered
    assert "Status: unsafe_shadow" in rendered
    assert "Policy shadow: passed" not in rendered
