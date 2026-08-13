from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import viola_handoff
from viola_handoff import RuntimeIdentity, inspect_bundle
from viola_ops.errors import ValidationError
from viola_ops.jsonutil import sha256_file, sha256_json, write_canonical_json
from viola_ops.policies import CANONICAL_TASK, get_policy_spec
from viola_ops.policy_ops import (
    ShadowEvidence,
    _load_frozen_state,
    shadow_command,
    verify_command,
)
from viola_ops.policy_runtime import AcceptedPolicyCandidate


class FakeEvidence:
    def record(self, *, project: str, run_id: str, event: str, metadata: Any) -> str:
        del event, metadata
        return f"https://wandb.ai/test/{project}/runs/{run_id}"


class FailingEvidence:
    def record(self, *, project: str, run_id: str, event: str, metadata: Any) -> str:
        del project, run_id, event, metadata
        raise ValidationError("simulated handoff publication failure")


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


@pytest.fixture(autouse=True)
def _stable_runtime_identity(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    monkeypatch.setattr(
        viola_handoff,
        "require_active_canonical_source",
        lambda accepted, **_kwargs: accepted,
    )


def _repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    return root


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


def _eligible_verification_for_shadow(
    tmp_path: Path,
    repository: Path,
) -> Any:
    return verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        repository,
        handoff_root=tmp_path / "handoffs",
        wandb_entity="test",
        runtime_factory=lambda _candidate: FakeRuntime(),
        observation_loader=lambda _candidate: _observation(),
        publisher=lambda identity, **_kwargs: identity,
        producer_identity=_identity(),
        clock=FastClock(),
        attempt_id="verification-for-shadow",
    )


def _frozen_state_material(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, Any]]:
    root = tmp_path / "frozen"
    root.mkdir()
    captured_at = datetime(2026, 8, 13, 1, 0, tzinfo=UTC)
    disconnected_at = captured_at + timedelta(milliseconds=1)
    values = [0.0] * 7
    state = {
        "schema_version": 1,
        "state": values,
        "captured_at": captured_at.isoformat(),
        "robot_connected_at_capture": True,
        "motor_disconnected_at": disconnected_at.isoformat(),
    }
    state_path = write_canonical_json(root / "frozen_state.json", state)
    setup_hashes = {
        "calibration": "1" * 64,
        "camera": "2" * 64,
        "robot": "3" * 64,
        "reset": "4" * 64,
    }
    repo = {
        "commit": _identity().repository_commit,
        "clean": True,
        "hostname": "pc-a",
        "python": "3.12.13",
    }
    state_sha = sha256_json(values)
    wandb = {
        "run_id": "capture-test",
        "url": "https://wandb.ai/test/project/runs/capture-test",
    }
    capture = {
        "schema_version": 1,
        "kind": "frozen_state_capture",
        "status": "captured_disconnected",
        "setup_id": "setup-one",
        "frozen_state_file": state_path.name,
        "frozen_state_sha256": sha256_file(state_path),
        "state_sha256": state_sha,
        "setup_hashes": setup_hashes,
        "operator": "operator",
        "captured_at": captured_at.isoformat(),
        "motor_disconnected_at": disconnected_at.isoformat(),
        "repo": repo,
        "wandb": wandb,
    }
    capture_path = write_canonical_json(root / "frozen_state_capture.json", capture)
    write_canonical_json(
        root / "frozen_state_capture_WANDB_SYNCED.json",
        {
            "schema_version": 1,
            "operation": "frozen_state_capture",
            "evidence_file": capture_path.name,
            "evidence_sha256": sha256_file(capture_path),
            "wandb": wandb,
            "binding": {
                "setup_id": "setup-one",
                "state_sha256": state_sha,
                "status": "captured_disconnected",
                "repo_commit": repo["commit"],
                "setup_hashes": setup_hashes,
                "frozen_state_sha256": sha256_file(state_path),
            },
            "synced_at": (disconnected_at + timedelta(milliseconds=1)).isoformat(),
        },
    )
    return root, setup_hashes, repo


def test_verify_command_records_online_sidecar_without_hardware(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate("pi0_fast")
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    calls, publisher = _publisher_calls()
    evidence = verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        _repo_root(tmp_path),
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
        _repo_root(tmp_path),
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
        _repo_root(tmp_path),
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
        _repo_root(tmp_path),
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


def test_shadow_rejects_a_revoked_canonical_candidate_before_runtime_load(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    monkeypatch.setattr(
        viola_handoff,
        "require_active_canonical_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            viola_handoff.BundleValidationError("candidate is revoked")
        ),
    )
    runtime_loads = 0

    def runtime_factory(_candidate: Any) -> FakeRuntime:
        nonlocal runtime_loads
        runtime_loads += 1
        return FakeRuntime()

    with pytest.raises(ValidationError, match="unavailable, changed, or revoked"):
        shadow_command(
            "/accepted/candidate",
            "replay",
            verification.verification_path,
            tmp_path / "output",
            repository,
            tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=runtime_factory,
            replay_loader=lambda _candidate: _frames(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            clock=FastClock(),
            attempt_id="revoked-before-shadow",
        )

    assert runtime_loads == 0


def test_shadow_rejects_revocation_during_online_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    freshness_checks = 0

    def canonical(accepted: Any, **_kwargs: Any) -> Any:
        nonlocal freshness_checks
        freshness_checks += 1
        if freshness_checks == 2:
            raise viola_handoff.BundleValidationError("candidate was revoked")
        return accepted

    monkeypatch.setattr(viola_handoff, "require_active_canonical_source", canonical)
    with pytest.raises(ValidationError, match="unavailable, changed, or revoked"):
        shadow_command(
            "/accepted/candidate",
            "replay",
            verification.verification_path,
            tmp_path / "output",
            repository,
            tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            replay_loader=lambda _candidate: _frames(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            clock=FastClock(),
            attempt_id="revoked-during-shadow-publish",
        )

    assert freshness_checks == 2
    attempt = next((tmp_path / "output").rglob("revoked-during-shadow-publish"))
    assert (attempt / "payload" / "shadow_evidence.json").is_file()
    assert not (attempt / "payload" / "shadow_WANDB_SYNCED.json").exists()
    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


def test_shadow_rejects_artifact_mutation_during_online_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    output = tmp_path / "output"

    def mutate_artifact(identity: Any, **_kwargs: Any) -> Any:
        trace = next(output.rglob("action_trace.jsonl"))
        trace.write_bytes(trace.read_bytes() + b"{}\n")
        return identity

    with pytest.raises(ValidationError, match="policy evidence changed"):
        shadow_command(
            "/accepted/candidate",
            "replay",
            verification.verification_path,
            output,
            repository,
            tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            replay_loader=lambda _candidate: _frames(),
            publisher=mutate_artifact,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            clock=FastClock(),
            attempt_id="mutated-shadow-artifact",
        )

    attempt = next(output.rglob("mutated-shadow-artifact"))
    assert not (attempt / "payload" / "shadow_WANDB_SYNCED.json").exists()
    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


def test_shadow_rechecks_evidence_immediately_before_sealing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    output = tmp_path / "output"
    freshness_checks = 0

    def mutate_during_final_freshness(accepted: Any, **_kwargs: Any) -> Any:
        nonlocal freshness_checks
        freshness_checks += 1
        if freshness_checks == 3:
            trace = next(output.rglob("action_trace.jsonl"))
            trace.write_bytes(trace.read_bytes() + b"{}\n")
        return accepted

    monkeypatch.setattr(
        viola_handoff,
        "require_active_canonical_source",
        mutate_during_final_freshness,
    )
    with pytest.raises(ValidationError, match="policy evidence changed"):
        shadow_command(
            "/accepted/candidate",
            "replay",
            verification.verification_path,
            output,
            repository,
            tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            replay_loader=lambda _candidate: _frames(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            clock=FastClock(),
            attempt_id="mutated-before-shadow-seal",
        )

    assert freshness_checks == 3
    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


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


def test_verify_retry_uses_a_fresh_attempt_after_wandb_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    run_ids: list[str] = []
    runtime_loads = 0

    def runtime_factory(_candidate: Any) -> FakeRuntime:
        nonlocal runtime_loads
        runtime_loads += 1
        return FakeRuntime()

    def publisher(identity: Any, **_kwargs: Any) -> Any:
        run_ids.append(identity.run_id)
        if len(run_ids) == 1:
            raise ValidationError("simulated W&B failure")
        return identity

    arguments = {
        "bundle": "/accepted/candidate",
        "output_root": tmp_path / "output",
        "repo_root": _repo_root(tmp_path),
        "wandb_entity": "test",
        "runtime_factory": runtime_factory,
        "observation_loader": lambda _candidate: _observation(),
        "publisher": publisher,
        "producer_identity": _identity(),
        "clock": FastClock(),
    }
    with pytest.raises(ValidationError, match="simulated W&B failure"):
        verify_command(**arguments)

    attempts_root = (
        tmp_path
        / "output"
        / "act"
        / candidate.bundle_id
        / _identity().repository_commit
        / "verification"
    )
    first = next(attempts_root.iterdir())
    first_bytes = (first / "verification.json").read_bytes()
    assert not (first / "verification_WANDB_SYNCED.json").exists()

    evidence = verify_command(**arguments)
    assert evidence.root != first
    assert len(evidence.root.name) == len(first.name) == 32
    assert (first / "verification.json").read_bytes() == first_bytes
    assert runtime_loads == 2
    assert len(set(run_ids)) == 2


def test_shadow_seal_retry_keeps_failed_attempt_unready_and_isolated(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    _calls, publisher = _publisher_calls()
    verification = verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        _repo_root(tmp_path),
        wandb_entity="test",
        runtime_factory=lambda _candidate: FakeRuntime(),
        observation_loader=lambda _candidate: _observation(),
        publisher=publisher,
        producer_identity=_identity(),
        clock=FastClock(),
        attempt_id="verification",
    )
    runtime_loads = 0
    replay_loads = 0
    shadow_run_ids: list[str] = []

    def runtime_factory(_candidate: Any) -> FakeRuntime:
        nonlocal runtime_loads
        runtime_loads += 1
        return FakeRuntime([0.0] * 6 + [float("nan")])

    def replay_loader(_candidate: Any) -> list[dict[str, Any]]:
        nonlocal replay_loads
        replay_loads += 1
        return _frames()

    def shadow_publisher(identity: Any, **_kwargs: Any) -> Any:
        shadow_run_ids.append(identity.run_id)
        return identity

    arguments = {
        "bundle": "/accepted/candidate",
        "mode": "replay",
        "verification_path": verification.verification_path,
        "output_root": tmp_path / "output",
        "repo_root": _repo_root(tmp_path),
        "handoff_root": tmp_path / "handoffs",
        "wandb_entity": "test",
        "runtime_factory": runtime_factory,
        "replay_loader": replay_loader,
        "publisher": shadow_publisher,
        "producer_identity": _identity(),
        "clock": FastClock(),
    }
    with pytest.raises(ValidationError, match="simulated handoff publication failure"):
        shadow_command(
            **arguments,
            bundle_evidence_logger=FailingEvidence(),
            attempt_id="attempt-one",
        )

    bundles = list((tmp_path / "handoffs" / "shadow_evidence").iterdir())
    assert len(bundles) == 1
    failed_bundle = bundles[0]
    failed_manifest = (failed_bundle / "manifest.json").read_bytes()
    assert not (failed_bundle / "READY.json").exists()

    evidence = shadow_command(
        **arguments,
        bundle_evidence_logger=FakeEvidence(),
        attempt_id="attempt-two",
    )
    assert evidence.root.name == "attempt-two"
    assert evidence.bundle.path != failed_bundle
    assert (failed_bundle / "manifest.json").read_bytes() == failed_manifest
    assert not (failed_bundle / "READY.json").exists()
    assert (evidence.bundle.path / "READY.json").is_file()
    assert runtime_loads == replay_loads == 2
    assert len(set(shadow_run_ids)) == 2


def test_attempt_ids_are_path_safe_and_existing_attempts_fail_before_inference(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    loads = 0

    def observation_loader(_candidate: Any) -> dict[str, Any]:
        nonlocal loads
        loads += 1
        return _observation()

    common = {
        "bundle": "/accepted/candidate",
        "output_root": tmp_path / "output",
        "repo_root": _repo_root(tmp_path),
        "wandb_entity": "test",
        "runtime_factory": lambda _candidate: FakeRuntime(),
        "observation_loader": observation_loader,
        "publisher": _publisher_calls()[1],
        "producer_identity": _identity(),
        "clock": FastClock(),
    }
    with pytest.raises(ValidationError, match="path-safe"):
        verify_command(**common, attempt_id="../escape")
    assert loads == 0

    verify_command(**common, attempt_id="same-attempt")
    with pytest.raises(ValidationError, match="will not be reused"):
        verify_command(**common, attempt_id="same-attempt")
    assert loads == 1


def test_verify_rejects_untrusted_identity_and_symlink_output_before_inference(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    loads = 0

    def observation_loader(_candidate: Any) -> dict[str, Any]:
        nonlocal loads
        loads += 1
        return _observation()

    arguments = {
        "bundle": "/accepted/candidate",
        "output_root": tmp_path / "output",
        "repo_root": _repo_root(tmp_path),
        "wandb_entity": "test",
        "runtime_factory": lambda _candidate: FakeRuntime(),
        "observation_loader": observation_loader,
        "publisher": _publisher_calls()[1],
        "clock": FastClock(),
    }
    unsafe_identity = RuntimeIdentity(
        role="pc_a",
        repository_commit="../unreviewed",
        repository_clean=True,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )
    with pytest.raises(ValidationError, match="full lowercase Git commit SHA"):
        verify_command(**arguments, producer_identity=unsafe_identity)

    with pytest.raises(ValidationError, match="outside the Repo-A worktree"):
        verify_command(
            **{**arguments, "output_root": arguments["repo_root"] / "evidence"},
            producer_identity=_identity(),
        )

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        verify_command(
            **{**arguments, "output_root": linked_output},
            producer_identity=_identity(),
        )
    assert loads == 0


def test_verify_rechecks_clean_identity_before_online_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    changed = RuntimeIdentity(
        role="pc_a",
        repository_commit="e" * 40,
        repository_clean=True,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )
    monkeypatch.setattr(
        RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: changed),
    )
    publications: list[Any] = []

    with pytest.raises(ValidationError, match="changed during policy evidence"):
        verify_command(
            "/accepted/candidate",
            tmp_path / "output",
            _repo_root(tmp_path),
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            observation_loader=lambda _candidate: _observation(),
            publisher=lambda *args, **kwargs: publications.append((args, kwargs)),
            producer_identity=_identity(),
            clock=FastClock(),
        )

    assert publications == []


def test_verify_rejects_a_revoked_canonical_candidate_before_inference(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    monkeypatch.setattr(
        viola_handoff,
        "require_active_canonical_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            viola_handoff.BundleValidationError("candidate is revoked")
        ),
    )
    loads = 0

    def observation_loader(_candidate: Any) -> dict[str, Any]:
        nonlocal loads
        loads += 1
        return _observation()

    with pytest.raises(ValidationError, match="unavailable, changed, or revoked"):
        verify_command(
            "/accepted/candidate",
            tmp_path / "output",
            _repo_root(tmp_path),
            handoff_root=tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            observation_loader=observation_loader,
            publisher=_publisher_calls()[1],
            producer_identity=_identity(),
            clock=FastClock(),
        )

    assert loads == 0
    assert not (tmp_path / "output").exists()


def test_verify_rejects_revocation_during_online_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    freshness_checks = 0

    def canonical(accepted: Any, **_kwargs: Any) -> Any:
        nonlocal freshness_checks
        freshness_checks += 1
        if freshness_checks == 2:
            raise viola_handoff.BundleValidationError("candidate was revoked")
        return accepted

    monkeypatch.setattr(viola_handoff, "require_active_canonical_source", canonical)
    with pytest.raises(ValidationError, match="unavailable, changed, or revoked"):
        verify_command(
            "/accepted/candidate",
            tmp_path / "output",
            _repo_root(tmp_path),
            handoff_root=tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            observation_loader=lambda _candidate: _observation(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            clock=FastClock(),
            attempt_id="revoked-during-publish",
        )

    assert freshness_checks == 2
    attempt = next((tmp_path / "output").rglob("revoked-during-publish"))
    assert (attempt / "verification.json").is_file()
    assert not (attempt / "verification_WANDB_SYNCED.json").exists()


def test_verify_rejects_evidence_root_replacement_during_publication(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    output = tmp_path / "output"

    def replace_attempt_root(identity: Any, **_kwargs: Any) -> Any:
        evidence_path = next(output.rglob("verification.json"))
        attempt = evidence_path.parent
        attempt.rename(attempt.with_name(f"{attempt.name}-original"))
        attempt.mkdir()
        return identity

    with pytest.raises(ValidationError, match="policy evidence changed"):
        verify_command(
            "/accepted/candidate",
            output,
            _repo_root(tmp_path),
            handoff_root=tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            observation_loader=lambda _candidate: _observation(),
            publisher=replace_attempt_root,
            producer_identity=_identity(),
            clock=FastClock(),
            attempt_id="replaced-during-publish",
        )

    attempt = next(output.rglob("replaced-during-publish"))
    assert not (attempt / "verification_WANDB_SYNCED.json").exists()


def test_frozen_state_is_bound_to_reviewed_setup_and_operator(tmp_path: Path) -> None:
    root, setup_hashes, repo = _frozen_state_material(tmp_path)

    state, binding, files = _load_frozen_state(
        root,
        expected_setup_id="setup-one",
        expected_operator="operator",
        expected_setup_hashes=setup_hashes,
        expected_repo=repo,
    )

    assert state == (0.0,) * 7
    assert binding["setup_id"] == "setup-one"
    assert tuple(path.name for path in files) == (
        "frozen_state.json",
        "frozen_state_capture.json",
        "frozen_state_capture_WANDB_SYNCED.json",
    )
    for field, wrong in (("expected_setup_id", "setup-two"), ("expected_operator", "other")):
        arguments = {
            "expected_setup_id": "setup-one",
            "expected_operator": "operator",
            "expected_setup_hashes": setup_hashes,
            "expected_repo": repo,
        }
        arguments[field] = wrong
        with pytest.raises(ValidationError, match="differs from the reviewed live setup"):
            _load_frozen_state(root, **arguments)


def test_frozen_state_rejects_symlinks_and_sync_before_disconnect(tmp_path: Path) -> None:
    root, setup_hashes, repo = _frozen_state_material(tmp_path)
    sync_path = root / "frozen_state_capture_WANDB_SYNCED.json"
    sync = json.loads(sync_path.read_bytes())
    sync["synced_at"] = "2026-08-13T00:59:59+00:00"
    sync_path.unlink()
    write_canonical_json(sync_path, sync)
    with pytest.raises(ValidationError, match="predates serial disconnection"):
        _load_frozen_state(
            root,
            expected_setup_id="setup-one",
            expected_operator="operator",
            expected_setup_hashes=setup_hashes,
            expected_repo=repo,
        )

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    state_path = root / "frozen_state.json"
    state_path.unlink()
    state_path.symlink_to(target)
    with pytest.raises(ValidationError, match="regular nonsymlink file"):
        _load_frozen_state(
            root,
            expected_setup_id="setup-one",
            expected_operator="operator",
            expected_setup_hashes=setup_hashes,
            expected_repo=repo,
        )
