from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import viola_handoff
import viola_ops.policy_ops as policy_ops
from viola_handoff import RuntimeIdentity, inspect_bundle
from viola_ops.errors import ValidationError
from viola_ops.jsonutil import sha256_file, sha256_json, write_canonical_json
from viola_ops.policies import CANONICAL_TASK, POLICY_TOKENS, get_policy_spec
from viola_ops.policy_ops import (
    ShadowEvidence,
    _load_frozen_state,
    shadow_command,
    verify_command,
)
from viola_ops.policy_runtime import AcceptedPolicyCandidate


_REAL_SNAPSHOT_REPLAY_SOURCE = policy_ops._snapshot_replay_source


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


@pytest.fixture(autouse=True)
def _small_policy_input_snapshots(tmp_path: Path, monkeypatch: Any) -> None:
    """Keep orchestration tests small; policy_runtime tests cover the real copier."""

    replay = tmp_path / "accepted-replay"
    replay.mkdir()
    replay.joinpath("frame.bin").write_bytes(b"accepted replay")
    replay_snapshot = policy_ops._snapshot_evidence_roots(
        {"accepted replay dataset": replay}
    )[0]

    @contextmanager
    def passthrough(candidate: AcceptedPolicyCandidate):
        yield SimpleNamespace(candidate=candidate, verify=lambda: None)

    monkeypatch.setattr(policy_ops, "_snapshot_candidate_runtime", passthrough)
    monkeypatch.setattr(policy_ops, "_snapshot_replay_source", lambda _candidate: replay_snapshot)


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


def _install_live_shadow_fixture(tmp_path: Path, monkeypatch: Any) -> SimpleNamespace:
    """Install small reviewed setup/frozen-state fakes without opening devices."""

    setup_path = tmp_path / "reviewed-setup.json"
    setup_path.write_text("{}\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    reset = tmp_path / "reset.md"
    entrypoint = tmp_path / "executor.py"
    for path in (calibration, reset, entrypoint):
        path.write_text(f"reviewed {path.name}\n", encoding="utf-8")
    setup = SimpleNamespace(
        source_path=setup_path,
        setup_id="setup-one",
        cameras={},
        calibration_path=calibration,
        reset_protocol_path=reset,
        executor_entrypoint=entrypoint,
        executor={
            "commit": _identity().repository_commit,
            "entrypoint_sha256": sha256_file(entrypoint),
        },
        estop={"operator": "operator"},
        robot_port="/dev/never-opened",
        joint_limits={
            name: ([0.0, 1.0] if name == "gripper" else [-1.0, 1.0])
            for name in policy_ops.JOINT_NAMES
        },
        max_step_deltas={name: 0.1 for name in policy_ops.JOINT_NAMES},
        speed_scale=1.0,
    )
    monkeypatch.setattr(policy_ops, "load_reviewed_setup", lambda *_args, **_kwargs: setup)

    frozen_root = tmp_path / "frozen-state"
    frozen_root.mkdir()
    frozen_files = tuple(
        frozen_root.joinpath(name)
        for name in (
            "frozen_state.json",
            "frozen_state_capture.json",
            "frozen_state_capture_WANDB_SYNCED.json",
        )
    )
    for path in frozen_files:
        path.write_text(f"signed {path.name}\n", encoding="utf-8")
    monkeypatch.setattr(
        policy_ops,
        "_load_frozen_state",
        lambda *_args, **_kwargs: (
            (0.0,) * 7,
            {"setup_id": "setup-one"},
            frozen_files,
        ),
    )

    class Recorder:
        def __init__(self, _root: Path) -> None:
            pass

        def record(self, _frames: Any) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(policy_ops, "_VideoPairRecorder", Recorder)
    return SimpleNamespace(
        setup=setup,
        setup_path=setup_path,
        frozen_root=frozen_root,
        frozen_files=frozen_files,
        calibration=calibration,
        reset=reset,
        entrypoint=entrypoint,
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


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_verify_command_records_online_sidecar_without_hardware(
    tmp_path: Path, monkeypatch: Any, token: str
) -> None:
    candidate = _candidate(token)
    monkeypatch.setattr("viola_ops.policy_ops.inspect_candidate", lambda _path: candidate)
    private_checkpoint = tmp_path / "private-runtime-checkpoint"
    private_checkpoint.mkdir()
    runtime_candidates: list[AcceptedPolicyCandidate] = []

    @contextmanager
    def private_runtime_inputs(source: AcceptedPolicyCandidate):
        yield SimpleNamespace(
            candidate=replace(source, checkpoint=private_checkpoint),
            verify=lambda: None,
        )

    def runtime_factory(source: AcceptedPolicyCandidate) -> FakeRuntime:
        runtime_candidates.append(source)
        return FakeRuntime()

    monkeypatch.setattr(policy_ops, "_snapshot_candidate_runtime", private_runtime_inputs)
    calls, publisher = _publisher_calls()
    evidence = verify_command(
        "/accepted/candidate",
        tmp_path / "output",
        _repo_root(tmp_path),
        wandb_entity="test",
        runtime_factory=runtime_factory,
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
    assert calls[0]["config"]["policy"] == token
    assert calls[0]["summary"]["latency_trials"] == 200
    assert [source.checkpoint for source in runtime_candidates] == [private_checkpoint]


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


@pytest.mark.parametrize("mutation", ("evidence", "canonical-source"))
def test_ineligible_verify_guards_the_handoff_wandb_boundary(
    tmp_path: Path, monkeypatch: Any, mutation: str
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    output = tmp_path / "output"
    revoked = False

    def canonical(accepted: Any, **_kwargs: Any) -> Any:
        if revoked:
            raise viola_handoff.BundleValidationError("candidate was revoked")
        return accepted

    monkeypatch.setattr(viola_handoff, "require_active_canonical_source", canonical)

    class BoundaryLogger:
        def record(self, **kwargs: Any) -> str:
            nonlocal revoked
            if mutation == "evidence":
                next(output.rglob("shadow_evidence.json")).parent.joinpath(
                    "unexpected.txt"
                ).write_text("changed\n", encoding="utf-8")
            else:
                revoked = True
            return f"https://wandb.ai/test/{kwargs['project']}/runs/{kwargs['run_id']}"

    with pytest.raises(ValidationError):
        verify_command(
            "/accepted/candidate",
            output,
            _repo_root(tmp_path),
            handoff_root=tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime([0.0] * 6 + [float("nan")]),
            observation_loader=lambda _candidate: _observation(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=BoundaryLogger(),
            attempt_id=f"guard-{mutation}",
        )

    bundles = list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))
    assert len(bundles) == 1
    assert not (bundles[0] / "READY.json").exists()


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_replay_shadow_command_publishes_and_seals_repo_b_shape(
    tmp_path: Path, monkeypatch: Any, token: str
) -> None:
    candidate = _candidate(token)
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
    assert not (attempt / "payload" / "shadow_evidence.json").exists()
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


def test_replay_shadow_rejects_accepted_dataset_drift_during_the_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    replay = tmp_path / "accepted-dataset"
    replay.mkdir()
    replay_file = replay / "heldout.bin"
    replay_file.write_bytes(b"signed replay bytes")
    replay_inventory = viola_handoff.inventory_root(replay)
    base = _candidate()
    candidate = replace(
        base,
        replay_dataset=replay,
        bundle=SimpleNamespace(
            bundle_id=base.bundle_id,
            content_id=base.content_id,
            manifest={
                "experiment": "viola-test",
                "artifacts": [
                    {
                        "name": "replay_dataset",
                        "root": str(replay),
                        **replay_inventory,
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    monkeypatch.setattr(policy_ops, "_snapshot_replay_source", _REAL_SNAPSHOT_REPLAY_SOURCE)

    def changed_replay(_candidate: Any) -> list[dict[str, Any]]:
        replay_file.write_bytes(b"changed replay bytes")
        return _frames()

    with pytest.raises(ValidationError, match="policy evidence changed"):
        shadow_command(
            "/accepted/candidate",
            "replay",
            verification.verification_path,
            tmp_path / "output",
            repository,
            tmp_path / "handoffs",
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            replay_loader=changed_replay,
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            clock=FastClock(),
            attempt_id="changed-accepted-replay",
        )

    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


def test_shadow_guards_artifacts_across_handoff_wandb(
    tmp_path: Path, monkeypatch: Any
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    output = tmp_path / "output"

    class MutatingLogger:
        def record(self, **kwargs: Any) -> str:
            next(output.rglob("shadow_record")).joinpath("unexpected.txt").write_text(
                "changed\n", encoding="utf-8"
            )
            return f"https://wandb.ai/test/{kwargs['project']}/runs/{kwargs['run_id']}"

    with pytest.raises(ValidationError, match="shadow artifact bytes changed"):
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
            bundle_evidence_logger=MutatingLogger(),
            clock=FastClock(),
            attempt_id="mutated-during-handoff-wandb",
        )

    bundle = next((tmp_path / "handoffs" / "shadow_evidence").glob("*"))
    assert not (bundle / "READY.json").exists()


@pytest.mark.parametrize("source_name", ("verification", "sync"))
def test_shadow_rejects_verification_changed_between_load_and_copy(
    tmp_path: Path,
    monkeypatch: Any,
    source_name: str,
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    verification_path = verification.verification_path
    sync_path = verification.sync_path
    target = verification_path if source_name == "verification" else sync_path
    original_copy = policy_ops.copy_regular_file
    mutated = False

    def mutate_then_copy(source: Path, destination: Path) -> Path:
        nonlocal mutated
        if Path(source) == target and not mutated:
            mutated = True
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b"\n")
        return original_copy(source, destination)

    monkeypatch.setattr(policy_ops, "copy_regular_file", mutate_then_copy)
    with pytest.raises(
        ValidationError,
        match="copied runtime verification differs|reviewed policy input changed",
    ):
        shadow_command(
            "/accepted/candidate",
            "replay",
            verification_path,
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
            attempt_id=f"changed-{source_name}-before-copy",
        )

    assert mutated is True
    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


@pytest.mark.parametrize("mutation", ("setup", "frozen-state"))
def test_live_soak_rechecks_reviewed_inputs_after_the_long_run(
    tmp_path: Path, monkeypatch: Any, mutation: str
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    live = _install_live_shadow_fixture(tmp_path, monkeypatch)

    def mutate_reviewed_source(*_args: Any, **_kwargs: Any) -> Any:
        target = live.setup_path if mutation == "setup" else live.frozen_files[0]
        target.write_text("changed during soak\n", encoding="utf-8")
        return SimpleNamespace(status="passed", summary={"captured_observations": 0})

    monkeypatch.setattr(policy_ops, "run_live_soak", mutate_reviewed_source)
    with pytest.raises(ValidationError, match="changed during shadow|policy evidence changed"):
        shadow_command(
            "/accepted/candidate",
            "live-soak",
            verification.verification_path,
            tmp_path / "output",
            repository,
            tmp_path / "handoffs",
            setup_path=live.setup_path,
            frozen_state_path=live.frozen_root,
            wandb_entity="test",
            cameras={"front": object(), "up": object()},
            runtime_factory=lambda _candidate: FakeRuntime(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            attempt_id=f"changed-live-{mutation}",
        )

    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


@pytest.mark.parametrize("source_name", ("calibration", "reset", "entrypoint"))
def test_live_soak_rejects_setup_file_changed_while_being_pinned(
    tmp_path: Path,
    monkeypatch: Any,
    source_name: str,
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    live = _install_live_shadow_fixture(tmp_path, monkeypatch)
    target = getattr(live, source_name)
    loads = 0

    def mutate_on_reload(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal loads
        loads += 1
        if loads == 2:
            target.write_text("changed while pinning\n", encoding="utf-8")
        return live.setup

    monkeypatch.setattr(policy_ops, "load_reviewed_setup", mutate_on_reload)
    camera_calls: list[object] = []
    monkeypatch.setattr(
        policy_ops._OpenedCameras,
        "open",
        classmethod(lambda cls, *_args, **_kwargs: camera_calls.append(object())),
    )

    with pytest.raises(ValidationError, match="reviewed policy input changed"):
        shadow_command(
            "/accepted/candidate",
            "live-soak",
            verification.verification_path,
            tmp_path / "output",
            repository,
            tmp_path / "handoffs",
            setup_path=live.setup_path,
            frozen_state_path=live.frozen_root,
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            attempt_id=f"changed-{source_name}-while-pinning",
        )

    assert camera_calls == []


def test_live_soak_rechecks_canonical_candidate_before_camera_open(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(policy_ops, "inspect_candidate", lambda _path: candidate)
    repository = _repo_root(tmp_path)
    verification = _eligible_verification_for_shadow(tmp_path, repository)
    live = _install_live_shadow_fixture(tmp_path, monkeypatch)
    freshness_checks = 0

    def revoke_on_final_pre_camera_check(accepted: Any, **_kwargs: Any) -> Any:
        nonlocal freshness_checks
        freshness_checks += 1
        if freshness_checks == 3:
            raise viola_handoff.BundleValidationError("candidate revoked")
        return accepted

    monkeypatch.setattr(
        viola_handoff,
        "require_active_canonical_source",
        revoke_on_final_pre_camera_check,
    )
    camera_calls: list[object] = []
    monkeypatch.setattr(
        policy_ops._OpenedCameras,
        "open",
        classmethod(lambda cls, *_args, **_kwargs: camera_calls.append(object())),
    )

    with pytest.raises(ValidationError, match="canonical policy candidate"):
        shadow_command(
            "/accepted/candidate",
            "live-soak",
            verification.verification_path,
            tmp_path / "output",
            repository,
            tmp_path / "handoffs",
            setup_path=live.setup_path,
            frozen_state_path=live.frozen_root,
            wandb_entity="test",
            runtime_factory=lambda _candidate: FakeRuntime(),
            publisher=lambda identity, **_kwargs: identity,
            producer_identity=_identity(),
            bundle_evidence_logger=FakeEvidence(),
            attempt_id="revoked-before-camera-open",
        )

    assert camera_calls == []


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
        traces = list(output.rglob("action_trace.jsonl"))
        if traces:
            trace = traces[0]
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

    assert freshness_checks >= 4
    assert not list((tmp_path / "handoffs" / "shadow_evidence").glob("*"))


def test_sync_receipt_is_the_only_exact_change_allowed_after_publication(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    payload.joinpath("verification.json").write_bytes(b"verified")
    before = policy_ops._snapshot_evidence_roots({"verification payload": payload})
    receipt = payload / "verification_WANDB_SYNCED.json"
    receipt.write_bytes(b"wrong receipt")

    with pytest.raises(ValidationError, match="differs from its exact bytes"):
        policy_ops._expect_added_files(before, {receipt: b"expected receipt"})


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
    assert not (attempt / "verification.json").exists()
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
