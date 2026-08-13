from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from viola_handoff import (
    CONTRACT_SHA256,
    KIND_RULES,
    SUPPORTED_KINDS,
    BundleValidationError,
    EnvironmentValidationError,
    EvidenceError,
    RuntimeIdentity,
    SealRequest,
    accept_bundle,
    ack_bundle,
    canonical_json_bytes,
    inspect_bundle,
    inventory_root,
    require_active_canonical_source,
    seal_bundle,
    seal_validated_rollout_session_bundle,
    wandb_run_id,
)
from viola_handoff.evidence import _require_remote_finished


@dataclass
class FakeEvidence:
    url: str = ""
    fixed_url: bool = False
    fail: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **event: Any) -> str:
        if self.fail:
            raise EvidenceError("simulated online failure")
        self.events.append(event)
        if not self.fixed_url:
            self.url = f"https://wandb.ai/tester/{event['project']}/runs/{event['run_id']}"
        return self.url


@pytest.fixture
def pc_a() -> RuntimeIdentity:
    return RuntimeIdentity(
        role="pc_a",
        repository_commit="a" * 40,
        repository_clean=True,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


@pytest.fixture
def pc_b() -> RuntimeIdentity:
    return RuntimeIdentity(
        role="pc_b",
        repository_commit="b" * 40,
        repository_clean=True,
        hostname="pc-b",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    payload = tmp_path / "payload-source"
    (payload / "nested").mkdir(parents=True)
    (payload / "lineage.json").write_bytes(b'{"episodes":34}')
    (payload / "nested" / "evidence.txt").write_bytes(b"small evidence")
    artifact = tmp_path / "artifact-source"
    (artifact / "checkpoint").mkdir(parents=True)
    (artifact / "checkpoint" / "model.safetensors").write_bytes(b"model bytes")
    (artifact / "checkpoint" / "config.json").write_bytes(b"{}")
    return payload, artifact


def _request(
    tmp_path: Path,
    identity: RuntimeIdentity,
    sources: tuple[Path, Path],
    **changes: Any,
) -> SealRequest:
    payload, artifact = sources
    values: dict[str, Any] = {
        "root": tmp_path / "nas",
        "kind": "dataset_release",
        "experiment": "right-to-left-v1",
        "subject": "34 accepted episodes",
        "producer": identity,
        "lineage": {"dataset_release_id": "rtl-v1", "accepted_episodes": 34},
        "wandb_project": "viola-handoffs",
        "payload_dir": payload,
        "artifact_roots": {"dataset": artifact},
        "created_at": "2026-08-12T12:00:00Z",
    }
    values.update(changes)
    return SealRequest(**values)


def _seal(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
    logger: FakeEvidence | None = None,
):
    logger = logger or FakeEvidence()
    return seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=logger), logger


def test_canonical_json_is_stable_and_rejects_non_json_numbers() -> None:
    assert canonical_json_bytes({"z": [2, 1], "a": "é"}) == b'{"a":"\xc3\xa9","z":[2,1]}'
    with pytest.raises(BundleValidationError, match="non-finite"):
        canonical_json_bytes({"bad": math.nan})
    with pytest.raises(BundleValidationError, match="non-finite"):
        canonical_json_bytes({"bad": math.inf})
    with pytest.raises(BundleValidationError, match="non-string"):
        canonical_json_bytes({1: "not JSON"})
    with pytest.raises(BundleValidationError, match="non-JSON"):
        canonical_json_bytes({"tuple": (1, 2)})


def test_public_inventory_root_matches_seal_and_rejects_symlinks(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    _, artifact = sources
    inventory = inventory_root(artifact)
    assert set(inventory) == {
        "directories",
        "files",
        "file_count",
        "byte_count",
        "inventory_sha256",
    }
    assert inventory["file_count"] == 2
    bundle, _ = _seal(tmp_path, pc_a, sources)
    assert inventory == {
        key: value
        for key, value in bundle.manifest["artifacts"][0].items()
        if key not in {"name", "root"}
    }

    link = artifact / "checkpoint" / "escape"
    link.symlink_to("/etc/passwd")
    with pytest.raises(BundleValidationError, match="symlink"):
        inventory_root(artifact)


def test_seal_is_content_addressed_ready_and_retry_safe(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    logger = FakeEvidence()
    first = seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=logger)
    second = seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=logger)

    assert first.path == second.path
    assert first.bundle_id == first.content_id
    assert first.path.name == first.bundle_id
    assert len(first.content_id) == 64
    assert first.manifest["contract_sha256"] == CONTRACT_SHA256
    assert first.manifest["wandb"]["run_id"] == wandb_run_id(first.content_id)
    assert first.ready["wandb_url"] == logger.url
    assert len(logger.events) == 1
    assert logger.events[0]["event"] == "sealed"
    assert all(
        isinstance(value, (str, int, float, bool, type(None)))
        for value in logger.events[0]["metadata"].values()
    )
    assert "model bytes" not in repr(logger.events)
    assert sorted(path.name for path in first.path.iterdir()) == [
        "READY.json",
        "manifest.json",
        "payload",
        "receipts",
    ]


def test_content_id_ignores_timestamp_machine_and_artifact_location(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    first, _ = _seal(tmp_path, pc_a, sources)
    payload, artifact = sources
    relocated_artifact = tmp_path / "relocated-artifact"
    for source_file in sorted(path for path in artifact.rglob("*") if path.is_file()):
        destination = relocated_artifact / source_file.relative_to(artifact)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_file.read_bytes())
    later_machine = RuntimeIdentity(
        role="pc_a",
        repository_commit="c" * 40,
        repository_clean=True,
        hostname="replacement-pc-a",
        python_version="3.12.14",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )
    second = seal_bundle(
        _request(
            tmp_path,
            later_machine,
            (payload, relocated_artifact),
            root=tmp_path / "second-nas",
            created_at="2026-08-13T01:02:03Z",
        ),
        evidence_logger=FakeEvidence(),
    )
    assert second.content_id == first.content_id
    assert second.manifest["artifacts"][0]["root"] != first.manifest["artifacts"][0]["root"]
    assert second.ready["manifest_sha256"] != first.ready["manifest_sha256"]


def test_retry_cannot_readdress_same_report_content(
    tmp_path: Path,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    request = _request(
        tmp_path,
        pc_b,
        sources,
        kind="report",
        consumer_role="pc_a",
        permission="report_only",
    )
    seal_bundle(request, evidence_logger=FakeEvidence())
    with pytest.raises(BundleValidationError, match="consumer"):
        seal_bundle(
            _request(
                tmp_path,
                pc_b,
                sources,
                kind="report",
                consumer_role="notion",
                permission="report_only",
            ),
            evidence_logger=FakeEvidence(),
        )


def test_interrupted_partial_staging_does_not_block_seal_retry(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    kind_root = tmp_path / "nas" / "dataset_release"
    interrupted = kind_root / f".{('f' * 64)}.{('0' * 32)}.partial"
    interrupted.mkdir(parents=True)
    (interrupted / "incomplete").write_bytes(b"partial")
    bundle, _ = _seal(tmp_path, pc_a, sources)
    assert bundle.path.is_dir()
    assert interrupted.is_dir()


def test_content_change_gets_new_bundle_id(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    first, logger = _seal(tmp_path, pc_a, sources)
    payload, _ = sources
    (payload / "lineage.json").write_bytes(b'{"episodes":35}')
    second = seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=logger)
    assert first.bundle_id != second.bundle_id
    assert first.path.exists() and second.path.exists()


@pytest.mark.parametrize(
    ("kind", "producer_role", "consumer_role", "permission", "lineage"),
    [
        ("dataset_release", "pc_a", "pc_b", "data_only", {}),
        ("policy_candidate", "pc_b", "pc_a", "disconnected_only", {}),
        ("shadow_evidence", "pc_a", "pc_b", "evidence_only", {}),
        ("session_inputs", "pc_a", "pc_b", "planning_only", {}),
        (
            "rollout_session",
            "pc_b",
            "pc_a",
            "blocked",
            {"blockers": ["typed live-session producer required"]},
        ),
        ("rollout_evidence", "pc_a", "pc_b", "evidence_only", {}),
        ("report", "pc_b", "notion", "report_only", {}),
    ],
)
def test_all_contract_kinds_and_directions(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
    kind: str,
    producer_role: str,
    consumer_role: str,
    permission: str,
    lineage: dict[str, Any],
) -> None:
    assert set(SUPPORTED_KINDS) == set(KIND_RULES)
    identity = pc_a if producer_role == "pc_a" else pc_b
    bundle = seal_bundle(
        _request(
            tmp_path,
            identity,
            sources,
            root=tmp_path / kind,
            kind=kind,
            consumer_role=consumer_role,
            permission=permission,
            lineage=lineage,
        ),
        evidence_logger=FakeEvidence(),
    )
    assert bundle.kind == kind
    assert bundle.permission == permission
    assert bundle.manifest["consumer"]["role"] == consumer_role


def test_live_session_rejects_any_blocker(
    tmp_path: Path,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    with pytest.raises(BundleValidationError, match="empty blocker"):
        seal_bundle(
            _request(
                tmp_path,
                pc_b,
                sources,
                kind="rollout_session",
                permission="live_session",
                lineage={"blockers": ["executor attestation missing"]},
            ),
            evidence_logger=FakeEvidence(),
        )


def _validated_rollout_request(tmp_path: Path, pc_b: RuntimeIdentity) -> SealRequest:
    payload_root = tmp_path / "rollout-payload"
    payload_root.mkdir()
    hashes = {
        name: character * 64 for name, character in zip("abcdefghijk", "123456789ab", strict=True)
    }
    conditions = [
        {
            "condition_id": f"nominal_{index:02d}",
            "stratum": "nominal",
            "blue_axis": None,
            "blue_offset_mm": 0.0,
            "red_axis": None,
            "red_offset_mm": 0.0,
        }
        for index in range(1, 7)
    ] + [
        {
            "condition_id": f"robustness_{index:02d}",
            "stratum": "robustness",
            "blue_axis": "pad_x",
            "blue_offset_mm": blue,
            "red_axis": "pad_y",
            "red_offset_mm": red,
        }
        for index, (blue, red) in enumerate(
            ((-25.0, -25.0), (-25.0, 25.0), (25.0, -25.0), (25.0, 25.0)),
            start=1,
        )
    ]
    order = [condition["condition_id"] for condition in conditions]
    session_inputs = {
        "bundle_id": hashes["a"],
        "content_id": hashes["a"],
        "manifest_sha256": hashes["b"],
        "payload_sha256": hashes["c"],
        "setup_record_inventory_sha256": hashes["d"],
    }
    source = {
        "sha256": hashes["e"],
        "created_at_utc": "2026-08-12T12:00:00Z",
        "benchmark_lineage_sha256": hashes["f"],
        "policy_bindings_sha256": hashes["g"],
        "policy_binding_sha256": hashes["h"],
        "physical_setup_binding_sha256": hashes["i"],
        "executor_binding_sha256": hashes["j"],
    }
    payload = {
        "schema_version": 1,
        "session_id": "act-session-001",
        "policy_bundle_id": hashes["k"],
        "policy_content_id": hashes["k"],
        "setup_hashes": {
            "calibration": hashes["a"],
            "camera": hashes["b"],
            "robot": hashes["c"],
            "reset": hashes["d"],
        },
        "phase_permissions": ["hold", "shakedown", "scored"],
        "blockers": [],
        "operator": "operator@example.org",
        "task": (
            "Move the blue cube, then the red cube, from the white pad on the right "
            "to the gray platform on the left."
        ),
        "session_inputs_binding": session_inputs,
        "source_session": source,
        "executor": {
            "repository": "starai-viola-lerobot-ops",
            "repository_commit": "b" * 40,
            "entrypoint_sha256": hashes["e"],
            "attestation_sha256": hashes["f"],
            "reviewed_at_utc": "2026-08-12T11:00:00Z",
        },
        "estop": {
            "operator": "operator@example.org",
            "tested_at_utc": "2026-08-12T11:30:00Z",
            "attestation_sha256": hashes["g"],
        },
        "act_infrastructure_clearance": None,
        "trial_protocol": {
            "seed": 1000,
            "hold_required": True,
            "shakedown_trials": 2,
            "shakedown_speed_scale": 0.25,
            "scored_trials": 10,
            "nominal_trials": 6,
            "perturbation_trials": 4,
            "trial_duration_s": 60,
            "stable_success_s": 3.0,
            "target_hz": 30.0,
            "action_dimensions": 7,
            "replan_actions": 10,
            "ordered_conditions": conditions,
            "schedule_order": order,
            "execution_order": order,
        },
    }
    (payload_root / "rollout_session.json").write_bytes(canonical_json_bytes(payload))
    lineage = {
        "session_id": payload["session_id"],
        "policy": "act",
        "policy_bundle_id": hashes["k"],
        "policy_content_id": hashes["k"],
        "session_manifest_sha256": hashes["e"],
        "source_session_sha256": hashes["e"],
        "session_inputs_bundle_id": hashes["a"],
        "session_inputs_content_id": hashes["a"],
        "session_inputs_manifest_sha256": hashes["b"],
        "session_inputs_payload_sha256": hashes["c"],
        "session_inputs_setup_record_inventory_sha256": hashes["d"],
        "benchmark_lineage_sha256": hashes["f"],
        "policy_bindings_sha256": hashes["g"],
        "policy_binding_sha256": hashes["h"],
        "physical_setup_binding_sha256": hashes["i"],
        "executor_binding_sha256": hashes["j"],
        "executor_attestation_sha256": hashes["f"],
        "estop_attestation_sha256": hashes["g"],
        "blockers": [],
        "act_infrastructure_clearance": None,
    }
    return SealRequest(
        root=tmp_path / "nas",
        kind="rollout_session",
        experiment="viola-benchmark-v1",
        subject="act-session-001",
        producer=pc_b,
        lineage=lineage,
        wandb_project="viola-handoffs",
        payload_dir=payload_root,
        permission="live_session",
        created_at="2026-08-12T12:00:00Z",
    )


def test_generic_sealer_cannot_grant_live_session(tmp_path: Path, pc_b: RuntimeIdentity) -> None:
    request = _validated_rollout_request(tmp_path, pc_b)
    with pytest.raises(BundleValidationError, match="generic sealing cannot grant"):
        seal_bundle(request, evidence_logger=FakeEvidence())


def test_typed_sealer_requires_and_seals_exact_live_session(
    tmp_path: Path, pc_b: RuntimeIdentity
) -> None:
    request = _validated_rollout_request(tmp_path, pc_b)
    bundle = seal_validated_rollout_session_bundle(request, evidence_logger=FakeEvidence())
    assert bundle.kind == "rollout_session"
    assert bundle.permission == "live_session"

    mutated = json.loads((Path(request.payload_dir) / "rollout_session.json").read_text())
    mutated["session_inputs_binding"]["payload_sha256"] = "f" * 64
    (Path(request.payload_dir) / "rollout_session.json").write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(BundleValidationError, match="session_inputs_payload_sha256"):
        seal_validated_rollout_session_bundle(request, evidence_logger=FakeEvidence())


def test_wrong_direction_is_rejected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    with pytest.raises(BundleValidationError, match="must be produced"):
        seal_bundle(
            _request(tmp_path, pc_a, sources, kind="policy_candidate"),
            evidence_logger=FakeEvidence(),
        )


def test_runtime_capture_requires_exact_wandb_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("viola_handoff.contract.sys.version_info", (3, 12, 13))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "lerobot")

    def version(name: str) -> str:
        return {"lerobot": "0.6.1", "wandb": "0.28.0"}[name]

    monkeypatch.setattr("viola_handoff.contract.importlib.metadata.version", version)
    with pytest.raises(EnvironmentValidationError, match=r"W&B 0\.27\.2"):
        RuntimeIdentity.capture(role="pc_a", repo_root=tmp_path)


def test_seal_requires_online_evidence_before_ready(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    with pytest.raises(EvidenceError, match="simulated"):
        seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=FakeEvidence(fail=True))

    kind_root = tmp_path / "nas" / "dataset_release"
    promoted = [path for path in kind_root.iterdir() if not path.name.endswith(".partial")]
    assert len(promoted) == 1
    assert not (promoted[0] / "READY.json").exists()
    with pytest.raises(BundleValidationError):
        inspect_bundle(promoted[0])

    logger = FakeEvidence()
    recovered = seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=logger)
    assert recovered.path == promoted[0]
    assert (recovered.path / "READY.json").is_file()
    assert [event["event"] for event in logger.events] == ["sealed"]


def test_seal_requires_https_wandb_url_before_ready(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    with pytest.raises(EvidenceError, match="HTTPS run URL"):
        seal_bundle(
            _request(tmp_path, pc_a, sources),
            evidence_logger=FakeEvidence(url="", fixed_url=True),
        )
    promoted = list((tmp_path / "nas" / "dataset_release").glob("[!.]*"))
    assert len(promoted) == 1
    assert not (promoted[0] / "READY.json").exists()


def test_wandb_evidence_requires_remote_finished_state() -> None:
    class RemoteApi:
        def run(self, path: str) -> Any:
            assert path == "entity/project/ho-test"
            return type("RemoteRun", (), {"state": "finished"})()

    class FakeWandb:
        @staticmethod
        def Api(*, timeout: int) -> RemoteApi:
            assert timeout == 20
            return RemoteApi()

    local = type("LocalRun", (), {"path": ["entity", "project", "ho-test"]})()
    _require_remote_finished(FakeWandb(), local, run_id="ho-test")


def test_wandb_evidence_rejects_remote_failed_state() -> None:
    class RemoteApi:
        def run(self, path: str) -> Any:
            del path
            return type("RemoteRun", (), {"state": "failed"})()

    class FakeWandb:
        @staticmethod
        def Api(*, timeout: int) -> RemoteApi:
            del timeout
            return RemoteApi()

    local = type("LocalRun", (), {"path": ["entity", "project", "ho-test"]})()
    with pytest.raises(EvidenceError, match="ended as 'failed'"):
        _require_remote_finished(FakeWandb(), local, run_id="ho-test")


def test_default_writer_refuses_offline_wandb(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(EvidenceError, match="online"):
        seal_bundle(_request(tmp_path, pc_a, sources))


def test_payload_and_artifact_tampering_are_detected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    bundle.payload_file("lineage.json").write_bytes(b"tampered")
    with pytest.raises(BundleValidationError, match="payload inventory"):
        inspect_bundle(bundle.path)

    # Restore by sealing into another root, then mutate the external artifact.
    clean = seal_bundle(
        _request(tmp_path, pc_a, sources, root=tmp_path / "other-nas"),
        evidence_logger=FakeEvidence(),
    )
    clean.artifact_root("dataset").joinpath("checkpoint/config.json").write_bytes(b'{"bad":true}')
    with pytest.raises(BundleValidationError, match="artifact inventory"):
        inspect_bundle(clean.path)


def test_extra_and_missing_payload_files_are_detected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    (bundle.path / "payload" / "extra.txt").write_bytes(b"extra")
    with pytest.raises(BundleValidationError, match="payload inventory"):
        inspect_bundle(bundle.path)

    other = seal_bundle(
        _request(tmp_path, pc_a, sources, root=tmp_path / "other-nas"),
        evidence_logger=FakeEvidence(),
    )
    other.payload_file("nested/evidence.txt").unlink()
    with pytest.raises(BundleValidationError, match="payload inventory"):
        inspect_bundle(other.path)


def test_extra_top_level_file_is_detected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    (bundle.path / "undeclared.bin").write_bytes(b"no")
    with pytest.raises(BundleValidationError, match="extra"):
        inspect_bundle(bundle.path)


def test_symlinks_are_rejected_in_sources_and_bundles(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    payload, _ = sources
    (payload / "escape").symlink_to("/etc/passwd")
    with pytest.raises(BundleValidationError, match="symlink"):
        seal_bundle(_request(tmp_path, pc_a, sources), evidence_logger=FakeEvidence())
    (payload / "escape").unlink()

    bundle, _ = _seal(tmp_path, pc_a, sources)
    victim = bundle.payload_file("lineage.json")
    victim.unlink()
    victim.symlink_to("/etc/passwd")
    with pytest.raises(BundleValidationError, match="symlink"):
        inspect_bundle(bundle.path)


def test_manifest_path_traversal_is_rejected_even_if_json_is_canonical(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["payload"]["files"][0]["path"] = "../escape"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(BundleValidationError, match="unsafe inventory path"):
        inspect_bundle(bundle.path)


def test_bundle_must_stay_under_its_kind_directory(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    wrong_parent = tmp_path / "wrong_kind"
    wrong_parent.mkdir()
    moved = wrong_parent / bundle.bundle_id
    bundle.path.rename(moved)
    with pytest.raises(BundleValidationError, match="parent directory"):
        inspect_bundle(moved)


def test_noncanonical_or_duplicate_json_is_rejected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    manifest_path = bundle.path / "manifest.json"
    value = json.loads(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="not canonical"):
        inspect_bundle(bundle.path)

    other = seal_bundle(
        _request(tmp_path, pc_a, sources, root=tmp_path / "other-nas"),
        evidence_logger=FakeEvidence(),
    )
    other.path.joinpath("READY.json").write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(BundleValidationError, match="duplicate JSON key"):
        inspect_bundle(other.path)


def test_ready_tampering_is_detected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    ready_path = bundle.path / "READY.json"
    ready = json.loads(ready_path.read_bytes())
    ready["manifest_sha256"] = "0" * 64
    ready_path.write_bytes(canonical_json_bytes(ready))
    with pytest.raises(BundleValidationError, match="manifest digest"):
        inspect_bundle(bundle.path)


@pytest.mark.parametrize(
    "wandb_url",
    (
        "https://wandb.ai/tester/foreign-project/runs/ho-deadbeef",
        "https://wandb.ai/tester/viola-handoffs/runs/foreign-run",
        "https://wandb.ai/tester/viola-handoffs/runs/foreign-run?view=1",
    ),
)
def test_ready_wandb_url_must_match_manifest_identity(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
    wandb_url: str,
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    ready_path = bundle.path / "READY.json"
    ready = json.loads(ready_path.read_bytes())
    ready["wandb_url"] = wandb_url
    ready_path.write_bytes(canonical_json_bytes(ready))
    with pytest.raises(BundleValidationError, match="manifest W&B project/run"):
        inspect_bundle(bundle.path)


def test_accept_checksum_copies_atomically_and_receipt_is_idempotent(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, seal_logger = _seal(tmp_path, pc_a, sources)
    accept_logger = FakeEvidence()
    first = accept_bundle(
        bundle.path,
        tmp_path / "accepted",
        receiver=pc_b,
        evidence_logger=accept_logger,
        note="full decode under LeRobot 0.6.1",
    )
    second = accept_bundle(
        bundle.path,
        tmp_path / "accepted",
        receiver=pc_b,
        evidence_logger=accept_logger,
        note="full decode under LeRobot 0.6.1",
    )

    assert first.path == second.path
    assert first.path != bundle.path
    assert first.manifest == bundle.manifest
    assert first.payload_file("lineage.json").read_bytes() == b'{"episodes":34}'
    local_artifact = first.accepted_artifact_root("dataset")
    assert local_artifact == tmp_path / "accepted" / "artifacts" / bundle.bundle_id / "dataset"
    assert local_artifact.joinpath("checkpoint/model.safetensors").read_bytes() == b"model bytes"
    assert local_artifact != bundle.artifact_root("dataset")
    assert [event["event"] for event in accept_logger.events] == ["accepted"]
    assert accept_logger.events[0]["metadata"]["accepted_artifact_count"] == 1
    assert len(list((bundle.path / "receipts").glob("*.json"))) == 1
    assert len(list((first.path / "receipts").glob("*.json"))) == 1
    receipt = first.receipts[0]
    assert receipt["accepted_artifacts"]["dataset"] == {
        "root": str(local_artifact),
        "inventory_sha256": first.manifest["artifacts"][0]["inventory_sha256"],
        "file_count": 2,
        "byte_count": 13,
    }
    assert len(seal_logger.events) == 1


def test_accept_does_not_promote_when_online_evidence_fails(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    destination = tmp_path / "accepted"
    with pytest.raises(EvidenceError, match="simulated"):
        accept_bundle(
            bundle.path,
            destination,
            receiver=pc_b,
            evidence_logger=FakeEvidence(fail=True),
        )
    kind_root = destination / "dataset_release"
    assert not (kind_root / bundle.bundle_id).exists()
    artifact_root = destination / "artifacts"
    assert not (artifact_root / bundle.bundle_id).exists()
    leftovers = list(kind_root.glob("*.partial"))
    assert leftovers
    assert all(path.name.startswith(".") for path in leftovers)
    artifact_leftovers = list(artifact_root.glob("*.partial"))
    assert artifact_leftovers
    assert all(path.name.startswith(".") for path in artifact_leftovers)
    assert list((bundle.path / "receipts").iterdir()) == []

    accepted = accept_bundle(
        bundle.path,
        destination,
        receiver=pc_b,
        evidence_logger=FakeEvidence(),
    )
    assert accepted.path == kind_root / bundle.bundle_id
    assert leftovers[0].is_dir()
    assert accepted.accepted_artifact_root("dataset").is_dir()


def test_source_bundle_cannot_supply_receiver_local_artifact(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    # A stray partial/local-looking directory is not mistaken for an accepted
    # receiver mapping and does not affect signed-source inspection.
    stray = bundle.path.parent.parent / "artifacts" / bundle.bundle_id
    stray.parent.mkdir()
    (stray / "wrong").mkdir(parents=True)
    assert inspect_bundle(bundle.path, verify_artifacts=False).content_id == bundle.content_id
    with pytest.raises(BundleValidationError, match="artifact names differ"):
        inspect_bundle(bundle.path, verify_artifacts=True)
    with pytest.raises(BundleValidationError, match="run viola-handoff accept"):
        bundle.accepted_artifact_root("dataset")


def test_accepted_artifact_tamper_and_extra_entries_fail_closed(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    accepted = accept_bundle(
        bundle.path,
        tmp_path / "accepted",
        receiver=pc_b,
        evidence_logger=FakeEvidence(),
    )
    local = accepted.accepted_artifact_root("dataset")
    local.joinpath("checkpoint/model.safetensors").write_bytes(b"tampered")
    with pytest.raises(BundleValidationError, match="differs"):
        accepted.accepted_artifact_root("dataset")

    # A retry checks the existing receiver-local copy and never overwrites it.
    with pytest.raises(BundleValidationError, match="inventory differs"):
        accept_bundle(
            bundle.path,
            tmp_path / "accepted",
            receiver=pc_b,
            evidence_logger=FakeEvidence(),
        )


def test_receiver_local_artifact_symlink_is_rejected(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    accepted = accept_bundle(
        bundle.path,
        tmp_path / "accepted",
        receiver=pc_b,
        evidence_logger=FakeEvidence(),
    )
    local = accepted.accepted_artifact_root("dataset")
    victim = local / "checkpoint/config.json"
    victim.unlink()
    victim.symlink_to("/etc/passwd")
    with pytest.raises(BundleValidationError, match="symlink"):
        accepted.accepted_artifact_root("dataset")


def test_accept_retries_existing_verified_artifact_without_copying_source(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    destination = tmp_path / "accepted"
    first = accept_bundle(
        bundle.path,
        destination,
        receiver=pc_b,
        evidence_logger=FakeEvidence(),
    )
    source_artifact = bundle.artifact_root("dataset")
    source_artifact.rename(tmp_path / "source-now-offline")
    # The source bundle can no longer be fully inspected, by design. The
    # receiver's accepted object remains independently checksum-verifiable.
    reloaded = inspect_bundle(first.path, verify_artifacts=True)
    assert reloaded.accepted_artifact_root("dataset").is_dir()


def test_repeated_accept_is_idempotent_when_source_artifact_disappears(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    destination = tmp_path / "accepted"
    logger = FakeEvidence()
    first = accept_bundle(
        bundle.path,
        destination,
        receiver=pc_b,
        evidence_logger=logger,
        note="verified",
    )
    bundle.artifact_root("dataset").rename(tmp_path / "source-offline")

    same_target = accept_bundle(
        bundle.path,
        destination,
        receiver=pc_b,
        evidence_logger=logger,
        note="verified",
    )
    assert same_target.path == first.path

    # A repeated command may point at the already accepted bundle; it neither
    # needs nor falls back to the vanished signed source root.
    second = accept_bundle(
        first.path,
        tmp_path / "accepted-again",
        receiver=pc_b,
        evidence_logger=logger,
        note="verified",
    )
    assert (
        second.accepted_artifact_root("dataset")
        .joinpath("checkpoint/model.safetensors")
        .read_bytes()
        == b"model bytes"
    )


def test_accept_rejects_wrong_receiver_and_symlink_destination(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    with pytest.raises(BundleValidationError, match="addressed"):
        accept_bundle(
            bundle.path,
            tmp_path / "accepted",
            receiver=pc_a,
            evidence_logger=FakeEvidence(),
        )
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    pc_b = RuntimeIdentity("pc_b", "b" * 40, True, "b", "3.12.13", "0.6.1", "lerobot")
    with pytest.raises(BundleValidationError, match="symlink"):
        accept_bundle(bundle.path, link, receiver=pc_b, evidence_logger=FakeEvidence())


def test_ack_is_append_only_retry_safe_and_tamper_evident(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    logger = FakeEvidence()
    first = ack_bundle(
        bundle.path,
        status="rejected",
        actor=pc_b,
        note="full decode failed",
        evidence_logger=logger,
    )
    original = first.read_bytes()
    second = ack_bundle(
        bundle.path,
        status="rejected",
        actor=pc_b,
        note="full decode failed",
        evidence_logger=logger,
    )
    assert first == second
    assert second.read_bytes() == original
    assert len(list((bundle.path / "receipts").glob("*.json"))) == 1
    assert len(logger.events) == 1
    with pytest.raises(BundleValidationError, match="inactive"):
        inspect_bundle(bundle.path)
    assert inspect_bundle(bundle.path, allow_inactive=True).is_active is False

    receipt = json.loads(first.read_bytes())
    receipt["note"] = "silently changed"
    first.write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(BundleValidationError, match="receipt key"):
        inspect_bundle(bundle.path)


def test_receipt_revocation_is_terminal_and_role_constrained(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    with pytest.raises(BundleValidationError, match="only by accept_bundle"):
        ack_bundle(
            bundle.path,
            status="accepted",
            actor=pc_a,
            evidence_logger=FakeEvidence(),
        )
    assert list((bundle.path / "receipts").iterdir()) == []

    revoked = ack_bundle(
        bundle.path,
        status="revoked",
        actor=pc_a,
        note="producer withdrew this release",
        evidence_logger=FakeEvidence(),
    )
    assert revoked.is_file()
    diagnostic = inspect_bundle(bundle.path, allow_inactive=True)
    assert [receipt["status"] for receipt in diagnostic.receipts] == ["revoked"]
    with pytest.raises(BundleValidationError, match="revoked"):
        inspect_bundle(bundle.path)
    with pytest.raises(BundleValidationError, match="inactive"):
        accept_bundle(
            bundle.path,
            tmp_path / "accepted",
            receiver=pc_b,
            evidence_logger=FakeEvidence(),
        )


def test_accepted_snapshot_requires_current_active_canonical_source(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    source, _ = _seal(tmp_path, pc_a, sources)
    accepted = accept_bundle(
        source.path,
        tmp_path / "accepted",
        receiver=pc_b,
        evidence_logger=FakeEvidence(),
    )
    assert (
        require_active_canonical_source(
            accepted,
            handoff_root=source.path.parent.parent,
        )
        is accepted
    )

    ack_bundle(
        source.path,
        status="revoked",
        actor=pc_a,
        note="producer withdrew the grant after acceptance",
        evidence_logger=FakeEvidence(),
    )

    # The immutable local snapshot remains useful for offline diagnostics, but
    # it cannot authorize a safety-sensitive action after the source changed.
    assert inspect_bundle(accepted.path, verify_artifacts=True).is_active
    with pytest.raises(BundleValidationError, match=r"canonical handoff source.*revoked"):
        require_active_canonical_source(
            accepted,
            handoff_root=source.path.parent.parent,
        )


def test_accepted_snapshot_freshness_fails_when_canonical_source_is_unavailable(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    source, _ = _seal(tmp_path, pc_a, sources)
    accepted = accept_bundle(
        source.path,
        tmp_path / "accepted",
        receiver=pc_b,
        evidence_logger=FakeEvidence(),
    )
    unavailable = tmp_path / "canonical-source-unavailable"
    source.path.rename(unavailable)

    assert inspect_bundle(accepted.path, verify_artifacts=True).is_active
    with pytest.raises(BundleValidationError, match="canonical handoff source is unavailable"):
        require_active_canonical_source(
            accepted,
            handoff_root=source.path.parent.parent,
        )


def test_ack_requires_online_evidence_before_receipt(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    pc_b: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    with pytest.raises(EvidenceError, match="simulated"):
        ack_bundle(
            bundle.path,
            status="rejected",
            actor=pc_b,
            evidence_logger=FakeEvidence(fail=True),
        )
    assert list((bundle.path / "receipts").iterdir()) == []


def test_verified_paths_require_declaration_and_full_artifact_verification(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    metadata_only = inspect_bundle(bundle.path, verify_artifacts=False)
    with pytest.raises(BundleValidationError, match="fully verified"):
        metadata_only.artifact_root("dataset")
    with pytest.raises(BundleValidationError, match="not declared"):
        metadata_only.payload_file("not-present")
    with pytest.raises(BundleValidationError, match="unsafe payload path"):
        metadata_only.payload_file("../escape")


def test_expected_kind_and_permission_are_enforced(
    tmp_path: Path,
    pc_a: RuntimeIdentity,
    sources: tuple[Path, Path],
) -> None:
    bundle, _ = _seal(tmp_path, pc_a, sources)
    with pytest.raises(BundleValidationError, match="expected bundle kind"):
        inspect_bundle(bundle.path, expected_kind="policy_candidate")
    with pytest.raises(BundleValidationError, match="expected permission"):
        inspect_bundle(bundle.path, required_permission="disconnected_only")
