from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest

import viola_handoff
import viola_ops.session_inputs as session_inputs_ops
from viola_ops.errors import ValidationError
from viola_ops.jsonutil import sha256_file, sha256_json
from viola_ops.schemas import EXECUTOR_CAPABILITIES, VIOLA_JOINTS
from viola_ops.session_inputs import load_reviewed_setup, seal_session_inputs


@dataclass
class FakeEvidence:
    def __post_init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, **event: Any) -> str:
        self.events.append(event)
        return f"https://wandb.ai/tester/{event['project']}/runs/{event['run_id']}"


@dataclass
class MutatingEvidence:
    mutation: Callable[[], None]

    def record(self, **event: Any) -> str:
        self.mutation()
        return f"https://wandb.ai/tester/{event['project']}/runs/{event['run_id']}"


def _identity() -> viola_handoff.RuntimeIdentity:
    return viola_handoff.RuntimeIdentity(
        role="pc_a",
        repository_commit="a" * 40,
        repository_clean=True,
        hostname="pc-a-test",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _repo_root(tmp_path: Path) -> Path:
    repository = tmp_path / "repo-a"
    repository.mkdir()
    return repository


def _replace_tree_with_same_bytes(path: Path) -> None:
    old = path.with_name(f"{path.name}-old")
    path.rename(old)
    shutil.copytree(old, path)


def _replace_file_with_same_bytes(path: Path) -> None:
    replacement = path.with_name(f"{path.name}.replacement")
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)


def _write_setup(tmp_path: Path, *, tested_at: datetime | None = None) -> Path:
    source = tmp_path / "reviewed"
    source.mkdir(parents=True)
    calibration = source / "calibration.json"
    reset = source / "reset_protocol.json"
    entrypoint = source / "reviewed_executor.py"
    calibration.write_text('{"calibration":"reviewed-test-fixture"}\n', encoding="utf-8")
    reset.write_text('{"description":"manual reset"}\n', encoding="utf-8")
    entrypoint.write_text("# reviewed public-API executor fixture\n", encoding="utf-8")

    cameras = {
        "front": {
            "type": "opencv",
            "index_or_path": "/dev/v4l/by-id/viola-front-video-index0",
            "width": 640,
            "height": 480,
            "fps": 30,
        },
        "up": {
            "type": "opencv",
            "index_or_path": "/dev/v4l/by-id/viola-up-video-index0",
            "width": 640,
            "height": 480,
            "fps": 30,
        },
    }
    joint_limits = {
        joint: ([0.0, 100.0] if joint == "gripper" else [-100.0, 100.0])
        for joint in VIOLA_JOINTS
    }
    max_step_deltas = {joint: 1.0 for joint in VIOLA_JOINTS}
    robot = {
        "robot_port": "/dev/viola-test-fixture",
        "joint_limits": joint_limits,
        "max_step_deltas": max_step_deltas,
        "speed_scale": 1.0,
    }
    timestamp = (tested_at or datetime.now(UTC)).replace(microsecond=0).isoformat()
    setup = {
        "schema_version": 1,
        "setup_id": "cross-repo-setup-v1",
        **robot,
        "cameras": cameras,
        "calibration_path": str(calibration),
        "calibration_sha256": sha256_file(calibration),
        "reset_protocol_path": str(reset),
        "camera_config_sha256": sha256_json(cameras),
        "robot_config_sha256": sha256_json(robot),
        "reset_protocol_sha256": sha256_file(reset),
        "executor_attestation": {
            "reviewed": True,
            "clean_commit": True,
            "commit": "a" * 40,
            "reviewer": "interop-reviewer",
            "reviewed_at": timestamp,
            "repository": "starai-viola-lerobot-ops",
            "python_version": "3.12.13",
            "lerobot_version": "0.6.1",
            "conda_environment": "lerobot",
            "execution_backend": "direct_lerobot_fashionstar",
            "entrypoint_path": str(entrypoint),
            "entrypoint_sha256": sha256_file(entrypoint),
            "capabilities": list(EXECUTOR_CAPABILITIES),
        },
        "estop": {"tested_at": timestamp, "operator": "interop-operator", "passed": True},
    }
    path = source / "hardware_setup.json"
    path.write_text(json.dumps(setup, indent=2), encoding="utf-8")
    return path


def test_session_inputs_are_readable_and_repo_b_compatible(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    evidence = FakeEvidence()
    receiver = tmp_path / "receiver-only"

    result = seal_session_inputs(
        setup_path,
        experiment="cross-repo-interop",
        subject="cross-repo-setup-v1",
        handoff_root=tmp_path / "handoffs",
        material_root=tmp_path / "producer-materials",
        destination_root=receiver,
        wandb_project="viola-interop-test",
        repo_root=_repo_root(tmp_path),
        producer_identity=_identity(),
        identity_capture=_identity,
        evidence_logger=evidence,
    )

    assert result.bundle.kind == "session_inputs"
    assert result.bundle.permission == "planning_only"
    assert not receiver.exists(), "Repo A must not self-accept on Repo B's behalf"
    assert sorted(path.name for path in result.payload_root.iterdir()) == [
        "hardware_setup.json",
        "session_inputs.json",
    ]
    payload = json.loads((result.payload_root / "session_inputs.json").read_text())
    hardware = json.loads((result.payload_root / "hardware_setup.json").read_text())
    assert payload["executor"]["capabilities"] == list(EXECUTOR_CAPABILITIES)
    assert hardware["executor_attestation"]["capabilities"] == list(EXECUTOR_CAPABILITIES)
    assert payload["setup_hashes"] == {
        "calibration": sha256_file(result.setup_record / "calibration.json"),
        "camera": sha256_json(hardware["cameras"]),
        "robot": sha256_json(
            {
                "robot_port": hardware["robot_port"],
                "joint_limits": hardware["joint_limits"],
                "max_step_deltas": hardware["max_step_deltas"],
                "speed_scale": hardware["speed_scale"],
            }
        ),
        "reset": sha256_file(result.setup_record / "reset_protocol.json"),
    }
    assert evidence.events[0]["event"] == "sealed"


def test_cross_repo_probe_can_replace_capture_and_seal_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_path = _write_setup(tmp_path)
    identity = _identity()
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: identity),
    )
    monkeypatch.setenv("WANDB_MODE", "offline")
    real_seal = viola_handoff.seal_bundle
    monkeypatch.setattr(
        viola_handoff,
        "seal_bundle",
        lambda request: real_seal(request, evidence_logger=FakeEvidence()),
    )

    result = seal_session_inputs(
        setup_path,
        experiment="cross-repo-interop",
        subject="cross-repo-setup-v1",
        handoff_root=tmp_path / "handoffs",
        material_root=tmp_path / "producer-materials",
        destination_root=tmp_path / "accepted",
        wandb_project="viola-interop-test",
        repo_root=_repo_root(tmp_path),
    )

    assert result.bundle.bundle_id == result.bundle.content_id


def test_stale_estop_and_unreviewed_limits_fail_before_any_bundle(tmp_path: Path) -> None:
    stale = datetime.now(UTC) - timedelta(hours=24)
    setup_path = _write_setup(tmp_path, tested_at=stale)

    with pytest.raises(ValidationError, match="less than 24 hours"):
        load_reviewed_setup(setup_path)

    assert not (tmp_path / "handoffs").exists()


def test_setup_rejects_hash_drift_and_capability_reordering(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    setup = json.loads(setup_path.read_text())
    setup["camera_config_sha256"] = "0" * 64
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    with pytest.raises(ValidationError, match="camera_config_sha256"):
        load_reviewed_setup(setup_path)

    setup_path = _write_setup(tmp_path / "second")
    setup = json.loads(setup_path.read_text())
    setup["executor_attestation"]["capabilities"].reverse()
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    with pytest.raises(ValidationError, match="capabilities"):
        load_reviewed_setup(setup_path)


def test_executor_capability_order_is_locked_to_repo_b_contract() -> None:
    assert EXECUTOR_CAPABILITIES == (
        "exact_camera_mapping",
        "fresh_position_write_limits",
        "keep_current_pose_startup",
        "no_automatic_reset_or_return",
        "operator_estop_ownership",
        "session_trial_recording",
        "ten_action_queue",
        "torque_retained_on_disconnect",
    )


@pytest.mark.parametrize("protected_root", ["material", "handoff"])
def test_session_inputs_require_external_producer_roots(
    tmp_path: Path,
    protected_root: str,
) -> None:
    setup_path = _write_setup(tmp_path)
    repository = _repo_root(tmp_path)
    material_root = tmp_path / "producer-materials"
    handoff_root = tmp_path / "handoffs"
    if protected_root == "material":
        material_root = repository / "producer-materials"
    else:
        handoff_root = repository / "handoffs"

    with pytest.raises(ValidationError, match="outside the Repo-A worktree"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=handoff_root,
            material_root=material_root,
            repo_root=repository,
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
        )

    assert not material_root.exists()
    assert not handoff_root.exists()


@pytest.mark.parametrize("protected_root", ["material", "handoff"])
def test_session_inputs_reject_symlinked_producer_roots(
    tmp_path: Path,
    protected_root: str,
) -> None:
    setup_path = _write_setup(tmp_path)
    repository = _repo_root(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "linked-external"
    linked.symlink_to(external, target_is_directory=True)
    material_root = tmp_path / "producer-materials"
    handoff_root = tmp_path / "handoffs"
    if protected_root == "material":
        material_root = linked / "producer-materials"
    else:
        handoff_root = linked / "handoffs"

    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=handoff_root,
            material_root=material_root,
            repo_root=repository,
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
        )


def test_session_inputs_reject_a_symlink_inside_the_material_path(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    material_root = tmp_path / "producer-materials"
    material_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (material_root / "cross-repo-setup-v1").symlink_to(
        elsewhere,
        target_is_directory=True,
    )

    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=tmp_path / "handoffs",
            material_root=material_root,
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
        )

    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"role": "pc_b"}, "pc_a"),
        ({"repository_clean": False}, "clean Repo-A"),
        ({"repository_commit": "A" * 40}, "full lowercase Git SHA"),
        ({"python_version": "3.11.9"}, "Python 3.12"),
        ({"lerobot_version": "0.5.0"}, "LeRobot 0.6.1"),
        ({"conda_environment": "base"}, "lerobot Conda"),
    ],
)
def test_session_inputs_validate_injected_identity_before_writing(
    tmp_path: Path,
    changes: dict[str, Any],
    message: str,
) -> None:
    setup_path = _write_setup(tmp_path)
    material_root = tmp_path / "producer-materials"

    with pytest.raises(ValidationError, match=message):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=tmp_path / "handoffs",
            material_root=material_root,
            repo_root=_repo_root(tmp_path),
            producer_identity=replace(_identity(), **changes),
            evidence_logger=FakeEvidence(),
        )

    assert not material_root.exists()


def test_session_inputs_recapture_the_clean_runtime_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_path = _write_setup(tmp_path)
    identities = iter([_identity(), replace(_identity(), repository_commit="b" * 40)])
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: next(identities)),
    )
    evidence = FakeEvidence()

    with pytest.raises(ValidationError, match="changed before session-input sealing"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=tmp_path / "handoffs",
            material_root=tmp_path / "producer-materials",
            repo_root=_repo_root(tmp_path),
            evidence_logger=evidence,
        )

    assert evidence.events == []
    assert not (tmp_path / "handoffs").exists()


def test_session_inputs_recheck_reviewed_sources_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_path = _write_setup(tmp_path)
    setup = json.loads(setup_path.read_text())
    calibration = Path(setup["calibration_path"])
    real_write = session_inputs_ops.write_canonical_json

    def write_then_change_source(path: str | Path, value: Any) -> Path:
        written = real_write(path, value)
        if Path(path).name == "session_inputs.json":
            calibration.write_text('{"calibration":"changed"}\n', encoding="utf-8")
        return written

    monkeypatch.setattr(
        session_inputs_ops,
        "write_canonical_json",
        write_then_change_source,
    )
    evidence = FakeEvidence()

    with pytest.raises(ValidationError, match="calibration_sha256"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=tmp_path / "handoffs",
            material_root=tmp_path / "producer-materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=evidence,
        )

    assert evidence.events == []
    assert not (tmp_path / "handoffs").exists()


def test_session_inputs_recheck_setup_record_inventory_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_path = _write_setup(tmp_path)
    material_root = tmp_path / "producer-materials"
    real_write = session_inputs_ops.write_canonical_json

    def write_then_change_copy(path: str | Path, value: Any) -> Path:
        written = real_write(path, value)
        if Path(path).name == "session_inputs.json":
            copied = next(material_root.rglob("setup_record/calibration.json"))
            os.chmod(copied, 0o644)
            copied.write_text('{"calibration":"changed-copy"}\n', encoding="utf-8")
        return written

    monkeypatch.setattr(
        session_inputs_ops,
        "write_canonical_json",
        write_then_change_copy,
    )
    evidence = FakeEvidence()

    with pytest.raises(ValidationError, match="setup-record inventory bytes changed"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=tmp_path / "handoffs",
            material_root=material_root,
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=evidence,
        )

    assert evidence.events == []
    assert not (tmp_path / "handoffs").exists()


@pytest.mark.parametrize("target", ["payload", "artifact", "reviewed_source"])
def test_session_inputs_reject_replacement_during_handoff_publication(
    tmp_path: Path,
    target: str,
) -> None:
    setup_path = _write_setup(tmp_path)
    setup = json.loads(setup_path.read_text())
    material_root = tmp_path / "producer-materials"
    handoff_root = tmp_path / "handoffs"

    def replace_material() -> None:
        if target == "reviewed_source":
            _replace_file_with_same_bytes(Path(setup["calibration_path"]))
            return
        name = "payload" if target == "payload" else "setup_record"
        _replace_tree_with_same_bytes(next(material_root.rglob(name)))

    with pytest.raises(ValidationError, match="changed during handoff W&B publication"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=handoff_root,
            material_root=material_root,
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            identity_capture=_identity,
            evidence_logger=MutatingEvidence(replace_material),
        )

    assert not list(handoff_root.rglob("READY.json"))


def test_session_inputs_reject_payload_replacement_between_caller_and_sealer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_path = _write_setup(tmp_path)
    handoff_root = tmp_path / "handoffs"
    real_seal = viola_handoff.seal_bundle

    def replace_then_seal(request: Any, **kwargs: Any) -> viola_handoff.VerifiedBundle:
        _replace_tree_with_same_bytes(Path(request.payload_dir))
        return real_seal(request, **kwargs)

    monkeypatch.setattr(viola_handoff, "seal_bundle", replace_then_seal)
    with pytest.raises(ValidationError, match="session-input payload changed before"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=handoff_root,
            material_root=tmp_path / "producer-materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            identity_capture=_identity,
            evidence_logger=FakeEvidence(),
        )

    assert not list(handoff_root.rglob("READY.json"))


def test_session_inputs_reject_checkout_dirtiness_during_handoff_publication(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path)
    handoff_root = tmp_path / "handoffs"
    dirty = False

    def capture_identity() -> viola_handoff.RuntimeIdentity:
        return replace(_identity(), repository_clean=False) if dirty else _identity()

    def dirty_checkout() -> None:
        nonlocal dirty
        dirty = True

    with pytest.raises(ValidationError, match="runtime identity changed"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=handoff_root,
            material_root=tmp_path / "producer-materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            identity_capture=capture_identity,
            evidence_logger=MutatingEvidence(dirty_checkout),
        )

    assert not list(handoff_root.rglob("READY.json"))


def test_session_inputs_recapture_even_with_an_injected_starting_identity(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path)
    evidence = FakeEvidence()

    with pytest.raises(ValidationError, match="runtime identity changed"):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=tmp_path / "handoffs",
            material_root=tmp_path / "producer-materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            identity_capture=lambda: replace(_identity(), repository_commit="b" * 40),
            evidence_logger=evidence,
        )

    assert evidence.events == []
    assert not list((tmp_path / "handoffs").rglob("READY.json"))


def test_session_inputs_reject_an_existing_ready_inventory_mismatch(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path)
    handoff_root = tmp_path / "handoffs"
    repository = _repo_root(tmp_path)
    first = seal_session_inputs(
        setup_path,
        experiment="test",
        subject="cross-repo-setup-v1",
        handoff_root=handoff_root,
        material_root=tmp_path / "producer-materials",
        repo_root=repository,
        producer_identity=_identity(),
        identity_capture=_identity,
        evidence_logger=FakeEvidence(),
    )
    ready_path = first.bundle.path / "READY.json"
    ready = json.loads(ready_path.read_text())
    ready["inventory_sha256"] = "0" * 64
    os.chmod(ready_path, 0o644)
    ready_path.write_bytes(viola_handoff.canonical_json_bytes(ready))

    with pytest.raises(viola_handoff.BundleValidationError):
        seal_session_inputs(
            setup_path,
            experiment="test",
            subject="cross-repo-setup-v1",
            handoff_root=handoff_root,
            material_root=tmp_path / "producer-materials",
            repo_root=repository,
            producer_identity=_identity(),
            identity_capture=_identity,
            evidence_logger=FakeEvidence(),
        )
