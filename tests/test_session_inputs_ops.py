from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import viola_handoff
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
        repo_root=tmp_path,
        producer_identity=_identity(),
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
        repo_root=tmp_path,
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
