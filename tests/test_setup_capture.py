from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import viola_handoff
from viola_ops.errors import SafetyGateError
from viola_ops.jsonutil import sha256_file, sha256_json
from viola_ops.schemas import EXECUTOR_CAPABILITIES, VIOLA_JOINTS
from viola_ops.setup import capture_frozen_state


class _ReadOnlyPort:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.monitor_calls = 0

    def openPort(self):
        self.opened = True
        return True

    def closePort(self):
        self.closed = True

    def ping(self, _servo_id):
        return True

    def SyncServoMonitor(self, motors, realtime=False):
        assert realtime is True
        self.monitor_calls += 1
        return {joint: SimpleNamespace(current_position=0.0) for joint in motors}


def _setup(tmp_path: Path) -> Path:
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                joint: {
                    "id": index,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": 0,
                    "range_max": 4096,
                }
                for index, joint in enumerate(VIOLA_JOINTS)
            }
        ),
        encoding="utf-8",
    )
    reset = tmp_path / "reset.json"
    reset.write_text('{"manual":true}\n', encoding="utf-8")
    executor = tmp_path / "execution.py"
    executor.write_text("# reviewed\n", encoding="utf-8")
    cameras = {
        name: {
            "type": "opencv",
            "index_or_path": f"/dev/{name}",
            "width": 640,
            "height": 480,
            "fps": 30,
        }
        for name in ("front", "up")
    }
    robot = {
        "robot_port": "/dev/robot",
        "joint_limits": {
            joint: ([0.0, 100.0] if joint == "gripper" else [-100.0, 100.0])
            for joint in VIOLA_JOINTS
        },
        "max_step_deltas": {joint: 1.0 for joint in VIOLA_JOINTS},
        "speed_scale": 1.0,
    }
    value = {
        "schema_version": 1,
        "setup_id": "setup-one",
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
            "reviewer": "reviewer",
            "reviewed_at": now,
            "repository": "starai-viola-lerobot-ops",
            "python_version": "3.12.13",
            "lerobot_version": "0.6.1",
            "conda_environment": "lerobot",
            "execution_backend": "direct_lerobot_fashionstar",
            "entrypoint_path": str(executor),
            "entrypoint_sha256": sha256_file(executor),
            "capabilities": list(EXECUTOR_CAPABILITIES),
        },
        "estop": {"tested_at": now, "operator": "operator", "passed": True},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _identity():
    return viola_handoff.RuntimeIdentity(
        role="pc_a",
        repository_commit="a" * 40,
        repository_clean=True,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def test_capture_reads_then_closes_before_wandb_publish(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    port = _ReadOnlyPort()
    events = []
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1), base + timedelta(seconds=1)))

    def publish(identity, **kwargs):
        assert port.closed is True
        events.append((identity, kwargs))
        return identity

    result = capture_frozen_state(
        setup,
        output_root=tmp_path / "capture",
        operator="operator",
        repo_root=tmp_path,
        wandb_entity="entity",
        confirm=lambda challenge: challenge == "READ FROZEN STATE setup-one",
        port_factory=lambda path, baud: port,
        publisher=publish,
        now=lambda: next(times),
    )

    assert port.opened and port.closed and port.monitor_calls == 1
    assert len(result.state) == 7
    assert result.sync_path.is_file()
    assert events[0][1]["config"]["operation"] == "frozen_state_capture"


def test_capture_refuses_without_operator_action_before_port_factory(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    calls = []
    with pytest.raises(SafetyGateError, match="explicit operator"):
        capture_frozen_state(
            setup,
            output_root=tmp_path / "capture",
            operator="operator",
            repo_root=tmp_path,
            wandb_entity="entity",
            confirm=lambda challenge: False,
            port_factory=lambda path, baud: calls.append((path, baud)),
        )
    assert calls == []
