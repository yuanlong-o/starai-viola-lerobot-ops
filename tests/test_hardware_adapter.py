from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import viola_ops.hardware as hardware
from viola_ops.errors import SafetyGateError
from viola_ops.hardware import (
    JointCalibration,
    PostWriteFeedbackError,
    SafeViolaConfig,
    SafeViolaRobot,
    degrees_to_raw,
    load_calibration,
    normalized_to_raw,
    raw_to_degrees,
    raw_to_normalized,
)
from viola_ops.safety import JOINTS, MotionPermit


@dataclass
class Monitor:
    current_position: float = 0.0


class FakePort:
    def __init__(self, *, current_position: float = 0.0) -> None:
        self.opened = False
        self.closed = False
        self.current_position = current_position
        self.fail_reads = False
        self.reads_before_failure = 0
        self.pings: list[int] = []
        self.writes: list[dict[str, Any]] = []

    def openPort(self) -> bool:
        self.opened = True
        return True

    def closePort(self) -> None:
        self.closed = True

    def ping(self, servo_id: int) -> bool:
        self.pings.append(servo_id)
        return True

    def SyncServoMonitor(self, motors: dict[str, int], realtime: bool = False):
        assert realtime is True
        if self.fail_reads:
            if self.reads_before_failure == 0:
                raise OSError("monitor unavailable")
            self.reads_before_failure -= 1
        return {joint: Monitor(self.current_position) for joint in motors}

    def SyncPositionControl_EX(self, motors: dict[str, Any]) -> None:
        self.writes.append(motors)


def _calibration(path: Path) -> Path:
    value = {
        joint: {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4096,
        }
        for index, joint in enumerate(JOINTS)
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _permit() -> MotionPermit:
    return MotionPermit(
        session_id="session",
        session_bundle_id="a" * 64,
        candidate_bundle_id="b" * 64,
        policy="act",
        phase="hold",
        trial="hold",
        operator="operator",
        setup_hashes={name: "c" * 64 for name in ("calibration", "camera", "robot", "reset")},
        absolute_limits={joint: ((0.0, 100.0) if joint == "gripper" else (-100.0, 100.0)) for joint in JOINTS},
        max_step_deltas={joint: 1.0 for joint in JOINTS},
        speed_scale=1.0,
        issued_at=datetime.now(UTC),
        estop_tested_at=datetime.now(UTC),
        _nonce="test",
        _authority=object(),
    )


def test_position_conversions_round_trip() -> None:
    body = JointCalibration(0, 0, 0, 1012, 3242)
    inverted = JointCalibration(0, 1, 0, 1012, 3242)
    gripper = JointCalibration(6, 0, 0, 2013, 3322)
    for normalized in (-100.0, -50.0, 0.0, 50.0, 100.0):
        raw = normalized_to_raw(normalized, body, gripper=False)
        assert raw_to_normalized(raw, body, gripper=False) == pytest.approx(normalized)
        raw_inverted = normalized_to_raw(normalized, inverted, gripper=False)
        assert raw_to_normalized(raw_inverted, inverted, gripper=False) == pytest.approx(normalized)
    for normalized in (0.0, 25.0, 50.0, 75.0, 100.0):
        raw = normalized_to_raw(normalized, gripper, gripper=True)
        assert raw_to_normalized(raw, gripper, gripper=True) == pytest.approx(normalized)
    for degrees in (-180.0, -90.0, 0.0, 90.0, 180.0):
        assert raw_to_degrees(degrees_to_raw(degrees)) == pytest.approx(degrees)


@pytest.mark.parametrize("degrees", [-180.0001, 180.0001, float("nan")])
def test_feedback_degrees_are_rejected_instead_of_clamped(degrees: float) -> None:
    with pytest.raises(SafetyGateError):
        degrees_to_raw(degrees)


@pytest.mark.parametrize("raw", [-0.001, 4096.001, float("inf")])
def test_raw_motor_targets_are_rejected_instead_of_clamped(raw: float) -> None:
    with pytest.raises(SafetyGateError):
        raw_to_degrees(raw)


@pytest.mark.parametrize("degrees", [-180.001, 180.001, float("nan")])
def test_feedback_outside_public_sdk_domain_is_rejected(degrees: float) -> None:
    with pytest.raises(SafetyGateError):
        degrees_to_raw(degrees)


def test_feedback_outside_reviewed_calibration_is_rejected() -> None:
    calibration = JointCalibration(0, 0, 0, 1012, 3242)
    with pytest.raises(SafetyGateError, match="reviewed calibration"):
        raw_to_normalized(1011.9, calibration, gripper=False)


def test_load_calibration_requires_exact_joint_ids(tmp_path: Path) -> None:
    path = _calibration(tmp_path / "calibration.json")
    assert tuple(load_calibration(path)) == JOINTS
    value = json.loads(path.read_text())
    value["Motor_2"]["id"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(Exception, match="must be 2"):
        load_calibration(path)


def test_safe_adapter_holds_measured_pose_without_plugin_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "assert_permit_current", lambda permit, **kwargs: None)
    port = FakePort()
    config = SafeViolaConfig(
        port="/dev/not-opened-by-test",
        calibration_path=_calibration(tmp_path / "calibration.json"),
        cameras={},
        id="test",
        calibration_dir=tmp_path / "lerobot-calibration",
    )
    robot = SafeViolaRobot(
        config,
        _permit(),
        port_factory=lambda _path, _baud: port,
        command_factory=lambda *values: values,
        camera_factory=lambda _configs: {},
    )
    robot.connect(calibrate=False)
    assert port.pings == list(range(7))
    assert len(port.writes) == 1
    hold = port.writes[0]
    assert set(hold) == set(JOINTS)
    assert all(command[1] == 0 for command in hold.values())
    assert robot.last_receipt is not None
    assert robot.last_receipt.proposed == robot.last_receipt.feedback_before
    robot.disconnect()
    assert port.closed is True


def test_out_of_domain_feedback_prevents_initial_hold_write_and_closes_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "assert_permit_current", lambda permit, **kwargs: None)
    port = FakePort(current_position=181.0)
    robot = SafeViolaRobot(
        SafeViolaConfig(
            port="/dev/fake",
            calibration_path=_calibration(tmp_path / "calibration.json"),
            cameras={},
            id="test",
            calibration_dir=tmp_path / "lerobot-calibration",
        ),
        _permit(),
        port_factory=lambda _path, _baud: port,
        command_factory=lambda *values: values,
        camera_factory=lambda _configs: {},
    )
    with pytest.raises(SafetyGateError, match="outside the public SDK position domain"):
        robot.connect(calibrate=False)
    assert port.writes == []
    assert port.closed is True
    assert robot.is_connected is False


def test_pose_outside_reviewed_session_limits_prevents_initial_hold_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "assert_permit_current", lambda permit, **kwargs: None)
    permit = _permit()
    permit.absolute_limits["Motor_0"] = (1.0, 2.0)
    port = FakePort(current_position=0.0)
    robot = SafeViolaRobot(
        SafeViolaConfig(
            port="/dev/fake",
            calibration_path=_calibration(tmp_path / "calibration.json"),
            cameras={},
            id="test",
            calibration_dir=tmp_path / "lerobot-calibration",
        ),
        permit,
        port_factory=lambda _path, _baud: port,
        command_factory=lambda *values: values,
        camera_factory=lambda _configs: {},
    )
    with pytest.raises(SafetyGateError, match="outside reviewed absolute limits"):
        robot.connect(calibrate=False)
    assert port.writes == []
    assert port.closed is True
    assert robot.is_connected is False


def test_rejected_action_causes_zero_additional_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "assert_permit_current", lambda permit, **kwargs: None)
    port = FakePort()
    config = SafeViolaConfig(
        port="/dev/fake",
        calibration_path=_calibration(tmp_path / "calibration.json"),
        cameras={},
        id="test",
        calibration_dir=tmp_path / "lerobot-calibration",
    )
    robot = SafeViolaRobot(
        config,
        _permit(),
        port_factory=lambda _path, _baud: port,
        command_factory=lambda *values: values,
        camera_factory=lambda _configs: {},
    )
    robot.connect(calibrate=False)
    action = {f"{joint}.pos": 0.0 for joint in JOINTS}
    action["Motor_0.pos"] = float("nan")
    with pytest.raises(SafetyGateError, match="finite"):
        robot.send_action(action)
    assert len(port.writes) == 1
    robot.disconnect()


def test_successful_write_retains_receipt_when_feedback_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "assert_permit_current", lambda permit, **kwargs: None)
    port = FakePort()
    robot = SafeViolaRobot(
        SafeViolaConfig(
            port="/dev/fake",
            calibration_path=_calibration(tmp_path / "calibration.json"),
            cameras={},
            id="test",
            calibration_dir=tmp_path / "lerobot-calibration",
        ),
        _permit(),
        port_factory=lambda _path, _baud: port,
        command_factory=lambda *values: values,
        camera_factory=lambda _configs: {},
    )
    robot.connect(calibrate=False)
    port.fail_reads = True
    port.reads_before_failure = 1
    action = {f"{joint}.pos": (50.0 if joint == "gripper" else 0.0) for joint in JOINTS}
    action["Motor_0.pos"] = 0.5

    with pytest.raises(PostWriteFeedbackError, match="motor write completed") as failed:
        robot.send_action(action)

    assert len(port.writes) == 2
    assert failed.value.action_receipt is robot.last_receipt
    assert robot.last_receipt is not None
    assert robot.last_receipt.command_sequence == 1
    assert robot.last_receipt.sent == action
    assert robot.last_receipt.feedback_after == {}
    assert robot.last_receipt.feedback_received_ns == 0
    assert robot.last_receipt.feedback_failed_ns >= robot.last_receipt.write_completed_ns
    assert robot.last_receipt.feedback_error == "OSError: monitor unavailable"
    robot.disconnect()
