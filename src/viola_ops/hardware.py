"""Reviewed public-API hardware adapter for the Viola follower arm.

This module never patches the installed StarAI plugins.  The stock plugin's
``connect`` method disables torque and commands a fixed pose, so production
execution instead uses the public FashionStar SDK to read the current pose and
then holds that same pose after the operator has received a motion permit.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Final, Protocol

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .errors import SafetyGateError, ValidationError
from .safety import JOINTS, MotionPermit, assert_permit_current, validate_action

FOLLOWER_MODELS: Final = {joint: "ra8-u25" for joint in JOINTS}
BODY_JOINTS: Final = tuple(joint for joint in JOINTS if joint != "gripper")
SERVO_RESOLUTION: Final = 4096
SERVO_HALF_RANGE_DEGREES: Final = 180.0
DEFAULT_MOTION_TIME_MS: Final = 100
DEFAULT_ACCEL_TIME: Final = 50
DEFAULT_DECEL_TIME: Final = 50
DEFAULT_BODY_POWER: Final = 0
DEFAULT_GRIPPER_POWER: Final = 1000


@dataclass(frozen=True, slots=True)
class JointCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    proposed: Mapping[str, float]
    sent: Mapping[str, float]
    feedback_before: Mapping[str, float]
    feedback_after: Mapping[str, float]
    command_sequence: int = 0
    control_started_ns: int = 0
    prewrite_received_ns: int = 0
    write_started_ns: int = 0
    write_completed_ns: int = 0
    feedback_received_ns: int = 0
    previous_feedback_received_ns: int = 0
    minimum_observable_progress: Mapping[str, float] = field(default_factory=dict)
    poll_count: int = 0
    feedback_failed_ns: int = 0
    feedback_error: str | None = None


class PostWriteFeedbackError(RuntimeError):
    """The public SDK accepted a write, but its required feedback read failed."""

    def __init__(self, receipt: ActionReceipt, cause: BaseException) -> None:
        self.action_receipt = receipt
        super().__init__(
            "motor write completed, but synchronous feedback failed: "
            f"{type(cause).__name__}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    """One synchronous state sample plus two public-camera read receipts."""

    observation: Mapping[str, Any]
    captured_at_ns: int
    observation_read_ms: float
    camera_reads: Mapping[str, Mapping[str, float | int]]


class ServoMonitor(Protocol):
    current_position: float


class PublicFashionStarPort(Protocol):
    """Only exported SDK operations used by the adapter."""

    def openPort(self) -> Any: ...

    def closePort(self) -> Any: ...

    def ping(self, servo_id: int) -> Any: ...

    def SyncServoMonitor(
        self, motors: dict[str, int], realtime: bool = False
    ) -> dict[str, ServoMonitor]: ...

    def SyncPositionControl_EX(self, motors: dict[str, Any]) -> None: ...


@RobotConfig.register_subclass("viola_safe_follower")
@dataclass
class SafeViolaConfig(RobotConfig):
    """Configuration bound by an accepted rollout session."""

    port: str = ""
    calibration_path: Path = Path()
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class SafeViolaRobot(Robot):
    """LeRobot follower that cannot exist without a current motion permit."""

    config_class = SafeViolaConfig
    name = "viola_safe_follower"

    def __init__(
        self,
        config: SafeViolaConfig,
        permit: MotionPermit,
        *,
        port_factory: Callable[[str, int], PublicFashionStarPort] | None = None,
        command_factory: Callable[..., Any] | None = None,
        camera_factory: Callable[[Mapping[str, CameraConfig]], dict[str, Any]] | None = None,
    ) -> None:
        assert_permit_current(permit, require_active=True)
        super().__init__(config)
        self.config = config
        self.permit = permit
        self._calibration = load_calibration(config.calibration_path)
        self._port_factory = port_factory or _fashionstar_port
        self._command_factory = command_factory or _fashionstar_command
        self.cameras = (
            camera_factory(config.cameras)
            if camera_factory is not None
            else make_cameras_from_configs(config.cameras)
        )
        self._port: PublicFashionStarPort | None = None
        self._connected = False
        self._last_receipt: ActionReceipt | None = None
        self._command_sequence = 0
        self._last_feedback_received_ns = 0

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        camera_features = {
            name: (config.height, config.width, 3)
            for name, config in self.config.cameras.items()
        }
        return {**self.action_features, **camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return set(self._calibration) == set(JOINTS)

    @property
    def last_receipt(self) -> ActionReceipt | None:
        return self._last_receipt

    def connect(self, calibrate: bool = False) -> None:
        """Open, read, and hold the measured pose after authorization.

        Calibration is immutable session input.  Runtime calibration is
        intentionally unsupported because it would change reviewed setup.
        """

        assert_permit_current(self.permit, require_active=True)
        if calibrate:
            raise SafetyGateError("runtime calibration is not allowed by a rollout session")
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")
        port = self._port_factory(self.config.port, 1_000_000)
        connected_cameras: list[Any] = []
        opened = False
        try:
            opened_result = port.openPort()
            if opened_result is False:
                raise DeviceNotConnectedError("FashionStar serial port did not open")
            opened = True
            for joint in JOINTS:
                if not port.ping(self._calibration[joint].id):
                    raise DeviceNotConnectedError(
                        f"FashionStar servo {self._calibration[joint].id} ({joint}) did not respond"
                    )
            self._port = port
            current = self._read_positions()
            # A calibration-valid position may still lie outside the narrower
            # limits reviewed for this session.  Reject it before cameras or a
            # motor write exist; never "hold" by commanding a clipped value.
            safe_current = validate_action(current, current, self.permit)
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
            self._connected = True
            # This is the first command and occurs only after the explicit arm
            # challenge.  Requested position equals the synchronous measurement.
            self._write_positions(safe_current)
            feedback = self._read_positions()
            connected_at = time.perf_counter_ns()
            self._last_receipt = ActionReceipt(
                safe_current,
                safe_current,
                current,
                feedback,
                command_sequence=0,
                feedback_received_ns=connected_at,
                minimum_observable_progress=self._minimum_observable_progress(),
                poll_count=1,
            )
            # Policy-action receipt chaining starts at zero.  The startup hold
            # is commissioning evidence, not action sequence 1.
            self._last_feedback_received_ns = 0
        except BaseException:
            self._connected = False
            self._port = None
            for camera in reversed(connected_cameras):
                try:
                    camera.disconnect()
                except Exception:
                    pass
            if opened:
                try:
                    port.closePort()
                except Exception:
                    pass
            raise

    def calibrate(self) -> None:
        raise SafetyGateError("calibration requires a separate reviewed commissioning workflow")

    def configure(self) -> None:
        """No implicit device configuration is permitted during rollout."""

    def get_observation(self) -> dict[str, Any]:
        return dict(self.capture_observation().observation)

    def capture_observation(self) -> ObservationReceipt:
        """Capture state and both images with monotonic public-API timing."""

        assert_permit_current(self.permit, require_active=True)
        self._require_connected()
        started = time.perf_counter_ns()
        observation: dict[str, Any] = dict(self._read_positions())
        camera_reads: dict[str, dict[str, float | int]] = {}
        for name, camera in self.cameras.items():
            camera_started = time.perf_counter_ns()
            observation[name] = camera.async_read(timeout_ms=100)
            received = time.perf_counter_ns()
            camera_reads[name] = {
                "started_at_ns": camera_started,
                "received_at_ns": received,
                "read_ms": (received - camera_started) / 1_000_000.0,
            }
        captured = time.perf_counter_ns()
        return ObservationReceipt(
            observation=observation,
            captured_at_ns=captured,
            observation_read_ms=(captured - started) / 1_000_000.0,
            camera_reads=camera_reads,
        )

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        self._require_connected()
        assert_permit_current(self.permit, require_active=True)
        control_started = time.perf_counter_ns()
        before = self._read_positions()
        prewrite_received = time.perf_counter_ns()
        safe = validate_action(action, before, self.permit)
        proposed = {key: float(action[key]) for key in self.action_features}
        minimum_progress = self._minimum_observable_progress()
        write_started = time.perf_counter_ns()
        self._write_positions(safe)
        write_completed = time.perf_counter_ns()
        # Once the public SDK call returns, this sequence number represents a
        # physical write even if the following monitor read fails.
        self._command_sequence += 1
        command_sequence = self._command_sequence
        deadline = control_started + round(300.0 * 1_000_000.0)
        poll_count = 0
        try:
            while True:
                poll_count += 1
                after = self._read_positions()
                feedback_received = time.perf_counter_ns()
                if self._feedback_has_minimum_progress(before, safe, after, minimum_progress):
                    break
                if feedback_received >= deadline:
                    break
                time.sleep(0.001)
        except BaseException as exc:
            failed_at = time.perf_counter_ns()
            receipt = ActionReceipt(
                proposed=proposed,
                sent=safe,
                feedback_before=before,
                feedback_after={},
                command_sequence=command_sequence,
                control_started_ns=control_started,
                prewrite_received_ns=prewrite_received,
                write_started_ns=write_started,
                write_completed_ns=write_completed,
                feedback_received_ns=0,
                previous_feedback_received_ns=self._last_feedback_received_ns,
                minimum_observable_progress=minimum_progress,
                poll_count=poll_count,
                feedback_failed_ns=failed_at,
                feedback_error=f"{type(exc).__name__}: {exc}",
            )
            self._last_receipt = receipt
            raise PostWriteFeedbackError(receipt, exc) from exc
        self._last_receipt = ActionReceipt(
            proposed=proposed,
            sent=safe,
            feedback_before=before,
            feedback_after=after,
            command_sequence=command_sequence,
            control_started_ns=control_started,
            prewrite_received_ns=prewrite_received,
            write_started_ns=write_started,
            write_completed_ns=write_completed,
            feedback_received_ns=feedback_received,
            previous_feedback_received_ns=self._last_feedback_received_ns,
            minimum_observable_progress=minimum_progress,
            poll_count=poll_count,
        )
        self._last_feedback_received_ns = feedback_received
        return safe

    def disconnect(self) -> None:
        """Close cameras and serial transport without changing motor torque."""

        failures: list[Exception] = []
        for camera in reversed(tuple(self.cameras.values())):
            try:
                if getattr(camera, "is_connected", True):
                    camera.disconnect()
            except Exception as exc:
                failures.append(exc)
        port, self._port = self._port, None
        self._connected = False
        if port is not None:
            try:
                port.closePort()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(f"Viola disconnect had {len(failures)} cleanup failure(s)")

    def _read_positions(self) -> dict[str, float]:
        port = self._require_port()
        ids = {joint: self._calibration[joint].id for joint in JOINTS}
        monitors = port.SyncServoMonitor(ids, realtime=True)
        if set(monitors) != set(JOINTS):
            raise DeviceNotConnectedError("synchronous feedback did not return all seven joints")
        result: dict[str, float] = {}
        for joint in JOINTS:
            monitor = monitors[joint]
            degrees = _finite(getattr(monitor, "current_position", None), f"{joint} feedback")
            raw = degrees_to_raw(degrees)
            result[f"{joint}.pos"] = raw_to_normalized(
                raw, self._calibration[joint], gripper=(joint == "gripper")
            )
        return result

    def _write_positions(self, values: Mapping[str, float]) -> None:
        port = self._require_port()
        expected = {f"{joint}.pos" for joint in JOINTS}
        if set(values) != expected:
            raise SafetyGateError("motor command must name exactly seven Viola joints")
        commands: dict[str, Any] = {}
        for joint in JOINTS:
            normalized = _finite(values[f"{joint}.pos"], f"{joint} command")
            raw = normalized_to_raw(
                normalized, self._calibration[joint], gripper=(joint == "gripper")
            )
            target_tenths = int(round(raw_to_degrees(raw) * 10.0))
            power = DEFAULT_GRIPPER_POWER if joint == "gripper" else DEFAULT_BODY_POWER
            commands[joint] = self._command_factory(
                self._calibration[joint].id,
                target_tenths,
                DEFAULT_MOTION_TIME_MS,
                power,
                DEFAULT_ACCEL_TIME,
                DEFAULT_DECEL_TIME,
            )
        port.SyncPositionControl_EX(commands)

    def _minimum_observable_progress(self) -> dict[str, float]:
        """Return one encoder-count quantum in normalized joint units."""

        values: dict[str, float] = {}
        for joint in JOINTS:
            span = self._calibration[joint].range_max - self._calibration[joint].range_min
            normalized_span = 100.0 if joint == "gripper" else 200.0
            values[joint] = normalized_span / span
        return values

    @staticmethod
    def _feedback_has_minimum_progress(
        before: Mapping[str, float],
        target: Mapping[str, float],
        feedback: Mapping[str, float],
        minimum: Mapping[str, float],
    ) -> bool:
        for joint in JOINTS:
            key = f"{joint}.pos"
            delta = target[key] - before[key]
            quantum = minimum[joint]
            if abs(delta) < math.nextafter(quantum, 0.0):
                continue
            progress = math.copysign(1.0, delta) * (feedback[key] - before[key])
            if progress < math.nextafter(quantum, 0.0):
                return False
        return True

    def _require_connected(self) -> None:
        if not self._connected or self._port is None:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def _require_port(self) -> PublicFashionStarPort:
        if self._port is None:
            raise DeviceNotConnectedError("FashionStar port is not connected")
        return self._port


def load_calibration(path: Path) -> dict[str, JointCalibration]:
    """Load and strictly validate the reviewed seven-joint calibration."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read reviewed calibration {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != set(JOINTS):
        raise ValidationError("calibration must name exactly seven Viola joints")
    result: dict[str, JointCalibration] = {}
    expected_fields = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    for expected_id, joint in enumerate(JOINTS):
        record = value[joint]
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValidationError(f"calibration fields are invalid for {joint}")
        calibration = JointCalibration(**record)
        if calibration.id != expected_id:
            raise ValidationError(f"calibration ID for {joint} must be {expected_id}")
        if calibration.drive_mode not in {0, 1} or calibration.homing_offset != 0:
            raise ValidationError(f"calibration mode/offset is invalid for {joint}")
        if not 0 <= calibration.range_min < calibration.range_max <= SERVO_RESOLUTION:
            raise ValidationError(f"calibration range is invalid for {joint}")
        result[joint] = calibration
    return result


def raw_to_normalized(raw: float, calibration: JointCalibration, *, gripper: bool) -> float:
    measured = _finite(raw, "raw motor position")
    if not calibration.range_min <= measured <= calibration.range_max:
        raise SafetyGateError(
            "measured motor position is outside the reviewed calibration range"
        )
    fraction = (measured - calibration.range_min) / (
        calibration.range_max - calibration.range_min
    )
    if gripper:
        value = fraction * 100.0
        return 100.0 - value if calibration.drive_mode else value
    value = fraction * 200.0 - 100.0
    return -value if calibration.drive_mode else value


def normalized_to_raw(
    normalized: float, calibration: JointCalibration, *, gripper: bool
) -> float:
    value = _finite(normalized, "normalized motor position")
    if gripper:
        if not 0.0 <= value <= 100.0:
            raise SafetyGateError("gripper command is outside 0..100")
        value = 100.0 - value if calibration.drive_mode else value
        fraction = value / 100.0
    else:
        if not -100.0 <= value <= 100.0:
            raise SafetyGateError("body joint command is outside -100..100")
        value = -value if calibration.drive_mode else value
        fraction = (value + 100.0) / 200.0
    return fraction * (calibration.range_max - calibration.range_min) + calibration.range_min


def degrees_to_raw(degrees: float) -> float:
    value = _finite(degrees, "servo position in degrees")
    if not -SERVO_HALF_RANGE_DEGREES <= value <= SERVO_HALF_RANGE_DEGREES:
        raise SafetyGateError("servo feedback is outside the public SDK position domain")
    return (value + SERVO_HALF_RANGE_DEGREES) / (2 * SERVO_HALF_RANGE_DEGREES) * SERVO_RESOLUTION


def raw_to_degrees(raw: float) -> float:
    value = _finite(raw, "raw motor target")
    if not 0.0 <= value <= SERVO_RESOLUTION:
        raise SafetyGateError("raw motor target is outside the public SDK position domain")
    return value / SERVO_RESOLUTION * (2 * SERVO_HALF_RANGE_DEGREES) - SERVO_HALF_RANGE_DEGREES


def config_from_permit(permit: MotionPermit) -> SafeViolaConfig:
    """Build the only production robot configuration: the permit's setup."""

    assert_permit_current(permit, require_active=True)
    if (
        not permit.robot_port.startswith("/dev/")
        or not permit.calibration_path.is_file()
        or set(permit.camera_configs) != {"front", "up"}
    ):
        raise SafetyGateError("motion permit does not contain a complete reviewed setup")
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cameras: dict[str, CameraConfig] = {}
    for name in ("front", "up"):
        item = permit.camera_configs[name]
        cameras[name] = OpenCVCameraConfig(
            index_or_path=Path(str(item["index_or_path"])),
            width=int(item["width"]),
            height=int(item["height"]),
            fps=int(item["fps"]),
        )
    return SafeViolaConfig(
        port=permit.robot_port,
        calibration_path=permit.calibration_path,
        cameras=cameras,
    )


def _fashionstar_port(port: str, baudrate: int) -> PublicFashionStarPort:
    from fashionstar_uart_sdk import PortHandler

    return PortHandler(port, baudrate)


def _fashionstar_command(*arguments: Any) -> Any:
    from fashionstar_uart_sdk import SyncPositionControlOptions

    return SyncPositionControlOptions(*arguments)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SafetyGateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SafetyGateError(f"{label} must be finite")
    return result
