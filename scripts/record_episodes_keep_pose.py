#!/usr/bin/env python3
"""Record StarAI Viola episodes without commanding the plugin startup pose.

The installed Viola and Violin plugins move both arms to a fixed configuration
while connecting.  This process-local wrapper replaces those startup methods and
maps leader displacement onto the follower's measured startup pose:

    follower target = follower startup + (leader current - leader startup)

The transformed, bounded follower target is returned by the teleoperator.  The
standard LeRobot recorder therefore sends that target to the Viola *and* stores
the same target in the dataset's action column.
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

# OpenCV reads this option when its cv2 module is imported.  Keep individual
# V4L2 reads short enough for the bounded startup recovery below to work.
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")

from lerobot.utils.import_utils import register_third_party_devices


MAX_STEP_ENV = "LEROBOT_KEEP_POSE_MAX_STEP"
CAMERA_STARTUP_TIMEOUT_ENV = "LEROBOT_CAMERA_STARTUP_TIMEOUT_S"
CAMERA_ASYNC_TIMEOUT_ENV = "LEROBOT_CAMERA_ASYNC_TIMEOUT_MS"
CAMERA_MAX_FRAME_AGE_ENV = "LEROBOT_CAMERA_MAX_FRAME_AGE_MS"
OPENCV_V4L_SELECT_TIMEOUT_ENV = "OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"
STARAI_READ_TIMEOUT_ENV = "LEROBOT_STARAI_READ_TIMEOUT_S"
MIN_EPISODE_TIME_ENV = "LEROBOT_MIN_EPISODE_TIME_S"
ACTION_KEYS = tuple([f"Motor_{index}.pos" for index in range(6)] + ["gripper.pos"])


def install_starai_position_read_retry(timeout_s: float, retry_delay_s: float = 0.02) -> None:
    """Retry transient StarAI monitor reads that contain a missing position.

    The third-party StarAI bus assumes every FashionStar monitor response has a
    numeric ``current_position``.  Immediately after connecting, or after an
    occasional incomplete serial response, the SDK can instead return ``None``
    and the bus raises ``TypeError`` while comparing it with the joint limits.

    Patch only this known failure for position/monitor reads and only inside
    this recorder process.  Unrelated type errors and persistent communication
    failures still abort with a clear diagnostic.
    """

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("StarAI read timeout must be a positive finite number")
    if not math.isfinite(retry_delay_s) or retry_delay_s <= 0:
        raise ValueError("StarAI retry delay must be a positive finite number")

    from lerobot_motor_starai.starai import StaraiMotorsBus

    original_sync_read = StaraiMotorsBus.sync_read
    if getattr(original_sync_read, "_keep_pose_missing_position_retry", False):
        return

    def sync_read_with_missing_position_retry(
        self: Any,
        data_name: str,
        motors: Any = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        failed_reads = 0
        last_error: TypeError | None = None

        while True:
            try:
                result = original_sync_read(
                    self,
                    data_name,
                    motors,
                    normalize=normalize,
                    num_retry=num_retry,
                )
                if failed_reads:
                    logging.info(
                        "StarAI %s read on %s recovered after %d incomplete response(s)",
                        data_name,
                        self.port,
                        failed_reads,
                    )
                return result
            except TypeError as error:
                is_missing_position = (
                    data_name in {"Present_Position", "Monitor"}
                    and "between instances of 'NoneType' and 'int'" in str(error)
                    and ("'>='" in str(error) or "'<='" in str(error))
                )
                if not is_missing_position:
                    raise

                failed_reads += 1
                last_error = error
                if time.monotonic() >= deadline:
                    break
                time.sleep(retry_delay_s)

        raise RuntimeError(
            f"StarAI {data_name} read on {self.port} returned a missing motor position "
            f"for {timeout_s:.2f}s ({failed_reads} incomplete responses). Check that all "
            "seven motors are powered and that the arm's USB/data cable is firmly connected."
        ) from last_error

    sync_read_with_missing_position_retry._keep_pose_missing_position_retry = True
    StaraiMotorsBus.sync_read = sync_read_with_missing_position_retry


@dataclass
class RelativePoseMapper:
    """Track startup anchors and map leader motion into safe follower targets."""

    max_step: float = 3.0
    follower_start: dict[str, float] | None = None
    leader_start: dict[str, float] | None = None
    previous_action: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_step) or self.max_step <= 0:
            raise ValueError("max_step must be a positive finite number")

    @staticmethod
    def _validated_action(action: dict[str, float], source: str) -> dict[str, float]:
        if set(action) != set(ACTION_KEYS):
            missing = sorted(set(ACTION_KEYS) - set(action))
            unexpected = sorted(set(action) - set(ACTION_KEYS))
            raise ValueError(
                f"{source} action keys do not match the seven expected StarAI joints; "
                f"missing={missing}, unexpected={unexpected}"
            )

        validated = {key: float(action[key]) for key in ACTION_KEYS}
        non_finite = [key for key, value in validated.items() if not math.isfinite(value)]
        if non_finite:
            raise ValueError(f"{source} action has non-finite values for {non_finite}")
        return validated

    def set_follower_start(self, action: dict[str, float]) -> None:
        self.follower_start = self._validated_action(action, "Follower")
        self.previous_action = self.follower_start.copy()

    def map_leader_action(self, leader_action: dict[str, float]) -> dict[str, float]:
        if self.follower_start is None:
            raise RuntimeError("Follower startup pose was not captured before teleoperation began")

        current = self._validated_action(leader_action, "Leader")
        if self.leader_start is None:
            self.leader_start = current.copy()

        mapped: dict[str, float] = {}
        for key, follower_start_value in self.follower_start.items():
            desired = follower_start_value + (current[key] - self.leader_start[key])
            lower_limit, upper_limit = (0.0, 100.0) if key == "gripper.pos" else (-100.0, 100.0)
            desired = min(upper_limit, max(lower_limit, desired))

            previous = self.previous_action[key]
            mapped[key] = min(previous + self.max_step, max(previous - self.max_step, desired))

        self.previous_action = mapped.copy()
        return mapped


def install_keep_pose_recording_patch(
    max_step: float,
) -> tuple[RelativePoseMapper, Callable[[], None]]:
    """Patch the third-party devices for this recorder process only."""

    from lerobot_robot_viola.starai_viola import StaraiViola
    from lerobot_teleoperator_violin.starai_violin import StaraiViolin

    mapper = RelativePoseMapper(max_step=max_step)
    connect_robot = StaraiViola.connect
    connect_teleop = StaraiViolin.connect
    read_follower_action = StaraiViola.get_action
    read_leader_action = StaraiViolin.get_action
    cleanup_lock = threading.Lock()
    tracked_cleanups: list[tuple[int, str, Callable[[], None]]] = []

    def force_close_bus(bus: Any, label: str) -> None:
        """Close a StarAI serial handle even if torque-disable cleanup failed."""

        try:
            if bus.is_connected:
                bus.port_handler.closePort()
                logging.warning(
                    "Forced the %s serial port closed after normal disconnect failed; "
                    "torque disable may not have completed",
                    label,
                )
        except Exception:
            logging.exception("Could not force the %s serial port closed", label)

    def disconnect_bus(bus: Any, disable_torque: bool, label: str) -> None:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque)
        except Exception:
            logging.exception("Could not disconnect the %s", label)
            force_close_bus(bus, label)

    def disconnect_camera(camera: Any) -> None:
        try:
            if camera.is_connected or getattr(camera, "thread", None) is not None:
                camera.disconnect()
            elif getattr(camera, "videocapture", None) is not None:
                camera.videocapture.release()
                camera.videocapture = None
        except Exception:
            logging.exception("Could not release %s during recorder cleanup", camera)
            try:
                if getattr(camera, "thread", None) is not None:
                    camera._stop_read_thread()
            except Exception:
                logging.exception("Could not force-stop %s read thread", camera)
            try:
                if getattr(camera, "videocapture", None) is not None:
                    camera.videocapture.release()
                    camera.videocapture = None
            except Exception:
                logging.exception("Could not force-release %s capture", camera)

    def cleanup_follower(robot: Any) -> None:
        disconnect_bus(
            robot.bus,
            bool(robot.config.disable_torque_on_disconnect),
            "follower bus",
        )
        for camera in robot.cameras.values():
            disconnect_camera(camera)

    def cleanup_teacher(teleop: Any) -> None:
        disconnect_bus(teleop.bus, True, "teacher bus")

    def track_cleanup(device: Any, label: str, cleanup: Callable[[], None]) -> None:
        with cleanup_lock:
            if not any(device_id == id(device) for device_id, _, _ in tracked_cleanups):
                tracked_cleanups.append((id(device), label, cleanup))

    def cleanup_connected_devices() -> None:
        """Best-effort cleanup for every device after normal exit or any exception."""

        with cleanup_lock:
            pending = list(reversed(tracked_cleanups))
            tracked_cleanups.clear()

        for _, label, cleanup in pending:
            try:
                cleanup()
            except Exception:
                # Cleanup callbacks are already defensive; keep this final guard
                # so one plugin cannot prevent the other device from releasing.
                logging.exception("Unexpected error cleaning up the connected %s", label)

    def connect_robot_with_rollback(self: Any, calibrate: bool = True) -> None:
        """Release every partially connected device if robot startup fails."""

        # Register before connecting so even an interrupt immediately after the
        # plugin returns cannot fall between connection and cleanup tracking.
        track_cleanup(self, "follower", lambda: cleanup_follower(self))
        try:
            connect_robot(self, calibrate=calibrate)
        except BaseException:
            cleanup_follower(self)
            raise

    def connect_teleop_with_rollback(self: Any, calibrate: bool = True) -> None:
        track_cleanup(self, "teacher", lambda: cleanup_teacher(self))
        try:
            connect_teleop(self, calibrate=calibrate)
        except BaseException:
            cleanup_teacher(self)
            raise

    def hold_follower_at_current_pose(self: Any) -> dict[str, float]:
        current_action = read_follower_action(self)
        mapper.set_follower_start(current_action)
        current_goal = {
            key.removesuffix(".pos"): value
            for key, value in current_action.items()
            if key.endswith(".pos")
        }
        # The camera objects are not connected at this point, so write through
        # the already-connected motor bus instead of calling robot.send_action().
        self.bus.sync_write("Goal_Position", current_goal, motion_time=100)
        return current_action

    def leave_leader_at_current_pose(self: Any) -> dict[str, float]:
        # The motor bus has already unlocked the leader.  Read only; do not
        # command the plugin's fixed startup pose.
        return read_leader_action(self)

    def get_relative_leader_action(self: Any) -> dict[str, float]:
        return mapper.map_leader_action(read_leader_action(self))

    StaraiViola.connect = connect_robot_with_rollback
    StaraiViolin.connect = connect_teleop_with_rollback
    StaraiViola.move_to_initial_position = hold_follower_at_current_pose
    StaraiViolin.move_to_initial_position = leave_leader_at_current_pose
    StaraiViolin.get_action = get_relative_leader_action
    return mapper, cleanup_connected_devices


def install_opencv_startup_retry(
    timeout_s: float,
    async_timeout_ms: float,
    max_frame_age_ms: float,
) -> None:
    """Recover from bounded V4L2 failures while waiting for startup frames.

    LeRobot's OpenCV camera warmup aborts on the first ``read()`` that returns
    false.  The two Logitech cameras can need several attempts after USB
    autosuspend, so retry only open failures and ``status=False`` reads during
    startup. A repeatedly failing capture is released and reopened. The same
    patch validates the asynchronous path before startup completes and gives
    later asynchronous reads enough time to outlast one bounded V4L2 read.
    """

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("camera startup timeout must be a positive finite number")
    if not math.isfinite(async_timeout_ms) or async_timeout_ms <= 0:
        raise ValueError("camera asynchronous timeout must be a positive finite number")
    if not math.isfinite(max_frame_age_ms) or max_frame_age_ms <= 0:
        raise ValueError("camera maximum frame age must be a positive finite number")
    if max_frame_age_ms >= async_timeout_ms:
        raise ValueError("camera maximum frame age must be shorter than the asynchronous timeout")

    try:
        v4l_select_timeout_s = int(os.environ[OPENCV_V4L_SELECT_TIMEOUT_ENV])
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{OPENCV_V4L_SELECT_TIMEOUT_ENV} must be a positive integer number of seconds"
        ) from error
    if v4l_select_timeout_s <= 0:
        raise ValueError(
            f"{OPENCV_V4L_SELECT_TIMEOUT_ENV} must be a positive integer number of seconds"
        )
    if v4l_select_timeout_s > timeout_s:
        raise ValueError(
            f"{OPENCV_V4L_SELECT_TIMEOUT_ENV} ({v4l_select_timeout_s}s) must not exceed "
            f"the camera startup timeout ({timeout_s:.1f}s)"
        )
    if async_timeout_ms < v4l_select_timeout_s * 1000:
        raise ValueError(
            f"camera asynchronous timeout ({async_timeout_ms:.0f}ms) must be at least the "
            f"V4L2 read timeout ({v4l_select_timeout_s * 1000}ms)"
        )

    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    original_connect = OpenCVCamera.connect
    original_read = OpenCVCamera.read
    original_async_read = OpenCVCamera.async_read
    if getattr(original_connect, "_keep_pose_startup_retry", False):
        return

    max_open_attempts = 100  # Defense only; the monotonic deadline is authoritative.
    max_failed_reads_per_open = 3
    reopen_delay_s = 1.0

    def release_capture(camera: Any) -> None:
        try:
            if getattr(camera, "thread", None) is not None:
                camera._stop_read_thread()
        except Exception:
            logging.exception("Could not stop the startup read thread for %s", camera)
        try:
            frame_lock = getattr(camera, "frame_lock", None)
            if frame_lock is not None:
                with frame_lock:
                    camera.latest_frame = None
                    camera.new_frame_event.clear()
                    if hasattr(camera, "_keep_pose_latest_hardware_frame"):
                        del camera._keep_pose_latest_hardware_frame
        except Exception:
            logging.exception("Could not clear stale startup frame state for %s", camera)
        try:
            if camera.videocapture is not None:
                camera.videocapture.release()
                camera.videocapture = None
        except Exception:
            logging.exception("Could not release the startup capture for %s", camera)

    def read_with_freshness_timestamp(self: Any, *args: Any, **kwargs: Any) -> Any:
        frame = original_read(self, *args, **kwargs)
        # Store frame identity and time together. The background thread publishes
        # this exact ndarray shortly afterwards; until then, an older cached
        # ndarray cannot accidentally receive the new frame's timestamp.
        self._keep_pose_latest_hardware_frame = (id(frame), time.monotonic())
        return frame

    def connect_with_startup_retry(self: Any, warmup: bool = True) -> None:
        if not warmup:
            original_connect(self, warmup=False)
            return

        started_at = time.monotonic()
        deadline = started_at + timeout_s
        failed_reads = 0
        failed_async_starts = 0
        open_attempts = 0
        last_error: BaseException | None = None

        while open_attempts < max_open_attempts and time.monotonic() < deadline:
            open_attempts += 1
            try:
                # Let LeRobot open and configure the camera, but replace its
                # fail-fast warmup with bounded reads and capture reopens.
                original_connect(self, warmup=False)
            except ConnectionError as error:
                release_capture(self)
                if "Failed to open" not in str(error):
                    raise
                last_error = error
            except BaseException:
                release_capture(self)
                raise
            else:
                first_frame_at: float | None = None
                consecutive_failed_reads = 0

                while time.monotonic() < deadline:
                    try:
                        original_read(self)
                    except RuntimeError as error:
                        if "read failed (status=" not in str(error):
                            release_capture(self)
                            raise
                        failed_reads += 1
                        consecutive_failed_reads += 1
                        first_frame_at = None
                        last_error = error
                        if consecutive_failed_reads >= max_failed_reads_per_open:
                            break
                    except BaseException:
                        release_capture(self)
                        raise
                    else:
                        consecutive_failed_reads = 0
                        now = time.monotonic()
                        if first_frame_at is None:
                            first_frame_at = now
                        if now - first_frame_at >= max(0.0, float(self.warmup_s)):
                            try:
                                # Start and validate the same asynchronous path
                                # used by get_observation before arming recording.
                                original_async_read(self, timeout_ms=async_timeout_ms)
                            except TimeoutError as error:
                                failed_async_starts += 1
                                last_error = error
                                break
                            except BaseException:
                                release_capture(self)
                                raise
                            else:
                                if failed_reads or failed_async_starts or open_attempts > 1:
                                    logging.info(
                                        "%s recovered after %d failed synchronous read(s), "
                                        "%d failed asynchronous start(s), and %d open attempt(s)",
                                        self,
                                        failed_reads,
                                        failed_async_starts,
                                        open_attempts,
                                    )
                                return

                    time.sleep(0.1)

                release_capture(self)

            if open_attempts < max_open_attempts and time.monotonic() < deadline:
                logging.warning(
                    "%s did not produce stable startup frames; reopening capture "
                    "(attempt %d, %.1fs remain)",
                    self,
                    open_attempts + 1,
                    max(0.0, deadline - time.monotonic()),
                )
                time.sleep(min(reopen_delay_s, max(0.0, deadline - time.monotonic())))

        release_capture(self)
        elapsed_s = time.monotonic() - started_at
        raise RuntimeError(
            f"{self} did not produce stable synchronous and asynchronous frames during the "
            f"{timeout_s:.1f}s startup window (elapsed {elapsed_s:.1f}s, "
            f"{open_attempts} open attempt(s), {failed_reads} failed synchronous read(s), "
            f"{failed_async_starts} failed asynchronous start(s)). "
            "Close any other camera viewer, reconnect the camera if needed, and retry."
        ) from last_error

    def async_read_with_v4l_timeout(self: Any, timeout_ms: float = 200) -> Any:
        effective_timeout_ms = max(float(timeout_ms), async_timeout_ms)

        # The Logitech pair physically delivers roughly 15 fresh frames/s even
        # when configured for 30 FPS. Return a recent cached frame immediately
        # so the 30 Hz robot/action loop is not throttled to camera cadence.
        # If frames stop updating, wait for a genuinely new one and fail after
        # the bounded asynchronous timeout instead of masking a dead camera.
        if self.is_connected and self.thread is not None and self.thread.is_alive():
            with self.frame_lock:
                hardware_frame = getattr(self, "_keep_pose_latest_hardware_frame", None)
                if (
                    self.latest_frame is not None
                    and hardware_frame is not None
                    and id(self.latest_frame) == hardware_frame[0]
                    and (time.monotonic() - hardware_frame[1]) * 1000 <= max_frame_age_ms
                ):
                    return self.latest_frame
                self.new_frame_event.clear()

        return original_async_read(self, timeout_ms=effective_timeout_ms)

    connect_with_startup_retry._keep_pose_startup_retry = True
    read_with_freshness_timestamp._keep_pose_freshness_timestamp = True
    async_read_with_v4l_timeout._keep_pose_async_timeout = True
    OpenCVCamera.connect = connect_with_startup_retry
    OpenCVCamera.read = read_with_freshness_timestamp
    OpenCVCamera.async_read = async_read_with_v4l_timeout


def install_recording_control_guard(min_episode_time_s: float) -> None:
    """Make keyboard episode transitions release-aware and safe to save.

    LeRobot represents keyboard input with shared booleans.  A held Right Arrow
    can therefore end the current recording/reset, repeat during the transition,
    and make the next recording loop exit before it adds its first frame.  The
    stock recorder then unconditionally calls ``save_episode`` on an empty
    buffer and crashes.

    This process-local patch ignores key auto-repeat until release, tracks
    whether the recorder is recording or resetting, and refuses to accept an
    episode before a small minimum duration.  Escape discards a partial episode
    during recording but preserves an already accepted episode during reset.
    """

    if not math.isfinite(min_episode_time_s) or min_episode_time_s <= 0:
        raise ValueError("minimum episode time must be a positive finite number")

    from lerobot.scripts import lerobot_record as record_module

    original_record_loop = record_module.record_loop
    if getattr(original_record_loop, "_keep_pose_control_guard", False):
        return

    def init_phase_safe_keyboard_listener() -> tuple[Any | None, dict[str, Any]]:
        events: dict[str, Any] = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "_phase": "starting",
            "_last_key": None,
            "_control_lock": threading.Lock(),
        }

        if record_module.is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras and keyboard controls are unavailable."
            )
            return None, events

        from pynput import keyboard

        pressed_keys: set[Any] = set()

        def on_press(key: Any) -> None:
            try:
                with events["_control_lock"]:
                    # pynput emits repeated on_press callbacks while a key is held.
                    # One physical press must trigger at most one phase transition.
                    if key in pressed_keys:
                        return
                    pressed_keys.add(key)

                    phase = events["_phase"]
                    if key == keyboard.Key.esc:
                        events["_last_key"] = "escape"
                        events["stop_recording"] = True
                        events["exit_early"] = True
                        if phase == "recording":
                            events["rerecord_episode"] = True
                            print(
                                "Escape pressed while recording: discarding the partial episode and stopping..."
                            )
                        else:
                            print("Escape pressed: stopping after preserving the last accepted episode...")
                        return

                    if phase not in {"recording", "reset"}:
                        if key in {keyboard.Key.right, keyboard.Key.left}:
                            if phase == "saving":
                                print(
                                    f"Ignoring {key} while episode data/video is being saved. "
                                    "Saving cannot be skipped safely; release the key and wait."
                                )
                            else:
                                print(
                                    f"Ignoring {key} during {phase}; release it and press again "
                                    "after recording or reset begins."
                                )
                        return

                    if key == keyboard.Key.right:
                        events["_last_key"] = "right"
                        events["exit_early"] = True
                        if phase == "recording":
                            print("Right Arrow pressed: accepting this episode...")
                        else:
                            print("Right Arrow pressed: reset complete; continuing...")
                    elif key == keyboard.Key.left:
                        events["_last_key"] = "left"
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                        print("Left Arrow pressed: discarding and re-recording this episode...")
            except Exception:
                logging.exception("Error handling recording key press")

        def on_release(key: Any) -> None:
            with events["_control_lock"]:
                pressed_keys.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        return listener, events

    def episode_buffer_size(dataset: Any) -> int:
        episode_buffer = getattr(dataset, "episode_buffer", None)
        if not isinstance(episode_buffer, dict):
            return 0
        return int(episode_buffer.get("size", 0))

    def guarded_record_loop(*args: Any, **kwargs: Any) -> Any:
        events = kwargs.get("events")
        dataset = kwargs.get("dataset")
        if not isinstance(events, dict):
            return original_record_loop(*args, **kwargs)

        phase = "recording" if dataset is not None else "reset"
        control_lock = events.setdefault("_control_lock", threading.Lock())
        with control_lock:
            events["_phase"] = phase
            events["_last_key"] = None
            if not events.get("stop_recording") and not events.get("rerecord_episode"):
                events["exit_early"] = False

        try:
            if dataset is None:
                result = original_record_loop(*args, **kwargs)
                events["_last_key"] = None
                return result

            fps = int(kwargs.get("fps", 0))
            if fps <= 0:
                raise ValueError("recording FPS must be positive")
            minimum_frames = max(1, math.ceil(min_episode_time_s * fps))

            previous_frame_count = episode_buffer_size(dataset)
            while True:
                result = original_record_loop(*args, **kwargs)
                frame_count = episode_buffer_size(dataset)

                # Decide and consume the listener event atomically. A key press
                # arriving after this block remains pending for the next loop;
                # it is never erased by timeout/early-press cleanup.
                with control_lock:
                    last_key = events.get("_last_key")
                    if events.get("rerecord_episode") or events.get("stop_recording"):
                        decision = "discard_or_stop"
                    elif last_key == "right" and frame_count >= minimum_frames:
                        events["_last_key"] = None
                        events["exit_early"] = False
                        decision = "accept"
                    elif last_key == "right":
                        events["exit_early"] = False
                        events["_last_key"] = None
                        events["_phase"] = "recording"
                        decision = "early_right"
                    elif last_key is not None or frame_count <= previous_frame_count:
                        raise RuntimeError(
                            "Recording loop ended without an accept/discard command or new frames "
                            f"({previous_frame_count} -> {frame_count}); refusing to save an invalid episode."
                        )
                    else:
                        # A natural time-window boundary. Keep recording, but do
                        # not clear a key that arrives after this atomic block.
                        events["exit_early"] = False
                        events["_last_key"] = None
                        events["_phase"] = "recording"
                        decision = "time_window"

                if decision in {"discard_or_stop", "accept"}:
                    return result

                if decision == "early_right":
                    logging.warning(
                        "Ignored an early Right Arrow at %d/%d frames; recording continues",
                        frame_count,
                        minimum_frames,
                    )
                    print(
                        f"Episode is too short to accept ({frame_count}/{minimum_frames} frames). "
                        "Release Right Arrow; recording is still active."
                    )
                    previous_frame_count = frame_count
                    continue

                # Reaching episode_time_s is only a time-window boundary, not
                # proof that the task succeeded. Continue appending to the same
                # episode until the operator deliberately accepts or discards it.
                logging.warning(
                    "Episode reached its %.1fs recording window at %d frames without Right Arrow; continuing",
                    float(kwargs.get("control_time_s", 0)),
                    frame_count,
                )
                print(
                    "Episode time window ended, but no accept key was pressed. "
                    "Recording continues; tap Right Arrow only after the task is complete."
                )
                previous_frame_count = frame_count
        finally:
            # Make the phase change and stale-Right-Arrow cleanup atomic with
            # the listener callback. A press arriving after this point is
            # ignored as a transition press instead of leaking into the next
            # record/reset loop.
            with control_lock:
                events["_phase"] = "saving"
                if not events.get("stop_recording") and not events.get("rerecord_episode"):
                    events["exit_early"] = False
                    events["_last_key"] = None

    guarded_record_loop._keep_pose_control_guard = True
    record_module.init_keyboard_listener = init_phase_safe_keyboard_listener
    record_module.record_loop = guarded_record_loop


def main() -> None:
    register_third_party_devices()
    max_step = float(os.environ.get(MAX_STEP_ENV, "3.0"))
    camera_startup_timeout = float(os.environ.get(CAMERA_STARTUP_TIMEOUT_ENV, "12.0"))
    camera_async_timeout = float(os.environ.get(CAMERA_ASYNC_TIMEOUT_ENV, "1500"))
    camera_max_frame_age = float(os.environ.get(CAMERA_MAX_FRAME_AGE_ENV, "250"))
    starai_read_timeout = float(os.environ.get(STARAI_READ_TIMEOUT_ENV, "1.0"))
    min_episode_time = float(os.environ.get(MIN_EPISODE_TIME_ENV, "5.0"))
    install_starai_position_read_retry(timeout_s=starai_read_timeout)
    _, cleanup_connected_devices = install_keep_pose_recording_patch(max_step=max_step)
    install_opencv_startup_retry(
        timeout_s=camera_startup_timeout,
        async_timeout_ms=camera_async_timeout,
        max_frame_age_ms=camera_max_frame_age,
    )
    install_recording_control_guard(min_episode_time_s=min_episode_time)
    logging.info(
        "Keep-pose recording enabled: max step %.3f, camera startup timeout %.2fs, "
        "camera async timeout %.0fms, maximum frame age %.0fms, V4L2 read timeout %ss, "
        "StarAI read timeout %.2fs, minimum episode time %.2fs",
        max_step,
        camera_startup_timeout,
        camera_async_timeout,
        camera_max_frame_age,
        os.environ[OPENCV_V4L_SELECT_TIMEOUT_ENV],
        starai_read_timeout,
        min_episode_time,
    )

    from lerobot.scripts.lerobot_record import main as record_main

    try:
        record_main()
    finally:
        cleanup_connected_devices()


if __name__ == "__main__":
    main()
