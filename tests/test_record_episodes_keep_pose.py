from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest

from lerobot.scripts import lerobot_record
from scripts import record_episodes_keep_pose as keep_pose


class FakeDataset:
    def __init__(self) -> None:
        self.episode_buffer = {"size": 0}


class FakeCapture:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += max(0.0, duration)


def install_guard(monkeypatch: pytest.MonkeyPatch, fake_record_loop, min_time_s: float = 5.0):
    monkeypatch.setattr(lerobot_record, "record_loop", fake_record_loop)
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda: (None, {}))
    keep_pose.install_recording_control_guard(min_episode_time_s=min_time_s)
    return lerobot_record.record_loop


def install_camera_retry(
    monkeypatch: pytest.MonkeyPatch,
    fake_connect,
    fake_read,
    fake_async_read=None,
    timeout_s: float = 12.0,
):
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    monkeypatch.setenv(keep_pose.OPENCV_V4L_SELECT_TIMEOUT_ENV, "1")
    clock = FakeClock()
    monkeypatch.setattr(keep_pose.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(keep_pose.time, "sleep", clock.sleep)
    monkeypatch.setattr(OpenCVCamera, "connect", fake_connect)
    monkeypatch.setattr(OpenCVCamera, "read", fake_read)
    if fake_async_read is None:
        fake_async_read = lambda _camera, timeout_ms=200: None
    monkeypatch.setattr(OpenCVCamera, "async_read", fake_async_read)
    keep_pose.install_opencv_startup_retry(
        timeout_s=timeout_s,
        async_timeout_ms=1500,
        max_frame_age_ms=250,
    )
    return OpenCVCamera.connect, OpenCVCamera.async_read, clock


def install_keep_pose_patch(
    monkeypatch: pytest.MonkeyPatch,
    fake_robot_connect,
    fake_teleop_connect=None,
):
    """Install the process-local robot patch with every changed class method restored by pytest."""

    from lerobot_robot_viola.starai_viola import StaraiViola
    from lerobot_teleoperator_violin.starai_violin import StaraiViolin

    monkeypatch.setattr(StaraiViola, "connect", fake_robot_connect)
    monkeypatch.setattr(StaraiViola, "get_action", lambda _robot: {})
    monkeypatch.setattr(StaraiViola, "move_to_initial_position", lambda _robot: {})
    monkeypatch.setattr(StaraiViolin, "get_action", lambda _teleop: {})
    if fake_teleop_connect is None:
        fake_teleop_connect = lambda _teleop, calibrate=True: None
    monkeypatch.setattr(StaraiViolin, "connect", fake_teleop_connect)
    monkeypatch.setattr(StaraiViolin, "move_to_initial_position", lambda _teleop: {})
    _, cleanup = keep_pose.install_keep_pose_recording_patch(max_step=3.0)
    return StaraiViola.connect, StaraiViolin.connect, cleanup


class FakeRollbackCamera:
    def __init__(
        self,
        *,
        is_connected: bool = False,
        thread=None,
        videocapture: FakeCapture | None = None,
        disconnect_error: Exception | None = None,
    ) -> None:
        self.is_connected = is_connected
        self.thread = thread
        self.videocapture = videocapture
        self.disconnect_error = disconnect_error
        self.disconnect_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error


class FakeRollbackBus:
    def __init__(self, *, is_connected: bool, disconnect_error: Exception | None = None) -> None:
        self.is_connected = is_connected
        self.disconnect_error = disconnect_error
        self.disconnect_calls: list[bool] = []

    def disconnect(self, disable_torque: bool) -> None:
        self.disconnect_calls.append(disable_torque)
        if self.disconnect_error is not None:
            raise self.disconnect_error


def test_robot_connect_failure_rolls_back_partial_cameras_and_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_error = RuntimeError("up camera failed to start")
    connect_calibrate_values: list[bool] = []

    def fake_robot_connect(_robot, calibrate=True):
        connect_calibrate_values.append(calibrate)
        raise startup_error

    connect, _, _ = install_keep_pose_patch(monkeypatch, fake_robot_connect)
    connected_camera = FakeRollbackCamera(is_connected=True)
    threaded_camera = FakeRollbackCamera(thread=object())
    unopened_capture = FakeCapture()
    partially_open_camera = FakeRollbackCamera(videocapture=unopened_capture)
    untouched_camera = FakeRollbackCamera()
    bus = FakeRollbackBus(is_connected=True)
    robot = SimpleNamespace(
        cameras={
            "connected": connected_camera,
            "threaded": threaded_camera,
            "partial": partially_open_camera,
            "untouched": untouched_camera,
        },
        bus=bus,
        config=SimpleNamespace(disable_torque_on_disconnect=True),
    )

    with pytest.raises(RuntimeError) as caught:
        connect(robot, calibrate=False)

    assert caught.value is startup_error
    assert connect_calibrate_values == [False]
    assert connected_camera.disconnect_calls == 1
    assert threaded_camera.disconnect_calls == 1
    assert partially_open_camera.disconnect_calls == 0
    assert unopened_capture.released is True
    assert partially_open_camera.videocapture is None
    assert untouched_camera.disconnect_calls == 0
    assert bus.disconnect_calls == [True]


def test_robot_connect_cleanup_errors_do_not_hide_startup_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    startup_error = RuntimeError("camera startup failed")

    def fake_robot_connect(_robot, calibrate=True):
        raise startup_error

    connect, _, _ = install_keep_pose_patch(monkeypatch, fake_robot_connect)
    camera = FakeRollbackCamera(
        is_connected=True,
        disconnect_error=RuntimeError("camera cleanup failed"),
    )
    bus = FakeRollbackBus(
        is_connected=True,
        disconnect_error=RuntimeError("bus cleanup failed"),
    )
    robot = SimpleNamespace(
        cameras={"front": camera},
        bus=bus,
        config=SimpleNamespace(disable_torque_on_disconnect=True),
    )

    with pytest.raises(RuntimeError) as caught:
        connect(robot)

    assert caught.value is startup_error
    assert camera.disconnect_calls == 1
    assert bus.disconnect_calls == [True]
    assert "Could not release" in caplog.text
    assert "Could not disconnect the follower bus" in caplog.text


def test_robot_connect_bus_cleanup_failure_force_closes_port_without_hiding_startup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_error = RuntimeError("camera startup failed")
    close_port_calls = 0

    def fake_robot_connect(_robot, calibrate=True):
        raise startup_error

    def close_port() -> None:
        nonlocal close_port_calls
        close_port_calls += 1

    connect, _, _ = install_keep_pose_patch(monkeypatch, fake_robot_connect)
    bus = FakeRollbackBus(
        is_connected=True,
        disconnect_error=RuntimeError("torque-disable cleanup failed"),
    )
    bus.port_handler = SimpleNamespace(closePort=close_port)
    robot = SimpleNamespace(
        cameras={},
        bus=bus,
        config=SimpleNamespace(disable_torque_on_disconnect=True),
    )

    with pytest.raises(RuntimeError) as caught:
        connect(robot)

    assert caught.value is startup_error
    assert bus.disconnect_calls == [True]
    assert close_port_calls == 1


def test_robot_connect_success_does_not_run_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_robot_connect(_robot, calibrate=True):
        return None

    connect, _, cleanup = install_keep_pose_patch(monkeypatch, fake_robot_connect)
    camera = FakeRollbackCamera(is_connected=True)
    bus = FakeRollbackBus(is_connected=True)
    robot = SimpleNamespace(
        cameras={"front": camera},
        bus=bus,
        config=SimpleNamespace(disable_torque_on_disconnect=True),
    )

    connect(robot)

    assert camera.disconnect_calls == 0
    assert bus.disconnect_calls == []

    cleanup()

    assert camera.disconnect_calls == 1
    assert bus.disconnect_calls == [True]


def test_teacher_is_cleaned_after_successful_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    _, teleop_connect, cleanup = install_keep_pose_patch(
        monkeypatch,
        lambda _robot, calibrate=True: None,
    )
    bus = FakeRollbackBus(is_connected=True)
    teleop = SimpleNamespace(bus=bus)

    teleop_connect(teleop)
    assert bus.disconnect_calls == []

    cleanup()
    assert bus.disconnect_calls == [True]


def test_teacher_connect_failure_rolls_back_without_hiding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_error = RuntimeError("teacher startup failed")

    def fail_teleop_connect(_teleop, calibrate=True):
        raise startup_error

    _, teleop_connect, _ = install_keep_pose_patch(
        monkeypatch,
        lambda _robot, calibrate=True: None,
        fail_teleop_connect,
    )
    bus = FakeRollbackBus(is_connected=True)
    teleop = SimpleNamespace(bus=bus)

    with pytest.raises(RuntimeError) as caught:
        teleop_connect(teleop)

    assert caught.value is startup_error
    assert bus.disconnect_calls == [True]


def test_main_cleans_up_connected_devices_when_recording_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_error = RuntimeError("recording failed")
    cleanup_calls = 0

    def cleanup_connected_devices() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    def fail_recording() -> None:
        raise recording_error

    monkeypatch.setattr(keep_pose, "register_third_party_devices", lambda: None)
    monkeypatch.setattr(keep_pose, "install_starai_position_read_retry", lambda **_kwargs: None)
    monkeypatch.setattr(
        keep_pose,
        "install_keep_pose_recording_patch",
        lambda **_kwargs: (SimpleNamespace(), cleanup_connected_devices),
    )
    monkeypatch.setattr(keep_pose, "install_opencv_startup_retry", lambda **_kwargs: None)
    monkeypatch.setattr(keep_pose, "install_recording_control_guard", lambda **_kwargs: None)
    monkeypatch.setattr(lerobot_record, "main", fail_recording)

    with pytest.raises(RuntimeError) as caught:
        keep_pose.main()

    assert caught.value is recording_error
    assert cleanup_calls == 1


def test_camera_startup_recovers_after_transient_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    open_count = 0
    read_count = 0

    def fake_connect(camera, warmup=True):
        nonlocal open_count
        open_count += 1
        camera.videocapture = FakeCapture()

    def fake_read(camera):
        nonlocal read_count
        read_count += 1
        if read_count <= 2:
            raise RuntimeError(f"{camera} read failed (status=False).")

    connect, _, _ = install_camera_retry(monkeypatch, fake_connect, fake_read)
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    connect(camera)

    assert open_count == 1
    assert read_count == 3
    assert camera.videocapture is not None
    assert camera.videocapture.released is False


def test_camera_startup_reopens_wedged_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[FakeCapture] = []
    read_count = 0

    def fake_connect(camera, warmup=True):
        capture = FakeCapture()
        captures.append(capture)
        camera.videocapture = capture

    def fake_read(camera):
        nonlocal read_count
        read_count += 1
        if read_count <= 3:
            raise RuntimeError(f"{camera} read failed (status=False).")

    connect, _, _ = install_camera_retry(monkeypatch, fake_connect, fake_read)
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    connect(camera)

    assert len(captures) == 2
    assert captures[0].released is True
    assert captures[1].released is False


def test_camera_startup_retries_failed_open_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    open_count = 0

    def fake_connect(camera, warmup=True):
        nonlocal open_count
        open_count += 1
        if open_count <= 2:
            raise ConnectionError(f"Failed to open {camera}")
        camera.videocapture = FakeCapture()

    connect, _, clock = install_camera_retry(monkeypatch, fake_connect, lambda _camera: None)
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    connect(camera)

    assert open_count == 3
    assert clock.now == pytest.approx(2.0)
    assert camera.videocapture is not None


def test_camera_async_start_failure_reopens_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[FakeCapture] = []
    async_calls = 0
    stale_frame = object()

    def fake_connect(camera, warmup=True):
        if captures:
            assert camera.latest_frame is None
            assert not camera.new_frame_event.is_set()
            assert not hasattr(camera, "_keep_pose_latest_hardware_frame")
        capture = FakeCapture()
        captures.append(capture)
        camera.videocapture = capture

    def fake_async_read(camera, timeout_ms=200):
        nonlocal async_calls
        async_calls += 1
        if async_calls == 1:
            # Match the state left by OpenCVCamera.async_read when its first
            # background frame times out. None of it may validate a reopen.
            camera.thread = object()
            camera.latest_frame = stale_frame
            camera.new_frame_event.set()
            camera._keep_pose_latest_hardware_frame = (id(stale_frame), 0.0)
            raise TimeoutError("first async frame timed out")

    connect, _, _ = install_camera_retry(
        monkeypatch,
        fake_connect,
        lambda _camera: None,
        fake_async_read,
    )
    camera = SimpleNamespace(
        videocapture=None,
        warmup_s=0.0,
        thread=None,
        frame_lock=threading.Lock(),
        latest_frame=None,
        new_frame_event=threading.Event(),
    )

    def stop_read_thread() -> None:
        camera.thread = None

    camera._stop_read_thread = stop_read_thread

    connect(camera)

    assert len(captures) == 2
    assert captures[0].released is True
    assert captures[1].released is False


def test_camera_stability_timer_resets_after_failed_read(monkeypatch: pytest.MonkeyPatch) -> None:
    read_count = 0
    async_call_times: list[float] = []

    def fake_connect(camera, warmup=True):
        camera.videocapture = FakeCapture()

    def fake_read(camera):
        nonlocal read_count
        read_count += 1
        if read_count == 3:
            raise RuntimeError(f"{camera} read failed (status=False).")

    def fake_async_read(_camera, timeout_ms=200):
        async_call_times.append(clock.now)

    connect, _, clock = install_camera_retry(
        monkeypatch,
        fake_connect,
        fake_read,
        fake_async_read,
    )
    camera = SimpleNamespace(videocapture=None, warmup_s=0.3, thread=None)

    connect(camera)

    assert read_count >= 7
    assert async_call_times[0] >= 0.6


def test_camera_async_reads_use_v4l_compatible_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_timeouts: list[float] = []

    def fake_connect(camera, warmup=True):
        camera.videocapture = FakeCapture()

    def fake_async_read(_camera, timeout_ms=200):
        requested_timeouts.append(timeout_ms)

    connect, async_read, _ = install_camera_retry(
        monkeypatch,
        fake_connect,
        lambda _camera: None,
        fake_async_read,
    )
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    connect(camera)
    camera.is_connected = True
    camera.thread = SimpleNamespace(is_alive=lambda: True)
    camera.frame_lock = threading.Lock()
    camera.latest_frame = None
    camera.new_frame_event = threading.Event()
    async_read(camera, timeout_ms=200)
    async_read(camera, timeout_ms=2000)

    assert requested_timeouts == [1500, 1500, 2000]


def test_camera_async_read_reuses_only_recent_cached_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_timeouts: list[float] = []

    def fake_connect(camera, warmup=True):
        camera.videocapture = FakeCapture()

    def fake_async_read(_camera, timeout_ms=200):
        requested_timeouts.append(timeout_ms)
        return "new-frame"

    connect, async_read, clock = install_camera_retry(
        monkeypatch,
        fake_connect,
        lambda _camera: None,
        fake_async_read,
    )
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)
    connect(camera)
    camera.is_connected = True
    camera.thread = SimpleNamespace(is_alive=lambda: True)
    camera.frame_lock = threading.Lock()
    camera.latest_frame = "cached-frame"
    camera.new_frame_event = threading.Event()
    camera._keep_pose_latest_hardware_frame = (id(camera.latest_frame), clock.now)

    assert async_read(camera) == "cached-frame"
    assert requested_timeouts == [1500]

    clock.sleep(0.251)
    assert async_read(camera) == "new-frame"
    assert requested_timeouts == [1500, 1500]


def test_camera_async_read_clears_stale_event_before_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = 0

    def fake_connect(camera, warmup=True):
        camera.videocapture = FakeCapture()

    def fake_async_read(camera, timeout_ms=200):
        nonlocal delegated
        delegated += 1
        assert not camera.new_frame_event.is_set()
        return "fresh-frame"

    connect, async_read, clock = install_camera_retry(
        monkeypatch,
        fake_connect,
        lambda _camera: None,
        fake_async_read,
    )
    camera = SimpleNamespace(
        videocapture=None,
        warmup_s=0.0,
        thread=None,
        new_frame_event=threading.Event(),
    )
    connect(camera)
    camera.is_connected = True
    camera.thread = SimpleNamespace(is_alive=lambda: True)
    camera.frame_lock = threading.Lock()
    camera.latest_frame = "stale-frame"
    camera.new_frame_event = threading.Event()
    camera.new_frame_event.set()
    camera._keep_pose_latest_hardware_frame = (
        id(camera.latest_frame),
        clock.now - 0.251,
    )

    assert async_read(camera) == "fresh-frame"
    assert delegated == 2  # startup validation plus the stale-frame fallback


def test_camera_read_publishes_frame_identity_and_timestamp_only_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    frame = object()
    should_fail = False

    def fake_read(_camera):
        if should_fail:
            raise RuntimeError("read failed (status=False).")
        return frame

    _, _, clock = install_camera_retry(
        monkeypatch,
        lambda _camera, warmup=True: None,
        fake_read,
    )
    camera = SimpleNamespace()

    assert OpenCVCamera.read(camera) is frame
    assert camera._keep_pose_latest_hardware_frame == (id(frame), clock.now)

    del camera._keep_pose_latest_hardware_frame
    should_fail = True
    with pytest.raises(RuntimeError, match="status=False"):
        OpenCVCamera.read(camera)
    assert not hasattr(camera, "_keep_pose_latest_hardware_frame")


def test_camera_startup_failure_is_bounded_and_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[FakeCapture] = []
    read_count = 0

    def fake_connect(camera, warmup=True):
        capture = FakeCapture()
        captures.append(capture)
        camera.videocapture = capture

    def fake_read(camera):
        nonlocal read_count
        read_count += 1
        raise RuntimeError(f"{camera} read failed (status=False).")

    connect, _, clock = install_camera_retry(monkeypatch, fake_connect, fake_read)
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    with pytest.raises(RuntimeError, match="startup window"):
        connect(camera)

    assert len(captures) >= 3
    assert (len(captures) - 1) * 3 < read_count <= len(captures) * 3
    assert clock.now >= 12.0
    assert all(capture.released for capture in captures)
    assert camera.videocapture is None


def test_camera_startup_does_not_hide_unrelated_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture()

    def fake_connect(camera, warmup=True):
        camera.videocapture = capture

    def fake_read(camera):
        raise RuntimeError("unexpected frame dimensions")

    connect, _, _ = install_camera_retry(monkeypatch, fake_connect, fake_read)
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    with pytest.raises(RuntimeError, match="unexpected frame dimensions"):
        connect(camera)

    assert capture.released is True
    assert camera.videocapture is None


def test_camera_startup_warmup_false_delegates_without_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_count = 0

    def fake_connect(camera, warmup=True):
        nonlocal open_count
        open_count += 1
        camera.videocapture = FakeCapture()

    def fake_read(camera):
        raise AssertionError("read must not be called when warmup=False")

    connect, _, _ = install_camera_retry(monkeypatch, fake_connect, fake_read)
    camera = SimpleNamespace(videocapture=None, warmup_s=0.0, thread=None)

    connect(camera, warmup=False)

    assert open_count == 1


@pytest.mark.parametrize("value", ["0", "-1", "one", "1.5"])
def test_camera_startup_rejects_invalid_v4l_timeout(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    monkeypatch.setenv(keep_pose.OPENCV_V4L_SELECT_TIMEOUT_ENV, value)
    monkeypatch.setattr(OpenCVCamera, "connect", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match=keep_pose.OPENCV_V4L_SELECT_TIMEOUT_ENV):
        keep_pose.install_opencv_startup_retry(
            timeout_s=12.0,
            async_timeout_ms=1500,
            max_frame_age_ms=250,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), 999.0])
def test_camera_startup_rejects_invalid_async_timeout(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    monkeypatch.setenv(keep_pose.OPENCV_V4L_SELECT_TIMEOUT_ENV, "1")
    monkeypatch.setattr(OpenCVCamera, "connect", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="asynchronous timeout"):
        keep_pose.install_opencv_startup_retry(
            timeout_s=12.0,
            async_timeout_ms=value,
            max_frame_age_ms=250,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), 1500.0])
def test_camera_startup_rejects_invalid_max_frame_age(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    monkeypatch.setenv(keep_pose.OPENCV_V4L_SELECT_TIMEOUT_ENV, "1")
    monkeypatch.setattr(OpenCVCamera, "connect", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="maximum frame age"):
        keep_pose.install_opencv_startup_retry(
            timeout_s=12.0,
            async_timeout_ms=1500,
            max_frame_age_ms=value,
        )


def test_control_guard_continues_after_early_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = FakeDataset()
    calls = 0

    def fake_record_loop(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            dataset.episode_buffer["size"] = 10
            kwargs["events"]["_last_key"] = "right"
        else:
            dataset.episode_buffer["size"] = 150
            # The first press was explicitly rejected as too early, so accepting
            # still requires one new press after the minimum duration.
            kwargs["events"]["_last_key"] = "right"

    guarded = install_guard(monkeypatch, fake_record_loop)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    guarded(dataset=dataset, events=events, fps=30, control_time_s=60)

    assert calls == 2
    assert dataset.episode_buffer["size"] == 150
    assert events["_phase"] == "saving"
    assert events["exit_early"] is False
    assert events["_last_key"] is None


def test_control_guard_requires_right_arrow_after_natural_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full time window is not proof that the manipulation was completed."""

    dataset = FakeDataset()
    calls = 0

    def fake_record_loop(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Simulate the configured episode_time_s expiring without a key press.
            dataset.episode_buffer["size"] = 900
        else:
            # A later, deliberate Right Arrow is the only accept command.
            dataset.episode_buffer["size"] = 930
            kwargs["events"]["_last_key"] = "right"

    guarded = install_guard(monkeypatch, fake_record_loop)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    guarded(dataset=dataset, events=events, fps=30, control_time_s=30)

    assert calls == 2
    assert dataset.episode_buffer["size"] == 930
    assert events["_phase"] == "saving"


def test_control_guard_preserves_right_arrow_arriving_at_time_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    calls = 0

    def fake_record_loop(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            dataset.episode_buffer["size"] = 900
        else:
            # A press arriving just after timeout handling must still be pending.
            assert kwargs["events"]["_last_key"] == "right"
            assert kwargs["events"]["exit_early"] is True

    def inject_right_after_timeout(message, *args, **kwargs):
        if "recording window" in message:
            events["_last_key"] = "right"
            events["exit_early"] = True

    monkeypatch.setattr(keep_pose.logging, "warning", inject_right_after_timeout)
    guarded = install_guard(monkeypatch, fake_record_loop)

    guarded(dataset=dataset, events=events, fps=30, control_time_s=30)

    assert calls == 2
    assert events["_phase"] == "saving"


def test_control_guard_clears_late_right_arrow_at_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = FakeDataset()

    def fake_record_loop(*args, **kwargs):
        dataset.episode_buffer["size"] = 150
        # Simulate a Right Arrow callback racing with natural loop completion.
        kwargs["events"]["_last_key"] = "right"
        kwargs["events"]["exit_early"] = True

    guarded = install_guard(monkeypatch, fake_record_loop)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    guarded(dataset=dataset, events=events, fps=30, control_time_s=60)

    assert events["_phase"] == "saving"
    assert events["exit_early"] is False
    assert events["_last_key"] is None


def test_control_guard_refuses_unexplained_empty_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = FakeDataset()
    guarded = install_guard(monkeypatch, lambda *args, **kwargs: None)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    with pytest.raises(RuntimeError, match="refusing to save an invalid episode"):
        guarded(dataset=dataset, events=events, fps=30, control_time_s=60)


def test_control_guard_refuses_natural_return_without_new_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset()
    dataset.episode_buffer["size"] = 150
    guarded = install_guard(monkeypatch, lambda *args, **kwargs: None)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    with pytest.raises(RuntimeError, match="without an accept/discard command or new frames"):
        guarded(dataset=dataset, events=events, fps=30, control_time_s=60)


@pytest.mark.parametrize("event_name", ["rerecord_episode", "stop_recording"])
def test_control_guard_allows_discard_or_stop_without_frames(
    monkeypatch: pytest.MonkeyPatch, event_name: str
) -> None:
    dataset = FakeDataset()

    def fake_record_loop(*args, **kwargs):
        kwargs["events"][event_name] = True

    guarded = install_guard(monkeypatch, fake_record_loop)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    guarded(dataset=dataset, events=events, fps=30, control_time_s=60)


def test_keyboard_ignores_auto_repeat_until_release(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeKey:
        right = "right"
        left = "left"
        esc = "esc"

    class FakeListener:
        def __init__(self, on_press, on_release):
            self.on_press = on_press
            self.on_release = on_release

        def start(self):
            return None

    fake_keyboard = SimpleNamespace(Key=FakeKey, Listener=FakeListener)
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=fake_keyboard))
    monkeypatch.setattr(lerobot_record, "is_headless", lambda: False)
    install_guard(monkeypatch, lambda *args, **kwargs: None)

    listener, events = lerobot_record.init_keyboard_listener()
    events["_phase"] = "recording"
    listener.on_press(FakeKey.right)
    assert events["exit_early"] is True

    # Simulate record_loop consuming the flag while the physical key remains held.
    events["exit_early"] = False
    listener.on_press(FakeKey.right)
    assert events["exit_early"] is False

    listener.on_release(FakeKey.right)
    listener.on_press(FakeKey.right)
    assert events["exit_early"] is True


def test_keyboard_transition_press_cannot_leak_into_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKey:
        right = "right"
        left = "left"
        esc = "esc"

    class FakeListener:
        def __init__(self, on_press, on_release):
            self.on_press = on_press
            self.on_release = on_release

        def start(self):
            return None

    fake_keyboard = SimpleNamespace(Key=FakeKey, Listener=FakeListener)
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=fake_keyboard))
    monkeypatch.setattr(lerobot_record, "is_headless", lambda: False)
    install_guard(monkeypatch, lambda *args, **kwargs: None)

    listener, events = lerobot_record.init_keyboard_listener()

    # A press that begins while saving/encoding must not become a reset skip.
    events["_phase"] = "saving"
    listener.on_press(FakeKey.right)
    assert events["exit_early"] is False

    events["_phase"] = "reset"
    listener.on_press(FakeKey.right)
    assert events["exit_early"] is False

    # Once released, one new deliberate press skips reset.
    listener.on_release(FakeKey.right)
    listener.on_press(FakeKey.right)
    assert events["exit_early"] is True
    assert events["_last_key"] == "right"


def test_escape_discards_only_during_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeKey:
        right = "right"
        left = "left"
        esc = "esc"

    class FakeListener:
        def __init__(self, on_press, on_release):
            self.on_press = on_press
            self.on_release = on_release

        def start(self):
            return None

    fake_keyboard = SimpleNamespace(Key=FakeKey, Listener=FakeListener)
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=fake_keyboard))
    monkeypatch.setattr(lerobot_record, "is_headless", lambda: False)
    install_guard(monkeypatch, lambda *args, **kwargs: None)

    listener, events = lerobot_record.init_keyboard_listener()
    events["_phase"] = "recording"
    listener.on_press(FakeKey.esc)
    assert events["stop_recording"] is True
    assert events["rerecord_episode"] is True

    listener, events = lerobot_record.init_keyboard_listener()
    events["_phase"] = "reset"
    listener.on_press(FakeKey.esc)
    assert events["stop_recording"] is True
    assert events["rerecord_episode"] is False


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_control_guard_rejects_invalid_minimum_time(value: float) -> None:
    with pytest.raises(ValueError, match="minimum episode time"):
        keep_pose.install_recording_control_guard(value)
