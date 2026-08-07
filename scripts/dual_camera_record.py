#!/usr/bin/env python3
"""Preview two cameras and record synchronized five-minute MP4 videos."""

from __future__ import annotations

import argparse
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageTk


class CameraReader:
    """Continuously capture the newest frame from one V4L2 camera."""

    def __init__(self, name: str, device: str, width: int, height: int, fps: int) -> None:
        self.name = name
        self.device = device
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open {name} camera at {device}")

        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame: Any | None = None
        self.frame_number = 0
        self.frame_time: float | None = None
        self.read_failures = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, name=f"camera-{name}", daemon=True)

        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((actual_fourcc >> (8 * index)) & 0xFF) for index in range(4))
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"{name} ({device}): {codec} {actual_width}x{actual_height} @ {actual_fps:g} fps")

    def start(self) -> None:
        self.thread.start()

    def _read_loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                self.read_failures += 1
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_number += 1
                self.frame_time = time.monotonic()

    def snapshot(self) -> tuple[Any | None, int, float | None]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, self.frame_number, self.frame_time

    def close(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.capture.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1)


class RecordingSession:
    """Sample both camera readers on one clock and write equal-length videos."""

    def __init__(
        self,
        cameras: list[CameraReader],
        initial_frames: list[Any],
        output_paths: list[Path],
        fps: int,
        duration: float,
        countdown: float,
        max_stale_seconds: float,
    ) -> None:
        self.cameras = cameras
        self.initial_frames = initial_frames
        self.output_paths = output_paths
        self.fps = fps
        self.duration = duration
        self.countdown = countdown
        self.max_stale_seconds = max_stale_seconds
        self.target_frames = max(1, round(duration * fps))

        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.thread = threading.Thread(target=self._record, name="dual-camera-recorder", daemon=True)
        self.error: Exception | None = None
        self.state = "preparing"
        self.countdown_deadline: float | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.frames_written = 0
        self.duplicate_frames = [0 for _ in cameras]

    def start(self) -> None:
        self.thread.start()

    def request_stop(self) -> None:
        self.stop_event.set()

    def join(self) -> None:
        self.thread.join()

    def status_text(self) -> str:
        now = time.monotonic()
        if self.state == "countdown" and self.countdown_deadline is not None:
            remaining = max(0.0, self.countdown_deadline - now)
            return f"Recording starts in {remaining:.1f}s"
        if self.state == "recording" and self.started_at is not None:
            elapsed = min(self.duration, max(0.0, now - self.started_at))
            remaining = max(0.0, self.duration - elapsed)
            return f"REC {format_seconds(elapsed)} / {format_seconds(self.duration)} ({remaining:.0f}s left)"
        if self.state == "finished":
            return "Recording complete"
        if self.state == "failed":
            return "Recording failed"
        return "Preparing"

    def _record(self) -> None:
        writers: list[cv2.VideoWriter] = []
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for frame, path in zip(self.initial_frames, self.output_paths, strict=True):
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (width, height))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Could not create MP4 video at {path}")
                writers.append(writer)

            if self.countdown > 0:
                self.state = "countdown"
                self.countdown_deadline = time.monotonic() + self.countdown
                print(f"Recording starts in {self.countdown:g} seconds...")
                if self.stop_event.wait(self.countdown):
                    return

            self.state = "recording"
            self.started_at = time.monotonic()
            deadline = self.started_at + self.duration
            last_frames = [frame.copy() for frame in self.initial_frames]
            last_frame_numbers = [-1 for _ in self.cameras]
            next_progress_second = 0

            print(f"Recording {self.duration:g} seconds ({self.target_frames} frames per camera).")
            while self.frames_written < self.target_frames and not self.stop_event.is_set():
                now = time.monotonic()
                target_time = self.started_at + self.frames_written / self.fps
                if now < target_time:
                    self.stop_event.wait(min(target_time - now, 0.05))
                    continue

                due_frame_count = min(
                    self.target_frames,
                    int((now - self.started_at) * self.fps) + 1,
                )
                frames_due = due_frame_count - self.frames_written

                snapshots = [camera.snapshot() for camera in self.cameras]
                for index, (frame, frame_number, frame_time) in enumerate(snapshots):
                    if frame is not None:
                        expected_shape = self.initial_frames[index].shape[:2]
                        if frame.shape[:2] != expected_shape:
                            raise RuntimeError(
                                f"{self.cameras[index].name} frame size changed from "
                                f"{expected_shape[::-1]} to {frame.shape[1::-1]}"
                            )
                        last_frames[index] = frame

                    if frame_time is None or now - frame_time > self.max_stale_seconds:
                        raise RuntimeError(
                            f"{self.cameras[index].name} camera stopped producing frames for more than "
                            f"{self.max_stale_seconds:g} seconds"
                        )

                    if frame_number == last_frame_numbers[index]:
                        self.duplicate_frames[index] += frames_due
                    elif frames_due > 1:
                        self.duplicate_frames[index] += frames_due - 1
                    last_frame_numbers[index] = frame_number

                for _ in range(frames_due):
                    for writer, frame in zip(writers, last_frames, strict=True):
                        writer.write(frame)
                self.frames_written = due_frame_count

                elapsed_seconds = int(now - self.started_at)
                if elapsed_seconds >= next_progress_second:
                    remaining = max(0, round(deadline - now))
                    print(
                        f"\rRecording: {format_seconds(now - self.started_at)} elapsed, "
                        f"{format_seconds(remaining)} remaining",
                        end="",
                        flush=True,
                    )
                    next_progress_second = elapsed_seconds + 1

            if self.frames_written == self.target_frames:
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
            print()
        except Exception as error:
            self.error = error
            self.state = "failed"
            self.stop_event.set()
        finally:
            for writer in writers:
                writer.release()
            self.finished_at = time.monotonic()
            if self.error is None:
                self.state = "finished"
            self.finished_event.set()


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def wait_for_initial_frames(cameras: list[CameraReader], timeout: float) -> list[Any]:
    deadline = time.monotonic() + timeout
    frames: list[Any | None] = [None for _ in cameras]
    while time.monotonic() < deadline:
        for index, camera in enumerate(cameras):
            frame, _, _ = camera.snapshot()
            if frame is not None:
                frames[index] = frame
        if all(frame is not None for frame in frames):
            return [frame for frame in frames if frame is not None]
        time.sleep(0.02)

    missing = [camera.name for camera, frame in zip(cameras, frames, strict=True) if frame is None]
    raise RuntimeError(f"Timed out waiting for frames from: {', '.join(missing)}")


def safe_filename(name: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return filename or "camera"


def create_session_directory(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"dual_camera_{timestamp}"
    suffix = 1
    while session_dir.exists():
        session_dir = output_dir / f"dual_camera_{timestamp}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir()
    return session_dir


def run_preview(cameras: list[CameraReader], session: RecordingSession, fps: int) -> None:
    root = tk.Tk()
    windows: list[tk.Misc] = [root, tk.Toplevel(root)]
    labels: list[tk.Label] = []
    photos: list[ImageTk.PhotoImage | None] = [None for _ in cameras]
    closing = False

    def close_all(_event: Any = None) -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        print("\nStopping early; finalizing both MP4 files...")
        session.request_stop()
        root.quit()

    for index, (window, camera) in enumerate(zip(windows, cameras, strict=True)):
        frame, _, _ = camera.snapshot()
        if frame is None:
            raise RuntimeError(f"No preview frame available from {camera.name}")
        height, width = frame.shape[:2]
        window.geometry(f"{width}x{height}+{20 + index * (width + 30)}+20")
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", close_all)
        window.bind("q", close_all)
        window.bind("<Escape>", close_all)
        label = tk.Label(window)
        label.pack(fill=tk.BOTH, expand=True)
        labels.append(label)

    def refresh_windows() -> None:
        status = session.status_text()
        for index, (camera, window, label) in enumerate(zip(cameras, windows, labels, strict=True)):
            frame, _, _ = camera.snapshot()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photos[index] = ImageTk.PhotoImage(Image.fromarray(rgb))
                label.configure(image=photos[index])
            window.title(f"{camera.name}: {camera.device} — {status}")

        if session.finished_event.is_set():
            root.quit()
        elif not closing:
            root.after(max(1, round(1000 / fps)), refresh_windows)

    print("Press Q or Esc in either preview window to stop early.")
    root.after(0, refresh_windows)
    try:
        root.mainloop()
    finally:
        root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        nargs=2,
        default=[
            "/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0",
            "/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0",
        ],
        metavar=("CAMERA_1", "CAMERA_2"),
        help="two V4L2 paths (defaults are the validated front/up camera by-id links)",
    )
    parser.add_argument(
        "--names",
        nargs=2,
        default=["front", "up"],
        metavar=("CAMERA_1", "CAMERA_2"),
        help="camera names used for windows and output files (default: front up)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=300, help="recording duration in seconds")
    parser.add_argument("--countdown", type=float, default=3, help="delay before recording begins")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--first-frame-timeout", type=float, default=10)
    parser.add_argument("--max-stale-seconds", type=float, default=2)
    parser.add_argument("--no-preview", action="store_true", help="record without opening preview windows")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.devices[0] == args.devices[1]:
        raise ValueError("--devices must contain two different camera paths")
    if args.names[0] == args.names[1]:
        raise ValueError("--names must contain two different camera names")
    if safe_filename(args.names[0]) == safe_filename(args.names[1]):
        raise ValueError("--names must produce two different output filenames")
    for option in ("width", "height", "fps"):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option} must be positive")
    for option in ("duration", "first_frame_timeout", "max_stale_seconds"):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")
    if args.countdown < 0:
        raise ValueError("--countdown cannot be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    cameras: list[CameraReader] = []
    session: RecordingSession | None = None

    try:
        for name, device in zip(args.names, args.devices, strict=True):
            camera = CameraReader(name, device, args.width, args.height, args.fps)
            cameras.append(camera)
            camera.start()

        print("Waiting for both cameras...")
        initial_frames = wait_for_initial_frames(cameras, args.first_frame_timeout)
        session_dir = create_session_directory(args.output_dir)
        output_paths = [session_dir / f"{safe_filename(name)}.mp4" for name in args.names]

        print("Output files:")
        for output_path in output_paths:
            print(f"  {output_path}")

        session = RecordingSession(
            cameras=cameras,
            initial_frames=initial_frames,
            output_paths=output_paths,
            fps=args.fps,
            duration=args.duration,
            countdown=args.countdown,
            max_stale_seconds=args.max_stale_seconds,
        )
        session.start()

        try:
            if args.no_preview:
                while not session.finished_event.wait(0.2):
                    pass
            else:
                run_preview(cameras, session, args.fps)
        except KeyboardInterrupt:
            print("\nStopping early; finalizing both MP4 files...")
            session.request_stop()

        session.join()
        if session.error is not None:
            raise RuntimeError(f"Recording failed; partial videos are in {session_dir}") from session.error

        video_seconds = session.frames_written / args.fps
        result = "Completed" if session.frames_written == session.target_frames else "Stopped early"
        print(f"{result}: {session.frames_written} frames per camera ({format_seconds(video_seconds)}).")
        for camera, output_path, duplicates in zip(
            cameras,
            output_paths,
            session.duplicate_frames,
            strict=True,
        ):
            print(
                f"  {camera.name}: {output_path} "
                f"({duplicates} duplicated frames, {camera.read_failures} capture failures)"
            )
    finally:
        if session is not None and not session.finished_event.is_set():
            session.request_stop()
            session.join()
        for camera in cameras:
            camera.close()


if __name__ == "__main__":
    main()
