#!/usr/bin/env python3
"""Low-latency two-camera preview using explicit MJPEG capture."""

from __future__ import annotations

import argparse
import threading
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk


class CameraReader:
    def __init__(self, device: str, width: int, height: int, fps: int, fourcc: str) -> None:
        self.device = device
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open {device}")

        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"{device}: {codec} {actual_width}x{actual_height} @ {actual_fps:g} fps")

    def start(self) -> None:
        self.thread.start()

    def _read_loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def latest_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1)
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        nargs=2,
        default=[
            "/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0",
            "/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0",
        ],
        metavar=("FRONT_CAMERA", "UP_CAMERA"),
        help="front and up V4L2 paths (defaults are the validated camera by-id links)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourccs", nargs=2, default=["MJPG", "YUYV"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras: list[CameraReader] = []
    root: tk.Tk | None = None
    try:
        cameras = [
            CameraReader(device, args.width, args.height, args.fps, fourcc)
            for device, fourcc in zip(args.devices, args.fourccs, strict=True)
        ]
        for camera in cameras:
            camera.start()

        root = tk.Tk()
        windows: list[tk.Misc] = [root, tk.Toplevel(root)]
        labels: list[tk.Label] = []
        photos: list[ImageTk.PhotoImage | None] = [None] * len(cameras)
        closing = False

        def close_all(_event=None) -> None:
            nonlocal closing
            closing = True
            if root is not None:
                root.quit()

        for index, (window, camera) in enumerate(zip(windows, cameras, strict=True)):
            window.title(f"Camera {camera.device}")
            window.geometry(f"{args.width}x{args.height}+{20 + index * (args.width + 30)}+20")
            window.resizable(False, False)
            window.protocol("WM_DELETE_WINDOW", close_all)
            window.bind("q", close_all)
            window.bind("<Escape>", close_all)
            label = tk.Label(window)
            label.pack(fill=tk.BOTH, expand=True)
            labels.append(label)

        def refresh_windows() -> None:
            for index, (camera, label) in enumerate(zip(cameras, labels, strict=True)):
                frame = camera.latest_frame()
                if frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photos[index] = ImageTk.PhotoImage(Image.fromarray(rgb))
                label.configure(image=photos[index])
            if not closing and root is not None:
                root.after(max(1, round(1000 / args.fps)), refresh_windows)

        print("Press Q or Esc in either preview window to close both.")
        root.after(0, refresh_windows)
        root.mainloop()
    finally:
        for camera in cameras:
            camera.close()
        if root is not None:
            root.destroy()


if __name__ == "__main__":
    main()
