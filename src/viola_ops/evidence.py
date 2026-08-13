"""Complete local evidence recording for the 30 Hz execution loop.

The control thread reserves queue capacity before every motor write.  A single
worker owns JSONL and video files, so no filesystem or network operation occurs
inside the write-critical section.  Saturation or a worker failure makes the
next reservation fail and therefore stops motion.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viola_handoff import canonical_json_bytes

from .errors import ValidationError


@dataclass(frozen=True, slots=True)
class TrialFiles:
    trial_id: str
    trace: Path
    front_video: Path
    up_video: Path
    rows: int
    video_frames: int


class PhaseEvidenceFactory:
    """Create one bounded writer per trial under a fresh motion record."""

    def __init__(self, root: str | Path, *, fps: int = 30, queue_size: int = 64) -> None:
        self.root = Path(root).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ValidationError(f"motion evidence directory is not empty: {self.root}")
        (self.root / "safety_traces").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.queue_size = queue_size
        self.completed: list[TrialFiles] = []

    def start_trial(self, trial_id: str) -> "QueuedTrialEvidence":
        index = len(self.completed)
        writer = QueuedTrialEvidence(
            trial_id,
            trace=self.root / "safety_traces" / f"trial-{index:02d}.jsonl",
            front_video=self.root / "videos" / f"trial-{index:02d}-front.mp4",
            up_video=self.root / "videos" / f"trial-{index:02d}-up.mp4",
            fps=self.fps,
            queue_size=self.queue_size,
            on_close=self.completed.append,
        )
        return writer

    @classmethod
    def reopen_completed(cls, root: str | Path) -> "PhaseEvidenceFactory":
        """Reopen finished local evidence for upload/sealing only.

        This path creates no worker and cannot record another action.  It lets
        an operator retry W&B or handoff finalization without repeating motion.
        """

        evidence_root = Path(root).resolve()
        if not (evidence_root / "safety_traces").is_dir() or not (
            evidence_root / "videos"
        ).is_dir():
            raise ValidationError("completed motion evidence is missing traces or videos")
        instance = object.__new__(cls)
        instance.root = evidence_root
        instance.fps = 30
        instance.queue_size = 0
        instance.completed = []
        return instance


class _Reservation:
    def __init__(self, owner: "QueuedTrialEvidence", *, front: Any, up: Any) -> None:
        self._owner = owner
        self._front = front
        self._up = up
        self._open = True

    def commit(self, row: Mapping[str, Any]) -> None:
        if not self._open:
            raise ValidationError("evidence reservation was already consumed")
        self._open = False
        try:
            self._owner._commit_reserved(row, front=self._front, up=self._up)
        except BaseException:
            self._owner._release_capacity()
            raise

    def cancel(self) -> None:
        if self._open:
            self._open = False
            self._owner._release_capacity()


class QueuedTrialEvidence:
    """A bounded producer and a single local writer thread."""

    def __init__(
        self,
        trial_id: str,
        *,
        trace: Path,
        front_video: Path,
        up_video: Path,
        fps: int,
        queue_size: int,
        on_close: Any,
    ) -> None:
        if queue_size < 1:
            raise ValidationError("evidence queue size must be positive")
        self.trial_id = trial_id
        self.trace = trace
        self.front_video = front_video
        self.up_video = up_video
        self.fps = fps
        self._on_close = on_close
        self._items: queue.Queue[Any] = queue.Queue()
        self._capacity = threading.BoundedSemaphore(queue_size)
        self._failure: BaseException | None = None
        self._closed = False
        self._rows = 0
        self._video_frames = 0
        self._thread = threading.Thread(
            target=self._worker,
            name=f"viola-evidence-{trial_id}",
            daemon=True,
        )
        self._thread.start()

    def reserve(self, *, front: Any, up: Any) -> _Reservation | None:
        if self._closed or self._failure is not None:
            return None
        if not self._capacity.acquire(blocking=False):
            return None
        if self._failure is not None:
            self._capacity.release()
            return None
        try:
            import numpy as np

            front_copy = np.ascontiguousarray(front).copy()
            up_copy = np.ascontiguousarray(up).copy()
            _require_frame(front_copy, "front")
            _require_frame(up_copy, "up")
        except Exception as exc:
            self._capacity.release()
            raise ValidationError(f"camera frames cannot be retained: {exc}") from exc
        return _Reservation(self, front=front_copy, up=up_copy)

    def record_terminal(self, row: Mapping[str, Any]) -> None:
        """Record after motion has stopped; bounded waiting is safe here."""

        if self._failure is not None:
            raise ValidationError(f"evidence writer already failed: {self._failure}")
        if not self._capacity.acquire(timeout=5.0):
            raise ValidationError("evidence queue could not retain the terminal row")
        self._items.put(("terminal", dict(row), None, None))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._items.put(("stop", None, None, None))
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise ValidationError("evidence writer did not stop within 30 seconds")
        if self._failure is not None:
            raise ValidationError(f"evidence writer failed: {self._failure}") from self._failure
        result = TrialFiles(
            self.trial_id,
            self.trace,
            self.front_video,
            self.up_video,
            self._rows,
            self._video_frames,
        )
        self._on_close(result)

    def _commit_reserved(self, row: Mapping[str, Any], *, front: Any, up: Any) -> None:
        if self._closed or self._failure is not None:
            raise ValidationError("evidence writer is unavailable")
        self._items.put_nowait(("frame", dict(row), front, up))

    def _release_capacity(self) -> None:
        self._capacity.release()

    def _worker(self) -> None:
        trace_handle = None
        front_container = up_container = None
        try:
            import av

            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.front_video.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = self.trace.open("xb")
            front_container = av.open(str(self.front_video), mode="w")
            up_container = av.open(str(self.up_video), mode="w")
            front_stream = front_container.add_stream("libx264", rate=self.fps)
            up_stream = up_container.add_stream("libx264", rate=self.fps)
            for stream in (front_stream, up_stream):
                stream.width = 640
                stream.height = 480
                stream.pix_fmt = "yuv420p"

            while True:
                kind, row, front, up = self._items.get()
                if kind == "stop":
                    break
                try:
                    enriched = dict(row)
                    if kind == "frame":
                        _require_frame(front, "front")
                        _require_frame(up, "up")
                        enriched["front_sha256"] = _array_sha256(front)
                        enriched["up_sha256"] = _array_sha256(up)
                        _encode(front_container, front_stream, front)
                        _encode(up_container, up_stream, up)
                        self._video_frames += 1
                    trace_handle.write(canonical_json_bytes(enriched) + b"\n")
                    self._rows += 1
                finally:
                    self._capacity.release()

            for packet in front_stream.encode():
                front_container.mux(packet)
            for packet in up_stream.encode():
                up_container.mux(packet)
            trace_handle.flush()
            os.fsync(trace_handle.fileno())
        except BaseException as exc:
            self._failure = exc
            # Drain reservations so producers cannot hang during teardown.
            while True:
                try:
                    kind, _row, _front, _up = self._items.get_nowait()
                except queue.Empty:
                    break
                if kind != "stop":
                    try:
                        self._capacity.release()
                    except ValueError:
                        pass
        finally:
            if trace_handle is not None:
                trace_handle.close()
            for container in (front_container, up_container):
                if container is not None:
                    try:
                        container.close()
                    except Exception as exc:
                        if self._failure is None:
                            self._failure = exc


def _encode(container: Any, stream: Any, image: Any) -> None:
    import av

    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def _require_frame(value: Any, label: str) -> None:
    if getattr(value, "shape", None) != (480, 640, 3):
        raise ValidationError(f"{label} frame must have shape 480x640x3")


def _array_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical JSONL for evidence finalization and tests."""

    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"trace line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
            raise ValidationError(f"trace line {line_number} is not a canonical object")
        rows.append(value)
    return rows


__all__ = [
    "PhaseEvidenceFactory",
    "QueuedTrialEvidence",
    "TrialFiles",
    "read_trace",
]
