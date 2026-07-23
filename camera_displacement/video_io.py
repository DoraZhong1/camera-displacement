"""Video loading and frame iteration.

Thin wrapper around ``cv2.VideoCapture`` that exposes metadata (fps, size,
frame count) and a simple frame iterator, with clear error handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or read."""


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoLoader:
    """Load a video and iterate over its frames.

    Usage::

        with VideoLoader("clip.mp4") as loader:
            for frame_index, frame in loader.frames():
                ...
    """

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise VideoError(f"Video file not found: {path}")
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise VideoError(
                f"Could not open video: {path}. "
                "The file may be corrupt or use an unsupported codec."
            )

        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        # Some containers report 0 or absurd fps; fall back to a sane default.
        if not np.isfinite(fps) or fps <= 0 or fps > 1000:
            fps = 30.0
        self.info = VideoInfo(
            path=path,
            fps=fps,
            frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # -- frame access ----------------------------------------------------
    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """Return a specific frame by index, or ``None`` if unavailable."""
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        return frame if ok else None

    def frames(self, start: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` from ``start`` to the end of the video."""
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            yield index, frame
            index += 1

    def timestamp(self, frame_index: int) -> float:
        """Timestamp in seconds for a given frame index (from fps)."""
        return frame_index / self.info.fps if self.info.fps > 0 else 0.0
