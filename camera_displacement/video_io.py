"""Video loading and frame iteration.

Thin wrapper around ``cv2.VideoCapture`` that exposes metadata (fps, size,
frame count) and a simple frame iterator, with clear error handling.

Also provides ``CameraRegion`` / ``parse_layout`` for videos that contain
multiple camera feeds arranged in a regular grid (e.g. a 2×2 quad-view).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or read."""


# ---------------------------------------------------------------------------
# Multi-camera grid support
# ---------------------------------------------------------------------------

@dataclass
class CameraRegion:
    """Pixel crop that defines one camera's sub-frame within a composite frame.

    Coordinates are in full-frame pixels.  ``crop()`` extracts the sub-frame.
    """

    camera_id: int        # 0-based, filled left-to-right then top-to-bottom
    x: int
    y: int
    width: int
    height: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the portion of *frame* that corresponds to this camera."""
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def parse_layout(
    layout_str: str,
    frame_width: int,
    frame_height: int,
    num_cameras: Optional[int] = None,
) -> List[CameraRegion]:
    """Parse a ``'COLSxROWS'`` grid string into :class:`CameraRegion` objects.

    Cameras are numbered left-to-right, top-to-bottom starting at 0.
    *num_cameras* limits how many slots are returned (``None`` = all slots).

    Example::

        regions = parse_layout("2x2", 1080, 810, num_cameras=3)
        # → 3 CameraRegions of size 540×405
    """
    parts = layout_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid layout '{layout_str}'. Expected format 'COLSxROWS', e.g. '2x2'."
        )
    cols, rows = int(parts[0]), int(parts[1])
    if cols < 1 or rows < 1:
        raise ValueError("Layout columns and rows must both be >= 1.")

    cell_w = frame_width // cols
    cell_h = frame_height // rows
    total_slots = cols * rows
    limit = total_slots if num_cameras is None else min(num_cameras, total_slots)

    regions: List[CameraRegion] = []
    for cam_id in range(limit):
        row_idx = cam_id // cols
        col_idx = cam_id % cols
        regions.append(
            CameraRegion(
                camera_id=cam_id,
                x=col_idx * cell_w,
                y=row_idx * cell_h,
                width=cell_w,
                height=cell_h,
            )
        )
    return regions


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
