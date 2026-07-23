"""Annotated output-video generation.

Overlays the baseline ROI box, the (moved) current reference position, tracked
feature points, the live displacement/rotation/confidence readout and an
unreliable-tracking warning onto each frame, and writes an MP4.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .calibration import Calibration
from .roi_selector import ROI
from .tracker import FrameResult, TrackingStatus

_STATUS_COLOR = {
    TrackingStatus.OK: (0, 200, 0),
    TrackingStatus.LOW_CONFIDENCE: (0, 200, 255),
    TrackingStatus.LOST: (0, 0, 255),
}


class AnnotatedVideoWriter:
    def __init__(
        self,
        path: str,
        fps: float,
        frame_size: Tuple[int, int],
        baseline_roi: ROI,
        calibration: Optional[Calibration] = None,
    ) -> None:
        self.baseline_roi = baseline_roi
        self.calibration = calibration
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}.")
        self.path = path

    def write(self, frame: np.ndarray, result: FrameResult) -> None:
        img = frame.copy()
        color = _STATUS_COLOR[result.tracking_status]
        x, y, w, h = self.baseline_roi

        # Baseline reference box (cyan, dashed look via thin rectangle).
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
        cv2.putText(img, "baseline", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # Tracked feature points.
        for px, py in result.tracked_points:
            cv2.circle(img, (int(round(px)), int(round(py))), 2, color, -1, cv2.LINE_AA)

        # Current reference position (box shifted by apparent object motion).
        if result.current_center is not None and result.tracking_status != TrackingStatus.LOST:
            base_cx, base_cy = x + w / 2.0, y + h / 2.0
            odx = result.current_center[0] - base_cx
            ody = result.current_center[1] - base_cy
            cx, cy = int(round(x + odx)), int(round(y + ody))
            cv2.rectangle(img, (cx, cy), (cx + w, cy + h), color, 2)

        self._draw_hud(img, result, color)
        self.writer.write(img)

    def _draw_hud(self, img, result: FrameResult, color) -> None:
        lines = [
            f"frame {result.frame_number}  t={result.timestamp_seconds:.2f}s",
            f"dx={result.displacement_x_pixels:+.1f}px  dy={result.displacement_y_pixels:+.1f}px",
            f"total={result.total_displacement_pixels:.1f}px  rot={result.rotation_degrees:+.2f}deg",
            f"inliers={result.number_of_inlier_features}/{result.number_of_matched_features}"
            f"  conf={result.tracking_confidence:.2f}",
        ]
        if self.calibration is not None and self.calibration.is_calibrated:
            mm = self.calibration.px_to_mm(result.total_displacement_pixels)
            lines.append(f"total={mm:.3f}mm")

        # Semi-transparent panel behind the text.
        panel = img.copy()
        cv2.rectangle(panel, (5, 5), (360, 20 + 20 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.45, img, 0.55, 0, img)
        yy = 25
        for line in lines:
            cv2.putText(img, line, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            yy += 20

        cv2.putText(img, result.tracking_status.value, (12, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if result.tracking_status != TrackingStatus.OK:
            h_img, w_img = img.shape[:2]
            msg = "WARNING: tracking unreliable" if result.tracking_status == \
                TrackingStatus.LOW_CONFIDENCE else "WARNING: reference object LOST"
            cv2.putText(img, msg, (12, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
