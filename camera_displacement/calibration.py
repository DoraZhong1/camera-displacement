"""Pixel-to-millimetre calibration.

Three ways to obtain pixels-per-millimetre are supported:

1. A known physical dimension visible in the image, together with the pixel
   length it spans (``from_known_dimension``).
2. A manually entered pixels-per-millimetre value (``from_ppm``).
3. A calibration checkerboard: detect the inner-corner grid and use the known
   square size (``from_checkerboard``).

IMPORTANT CAVEAT (also surfaced in the report): a single 2-D video measures
displacement in the *image plane*. A pixels-per-millimetre factor is only
physically accurate at the depth where it was measured (the reference
object's distance). True 3-D camera translation in millimetres additionally
requires intrinsic calibration and a known camera-to-object distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Calibration:
    pixels_per_mm: Optional[float] = None
    method: str = "none"
    note: str = ""

    @property
    def is_calibrated(self) -> bool:
        return self.pixels_per_mm is not None and self.pixels_per_mm > 0

    def px_to_mm(self, pixels: float) -> Optional[float]:
        if not self.is_calibrated:
            return None
        return pixels / self.pixels_per_mm

    def describe(self) -> str:
        if not self.is_calibrated:
            return "Uncalibrated (results in pixels only)."
        return (
            f"{self.pixels_per_mm:.4f} px/mm via {self.method}. "
            "Valid only at the reference object's depth; see README caveat."
        )

    @staticmethod
    def none() -> "Calibration":
        return Calibration(pixels_per_mm=None, method="none")

    @staticmethod
    def from_ppm(pixels_per_mm: float) -> "Calibration":
        if pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be positive.")
        return Calibration(pixels_per_mm=pixels_per_mm, method="manual pixels/mm")

    @staticmethod
    def from_known_dimension(pixel_length: float, real_mm: float) -> "Calibration":
        if pixel_length <= 0 or real_mm <= 0:
            raise ValueError("pixel_length and real_mm must be positive.")
        return Calibration(
            pixels_per_mm=pixel_length / real_mm,
            method=f"known dimension ({real_mm} mm = {pixel_length} px)",
        )

    @staticmethod
    def from_checkerboard(
        frame: np.ndarray,
        pattern_size: Tuple[int, int],
        square_size_mm: float,
    ) -> "Calibration":
        """Detect a checkerboard and derive px/mm from the known square size.

        ``pattern_size`` is the number of *inner* corners (cols, rows).
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            raise ValueError(
                "Checkerboard not detected. Check the pattern size (inner corners) "
                "and that the board is fully visible and in focus."
            )
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1),
        ).reshape(-1, 2)

        cols, rows = pattern_size
        grid = corners.reshape(rows, cols, 2)
        # Average pixel spacing between horizontally and vertically adjacent corners.
        h_sp = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean()
        v_sp = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
        px_per_square = float((h_sp + v_sp) / 2.0)
        return Calibration(
            pixels_per_mm=px_per_square / square_size_mm,
            method=f"checkerboard ({square_size_mm} mm squares)",
        )
