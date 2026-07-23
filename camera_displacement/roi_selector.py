"""Region-of-interest (ROI) selection and baseline-frame confirmation.

Provides an interactive OpenCV window (``cv2.selectROI``) for drawing the
bounding box around the stationary reference object, plus non-interactive
fallbacks so the tool can also run headless (e.g. on a CI server) by passing
an explicit ROI on the command line.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

ROI = Tuple[int, int, int, int]  # (x, y, w, h)


def select_roi_interactive(
    frame: np.ndarray, window_title: str = "Select reference object, then ENTER/SPACE"
) -> Optional[ROI]:
    """Open a window and let the user draw a bounding box.

    Returns ``(x, y, w, h)`` or ``None`` if the user cancelled (ESC) or drew
    an empty box.
    """
    display = frame.copy()
    cv2.putText(
        display,
        "Drag a box around the stationary reference object. ENTER=confirm, C=cancel",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    try:
        x, y, w, h = cv2.selectROI(window_title, display, showCrosshair=True, fromCenter=False)
    except cv2.error as exc:  # pragma: no cover - depends on GUI backend
        raise RuntimeError(
            "Interactive ROI selection is unavailable (no GUI backend). "
            "Pass the ROI explicitly with --roi x y w h."
        ) from exc
    finally:
        cv2.destroyAllWindows()
        # Extra waitKey calls flush the window close on macOS.
        for _ in range(4):
            cv2.waitKey(1)

    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def validate_roi(roi: ROI, frame_shape: Tuple[int, int]) -> ROI:
    """Clamp an ROI to the frame bounds and sanity-check it."""
    h_img, w_img = frame_shape[:2]
    x, y, w, h = roi
    x = max(0, min(int(x), w_img - 1))
    y = max(0, min(int(y), h_img - 1))
    w = max(1, min(int(w), w_img - x))
    h = max(1, min(int(h), h_img - y))
    if w < 8 or h < 8:
        raise ValueError(
            f"ROI is too small ({w}x{h}). Select a region at least 8x8 pixels."
        )
    return x, y, w, h


def roi_mask(frame_shape: Tuple[int, int], roi: ROI) -> np.ndarray:
    """Build a uint8 mask (255 inside the ROI, 0 elsewhere) for feature detection."""
    h_img, w_img = frame_shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    x, y, w, h = roi
    mask[y : y + h, x : x + w] = 255
    return mask


def roi_center(roi: ROI) -> Tuple[float, float]:
    x, y, w, h = roi
    return (x + w / 2.0, y + h / 2.0)
