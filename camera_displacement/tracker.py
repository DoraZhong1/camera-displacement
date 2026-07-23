"""Feature detection, tracking, frame registration and transform estimation.

Two complementary tracking pipelines are implemented:

* :class:`BaselineTracker` -- ORB (or SIFT) feature matching of each frame
  *directly against the baseline frame*, followed by RANSAC partial-affine
  estimation. This produces the **absolute displacement** (Mode 2) without
  accumulating drift, and self-heals after tracking loss because every frame
  is re-registered against the baseline independently.

* :class:`ConsecutiveTracker` -- Shi-Tomasi corners seeded inside the ROI and
  tracked forward with Lucas-Kanade optical flow. This produces the
  **consecutive frame-to-frame movement** (Mode 1).

Both pipelines return a :class:`FrameResult` describing translation, rotation,
match/inlier counts, confidence and a status flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .roi_selector import ROI, roi_center, roi_mask


class TrackingStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOST = "LOST"


@dataclass
class FrameResult:
    """Per-frame measurement for one tracking mode."""

    frame_number: int
    timestamp_seconds: float
    # Camera displacement (image plane, pixels). Sign convention: camera
    # displacement is the negation of the apparent reference-object motion.
    displacement_x_pixels: float = 0.0
    displacement_y_pixels: float = 0.0
    total_displacement_pixels: float = 0.0
    rotation_degrees: float = 0.0
    scale: float = 1.0
    number_of_matched_features: int = 0
    number_of_inlier_features: int = 0
    tracking_confidence: float = 0.0
    tracking_status: TrackingStatus = TrackingStatus.LOST
    # Tracked point locations in the *current* frame, for annotation.
    tracked_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    # Current location of the reference-object center, for annotation.
    current_center: Optional[Tuple[float, float]] = None

    def as_row(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "displacement_x_pixels": round(self.displacement_x_pixels, 4),
            "displacement_y_pixels": round(self.displacement_y_pixels, 4),
            "total_displacement_pixels": round(self.total_displacement_pixels, 4),
            "rotation_degrees": round(self.rotation_degrees, 5),
            "scale": round(self.scale, 5),
            "number_of_matched_features": self.number_of_matched_features,
            "number_of_inlier_features": self.number_of_inlier_features,
            "tracking_confidence": round(self.tracking_confidence, 4),
            "tracking_status": self.tracking_status.value,
        }


# --------------------------------------------------------------------------
# Transform helpers
# --------------------------------------------------------------------------
def decompose_affine(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Decompose a 2x3 partial-affine matrix into (tx, ty, rotation_deg, scale)."""
    a, b = matrix[0, 0], matrix[1, 0]
    tx, ty = matrix[0, 2], matrix[1, 2]
    rotation = math.degrees(math.atan2(b, a))
    scale = math.hypot(a, b)
    return tx, ty, rotation, scale


def apply_affine(matrix: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    """Apply a 2x3 affine matrix to a single (x, y) point."""
    x, y = point
    nx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    ny = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return float(nx), float(ny)


# --------------------------------------------------------------------------
# Baseline (absolute) tracker: ORB / SIFT matching + RANSAC
# --------------------------------------------------------------------------
class BaselineTracker:
    """Register each frame directly against the baseline reference ROI."""

    def __init__(
        self,
        baseline_frame: np.ndarray,
        roi: ROI,
        detector: str = "ORB",
        max_features: int = 2000,
        ratio_test: float = 0.75,
        min_inliers: int = 12,
        min_inlier_ratio: float = 0.30,
        ransac_threshold: float = 3.0,
    ) -> None:
        self.roi = roi
        self.roi_center = roi_center(roi)
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ratio_test = ratio_test
        self.ransac_threshold = ransac_threshold

        self.detector, self.norm = _build_detector(detector, max_features)
        self.matcher = cv2.BFMatcher(self.norm)

        gray = _to_gray(baseline_frame)
        mask = roi_mask(baseline_frame.shape, roi)
        self.base_kp, self.base_desc = self.detector.detectAndCompute(gray, mask)
        if self.base_desc is None or len(self.base_kp) < min_inliers:
            raise ValueError(
                f"Only {0 if self.base_kp is None else len(self.base_kp)} features found in the "
                "baseline ROI. Choose a more textured reference object, enlarge the ROI, "
                "or try --detector SIFT."
            )
        self.base_pts = np.float32([kp.pt for kp in self.base_kp])

    def process(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        result = FrameResult(frame_number=frame_number, timestamp_seconds=timestamp)
        gray = _to_gray(frame)
        kp, desc = self.detector.detectAndCompute(gray, None)
        if desc is None or len(kp) < 2:
            return result  # LOST

        # knn match + Lowe ratio test.
        good = _ratio_matched_pairs(self.matcher, self.base_desc, desc, self.ratio_test)
        result.number_of_matched_features = len(good)
        if len(good) < self.min_inliers:
            result.tracking_status = TrackingStatus.LOST
            return result

        base_matched = np.float32([self.base_pts[m.queryIdx] for m in good])
        cur_matched = np.float32([kp[m.trainIdx].pt for m in good])

        matrix, inliers = cv2.estimateAffinePartial2D(
            base_matched,
            cur_matched,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=5000,
            confidence=0.99,
        )
        if matrix is None or inliers is None:
            result.tracking_status = TrackingStatus.LOST
            return result

        inlier_mask = inliers.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())
        result.number_of_inlier_features = n_inliers
        result.tracked_points = cur_matched[inlier_mask]

        inlier_ratio = n_inliers / max(len(good), 1)
        # Confidence blends absolute inlier support with the inlier ratio.
        support = min(1.0, n_inliers / 60.0)
        result.tracking_confidence = float(np.clip(0.5 * inlier_ratio + 0.5 * support, 0.0, 1.0))

        # Apparent object motion = where the baseline ROI center maps to now.
        new_center = apply_affine(matrix, self.roi_center)
        result.current_center = new_center
        obj_dx = new_center[0] - self.roi_center[0]
        obj_dy = new_center[1] - self.roi_center[1]
        _, _, rotation, scale = decompose_affine(matrix)

        # Camera displacement is the negation of apparent object motion.
        result.displacement_x_pixels = -obj_dx
        result.displacement_y_pixels = -obj_dy
        result.total_displacement_pixels = math.hypot(obj_dx, obj_dy)
        result.rotation_degrees = -rotation
        result.scale = scale

        if n_inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            result.tracking_status = TrackingStatus.LOW_CONFIDENCE
        else:
            result.tracking_status = TrackingStatus.OK
        return result


# --------------------------------------------------------------------------
# Consecutive tracker: Shi-Tomasi + Lucas-Kanade optical flow
# --------------------------------------------------------------------------
class ConsecutiveTracker:
    """Track ROI corners frame-to-frame with LK optical flow (Mode 1)."""

    def __init__(
        self,
        baseline_frame: np.ndarray,
        roi: ROI,
        max_corners: int = 400,
        quality: float = 0.01,
        min_distance: float = 5.0,
        min_points: int = 8,
        reseed_below: int = 20,
    ) -> None:
        self.roi = roi
        self.min_points = min_points
        self.reseed_below = reseed_below
        self.corner_kwargs = dict(
            maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance, blockSize=7
        )
        self.lk_kwargs = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray = _to_gray(baseline_frame)
        self.prev_pts = self._detect(self.prev_gray, roi)

    def _detect(self, gray: np.ndarray, roi: ROI) -> np.ndarray:
        mask = roi_mask(gray.shape, roi)
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.corner_kwargs)
        return pts if pts is not None else np.empty((0, 1, 2), np.float32)

    def process(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        result = FrameResult(frame_number=frame_number, timestamp_seconds=timestamp)
        gray = _to_gray(frame)

        if self.prev_pts is None or len(self.prev_pts) < self.min_points:
            # Nothing to track from the previous frame.
            self.prev_gray = gray
            self.prev_pts = self._detect(gray, self.roi)
            return result  # LOST

        nxt, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_kwargs
        )
        if nxt is None or status is None:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray, self.roi)
            return result

        good = status.ravel() == 1
        prev_good = self.prev_pts[good].reshape(-1, 2)
        next_good = nxt[good].reshape(-1, 2)
        result.number_of_matched_features = int(good.sum())

        if len(prev_good) >= self.min_points:
            matrix, inliers = cv2.estimateAffinePartial2D(
                prev_good, next_good, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if matrix is not None and inliers is not None:
                inlier_mask = inliers.ravel().astype(bool)
                n_inliers = int(inlier_mask.sum())
                result.number_of_inlier_features = n_inliers
                result.tracked_points = next_good[inlier_mask]

                tx, ty, rotation, scale = decompose_affine(matrix)
                # Consecutive apparent motion of the object = translation term
                # at the ROI center.
                cx, cy = roi_center(self.roi)
                new_c = apply_affine(matrix, (cx, cy))
                obj_dx, obj_dy = new_c[0] - cx, new_c[1] - cy
                result.displacement_x_pixels = -obj_dx
                result.displacement_y_pixels = -obj_dy
                result.total_displacement_pixels = math.hypot(obj_dx, obj_dy)
                result.rotation_degrees = -rotation
                result.scale = scale
                result.current_center = new_c

                inlier_ratio = n_inliers / max(len(prev_good), 1)
                result.tracking_confidence = float(
                    np.clip(0.5 * inlier_ratio + 0.5 * min(1.0, n_inliers / 60.0), 0, 1)
                )
                result.tracking_status = (
                    TrackingStatus.OK
                    if n_inliers >= self.min_points and inlier_ratio >= 0.3
                    else TrackingStatus.LOW_CONFIDENCE
                )
                next_good = next_good[inlier_mask]

        # Advance state; reseed corners if we are running low.
        self.prev_gray = gray
        if len(next_good) < self.reseed_below:
            self.prev_pts = self._detect(gray, self.roi)
        else:
            self.prev_pts = next_good.reshape(-1, 1, 2)
        return result


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _build_detector(name: str, max_features: int):
    """Return (detector, norm_type). Falls back to ORB if SIFT is unavailable."""
    name = name.upper()
    if name == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
        raise ValueError(
            "SIFT is unavailable. Install 'opencv-contrib-python' or use --detector ORB."
        )
    # Default: ORB (fast, license-free, Hamming distance).
    return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING


def _ratio_matched_pairs(matcher, desc1, desc2, ratio: float) -> List[cv2.DMatch]:
    """kNN match desc1 -> desc2 and keep matches passing Lowe's ratio test."""
    if desc1 is None or desc2 is None or len(desc2) < 2:
        return []
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good
