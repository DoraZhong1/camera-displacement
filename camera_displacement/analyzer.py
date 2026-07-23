"""Top-level analysis orchestration.

Ties together video loading, the two tracking pipelines, annotated-video
writing, CSV export and graph generation in a single pass over the video.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .annotate import AnnotatedVideoWriter
from .calibration import Calibration
from .reporting import generate_plots, summarize, write_csv
from .roi_selector import ROI, validate_roi
from .tracker import BaselineTracker, ConsecutiveTracker, FrameResult
from .video_io import VideoLoader


@dataclass
class AnalyzerConfig:
    video_path: str
    roi: ROI
    output_dir: str
    baseline_frame_index: int = 0
    detector: str = "ORB"            # ORB or SIFT
    max_features: int = 2000
    min_inliers: int = 12
    min_inlier_ratio: float = 0.30
    ratio_test: float = 0.75
    calibration: Calibration = field(default_factory=Calibration.none)
    write_annotated_video: bool = True
    progress: Optional[Callable[[int, int], None]] = None  # (current, total)


@dataclass
class AnalyzerOutputs:
    absolute_csv: str
    consecutive_csv: str
    absolute_plots: List[str]
    consecutive_plots: List[str]
    annotated_video: Optional[str]
    absolute_results: List[FrameResult]
    consecutive_results: List[FrameResult]
    absolute_summary: dict
    consecutive_summary: dict


class DisplacementAnalyzer:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

    def run(self) -> AnalyzerOutputs:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        with VideoLoader(cfg.video_path) as loader:
            info = loader.info
            baseline = loader.read_frame(cfg.baseline_frame_index)
            if baseline is None:
                raise ValueError(
                    f"Could not read baseline frame {cfg.baseline_frame_index} "
                    f"(video has ~{info.frame_count} frames)."
                )
            roi = validate_roi(cfg.roi, baseline.shape)

            # Mode 2 (primary): absolute registration against the baseline.
            baseline_tracker = BaselineTracker(
                baseline,
                roi,
                detector=cfg.detector,
                max_features=cfg.max_features,
                ratio_test=cfg.ratio_test,
                min_inliers=cfg.min_inliers,
                min_inlier_ratio=cfg.min_inlier_ratio,
            )
            # Mode 1: consecutive frame-to-frame optical flow.
            consecutive_tracker = ConsecutiveTracker(baseline, roi)

            annotator = None
            annotated_path = None
            if cfg.write_annotated_video:
                annotated_path = os.path.join(cfg.output_dir, "annotated.mp4")
                annotator = AnnotatedVideoWriter(
                    annotated_path,
                    info.fps,
                    (info.width, info.height),
                    roi,
                    cfg.calibration,
                )

            abs_results: List[FrameResult] = []
            con_results: List[FrameResult] = []

            try:
                for idx, frame in loader.frames(start=cfg.baseline_frame_index):
                    ts = loader.timestamp(idx)
                    abs_res = baseline_tracker.process(frame, idx, ts)
                    con_res = consecutive_tracker.process(frame, idx, ts)
                    abs_results.append(abs_res)
                    con_results.append(con_res)

                    if annotator is not None:
                        annotator.write(frame, abs_res)  # annotate with absolute result
                    if cfg.progress is not None:
                        cfg.progress(idx - cfg.baseline_frame_index + 1, info.frame_count)
            finally:
                if annotator is not None:
                    annotator.release()

        # Export.
        abs_csv = os.path.join(cfg.output_dir, "absolute_displacement.csv")
        con_csv = os.path.join(cfg.output_dir, "consecutive_displacement.csv")
        write_csv(abs_results, abs_csv, cfg.calibration)
        write_csv(con_results, con_csv, cfg.calibration)

        abs_plots = generate_plots(abs_results, cfg.output_dir, prefix="absolute_")
        con_plots = generate_plots(con_results, cfg.output_dir, prefix="consecutive_")

        return AnalyzerOutputs(
            absolute_csv=abs_csv,
            consecutive_csv=con_csv,
            absolute_plots=abs_plots,
            consecutive_plots=con_plots,
            annotated_video=annotated_path,
            absolute_results=abs_results,
            consecutive_results=con_results,
            absolute_summary=summarize(abs_results),
            consecutive_summary=summarize(con_results),
        )
