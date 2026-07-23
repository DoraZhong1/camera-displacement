"""CSV export and graph generation."""

from __future__ import annotations

import csv
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must precede pyplot import.
import matplotlib.pyplot as plt  # noqa: E402

from .calibration import Calibration  # noqa: E402
from .tracker import FrameResult, TrackingStatus  # noqa: E402

CSV_COLUMNS = [
    "frame_number",
    "timestamp_seconds",
    "displacement_x_pixels",
    "displacement_y_pixels",
    "total_displacement_pixels",
    "rotation_degrees",
    "scale",
    "number_of_matched_features",
    "number_of_inlier_features",
    "tracking_confidence",
    "tracking_status",
]


def write_csv(
    results: List[FrameResult],
    path: str,
    calibration: Optional[Calibration] = None,
) -> None:
    """Write per-frame results to CSV, adding *_mm columns when calibrated."""
    calibrated = calibration is not None and calibration.is_calibrated
    columns = list(CSV_COLUMNS)
    if calibrated:
        # Insert mm columns right after their pixel counterparts.
        for px_col, mm_col in [
            ("displacement_x_pixels", "displacement_x_mm"),
            ("displacement_y_pixels", "displacement_y_mm"),
            ("total_displacement_pixels", "total_displacement_mm"),
        ]:
            columns.insert(columns.index(px_col) + 1, mm_col)

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in results:
            row = r.as_row()
            if calibrated:
                row["displacement_x_mm"] = round(calibration.px_to_mm(r.displacement_x_pixels), 5)
                row["displacement_y_mm"] = round(calibration.px_to_mm(r.displacement_y_pixels), 5)
                row["total_displacement_mm"] = round(
                    calibration.px_to_mm(r.total_displacement_pixels), 5
                )
            writer.writerow(row)


def _series(results: List[FrameResult]):
    t = [r.timestamp_seconds for r in results]
    return {
        "t": t,
        "dx": [r.displacement_x_pixels for r in results],
        "dy": [r.displacement_y_pixels for r in results],
        "total": [r.total_displacement_pixels for r in results],
        "rot": [r.rotation_degrees for r in results],
        "conf": [r.tracking_confidence for r in results],
    }


def generate_plots(results: List[FrameResult], out_dir: str, prefix: str = "") -> List[str]:
    """Create the five required time-series graphs. Returns saved file paths."""
    if not results:
        return []
    os.makedirs(out_dir, exist_ok=True)
    s = _series(results)

    specs = [
        ("dx", "Horizontal displacement Δx (px)", "displacement_x_vs_time.png", "tab:blue"),
        ("dy", "Vertical displacement Δy (px)", "displacement_y_vs_time.png", "tab:orange"),
        ("total", "Total displacement (px)", "total_displacement_vs_time.png", "tab:red"),
        ("rot", "Rotation (degrees)", "rotation_vs_time.png", "tab:green"),
        ("conf", "Tracking confidence (0-1)", "confidence_vs_time.png", "tab:purple"),
    ]

    saved = []
    for key, ylabel, fname, color in specs:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s["t"], s[key], color=color, linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs time")
        ax.grid(True, alpha=0.3)
        if key == "conf":
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        path = os.path.join(out_dir, prefix + fname)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    # Bonus combined overview.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(s["t"], s["dx"], label="Δx", color="tab:blue")
    axes[0].plot(s["t"], s["dy"], label="Δy", color="tab:orange")
    axes[0].set_ylabel("px"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Camera displacement overview")
    axes[1].plot(s["t"], s["total"], color="tab:red")
    axes[1].set_ylabel("total px"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(s["t"], s["conf"], color="tab:purple")
    axes[2].set_ylabel("confidence"); axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.02, 1.02); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, prefix + "overview.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)
    return saved


def summarize(results: List[FrameResult]) -> dict:
    """Compute headline statistics for a console summary."""
    if not results:
        return {}
    ok = [r for r in results if r.tracking_status == TrackingStatus.OK]
    totals = [r.total_displacement_pixels for r in results if r.tracking_status != TrackingStatus.LOST]
    max_r = max(results, key=lambda r: r.total_displacement_pixels)
    lost = sum(1 for r in results if r.tracking_status == TrackingStatus.LOST)
    low = sum(1 for r in results if r.tracking_status == TrackingStatus.LOW_CONFIDENCE)
    return {
        "frames": len(results),
        "ok_frames": len(ok),
        "low_confidence_frames": low,
        "lost_frames": lost,
        "max_total_displacement_px": max_r.total_displacement_pixels,
        "max_total_at_frame": max_r.frame_number,
        "max_total_at_timestamp": max_r.timestamp_seconds,
        "mean_total_displacement_px": (sum(totals) / len(totals)) if totals else 0.0,
    }


def append_mm_to_csv(csv_path: str, ppm: float) -> None:
    """Read an existing displacement CSV and add ``*_mm`` columns in-place.

    Skips files that already have mm columns (idempotent).
    """
    import csv as _csv

    if ppm <= 0:
        raise ValueError("pixels-per-mm must be > 0.")

    with open(csv_path, newline="") as fh:
        reader = _csv.DictReader(fh)
        rows = list(reader)
        existing_cols = list(reader.fieldnames or [])

    if not rows:
        return
    if "displacement_x_mm" in existing_cols:
        return  # already converted

    px_mm_pairs = [
        ("displacement_x_pixels", "displacement_x_mm"),
        ("displacement_y_pixels", "displacement_y_mm"),
        ("total_displacement_pixels", "total_displacement_mm"),
    ]

    # Build new column order: insert mm col right after each px col.
    new_cols = list(existing_cols)
    for px_col, mm_col in px_mm_pairs:
        if px_col in new_cols:
            new_cols.insert(new_cols.index(px_col) + 1, mm_col)

    for row in rows:
        for px_col, mm_col in px_mm_pairs:
            if px_col in row:
                try:
                    row[mm_col] = round(float(row[px_col]) / ppm, 5)
                except (ValueError, ZeroDivisionError):
                    row[mm_col] = ""

    with open(csv_path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=new_cols)
        writer.writeheader()
        writer.writerows(rows)
