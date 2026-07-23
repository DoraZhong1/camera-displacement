"""CSV export, graph generation, and HTML report."""

from __future__ import annotations

import base64
import csv
import os
import webbrowser
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(
    camera_results: List[Tuple[str, dict, str, Optional[str]]],
    output_path: str,
    video_name: str = "",
    open_browser: bool = True,
) -> str:
    """Write a self-contained HTML summary report and optionally open it.

    Parameters
    ----------
    camera_results:
        List of ``(label, summary_dict, output_dir, annotated_video_path)``.
        ``summary_dict`` is the dict returned by :func:`summarize`.
    output_path:
        Where to write the ``.html`` file.
    video_name:
        Source video filename shown in the report header.
    open_browser:
        If ``True``, open the report in the default browser after writing.
    """

    def _img_tag(png_path: str) -> str:
        """Return an <img> tag with the PNG embedded as base64."""
        if not os.path.isfile(png_path):
            return ""
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px;">'

    def _quality_badge(pct: int) -> str:
        color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
        return (
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-weight:600;font-size:0.9em;">{pct}%</span>'
        )

    camera_blocks = []
    for label, s, out_dir, _ in camera_results:
        peak_px = s.get("max_total_displacement_px", 0.0)
        mean_px = s.get("mean_total_displacement_px", 0.0)
        peak_frame = s.get("max_total_at_frame", 0)
        peak_t = s.get("max_total_at_timestamp", 0.0)
        ok = s.get("ok_frames", 0)
        low = s.get("low_confidence_frames", 0)
        lost = s.get("lost_frames", 0)
        total = s.get("frames", 0)
        quality_pct = int(100 * ok / total) if total else 0

        # Collect graph images from the output dir.
        graph_tags = ""
        for fname in [
            "absolute_total_displacement_vs_time.png",
            "absolute_displacement_x_vs_time.png",
            "absolute_displacement_y_vs_time.png",
            "absolute_rotation_vs_time.png",
            "absolute_confidence_vs_time.png",
            "absolute_overview.png",
        ]:
            tag = _img_tag(os.path.join(out_dir, fname))
            if tag:
                graph_tags += f'<div style="margin-bottom:12px;">{tag}</div>\n'

        graphs_section = (
            f'<h3 style="margin-top:24px;">Graphs</h3>{graph_tags}'
            if graph_tags
            else '<p style="color:#6b7280;font-style:italic;">No graphs — rerun with <code>--graphs</code> to generate them.</p>'
        )

        camera_blocks.append(f"""
        <div style="background:#1e293b;border-radius:12px;padding:24px;margin-bottom:28px;">
          <h2 style="margin:0 0 16px;color:#f8fafc;font-size:1.3em;letter-spacing:0.03em;">
            {label}
          </h2>
          <table style="border-collapse:collapse;width:100%;color:#e2e8f0;font-size:0.95em;">
            <tr>
              <td style="padding:6px 16px 6px 0;color:#94a3b8;">Peak displacement</td>
              <td style="padding:6px 0;font-weight:600;color:#f8fafc;">{peak_px:.2f} px</td>
              <td style="padding:6px 16px;color:#94a3b8;">at frame {peak_frame} ({peak_t:.2f} s)</td>
            </tr>
            <tr>
              <td style="padding:6px 16px 6px 0;color:#94a3b8;">Mean displacement</td>
              <td style="padding:6px 0;font-weight:600;color:#f8fafc;">{mean_px:.2f} px</td>
              <td></td>
            </tr>
            <tr>
              <td style="padding:6px 16px 6px 0;color:#94a3b8;">Tracking quality</td>
              <td style="padding:6px 0;">{_quality_badge(quality_pct)}</td>
              <td style="padding:6px 16px;color:#64748b;font-size:0.88em;">
                {ok} OK &nbsp;/&nbsp; {low} low-conf &nbsp;/&nbsp; {lost} lost &nbsp;(of {total} frames)
              </td>
            </tr>
          </table>
          <div style="margin-top:16px;font-size:0.82em;color:#64748b;">
            Output folder: <code style="color:#93c5fd;">{os.path.abspath(out_dir)}</code>
          </div>
          <div style="margin-top:20px;">{graphs_section}</div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Camera Displacement Report</title>
  <style>
    body {{ margin:0; padding:32px; font-family:'Segoe UI',system-ui,sans-serif;
           background:#0f172a; color:#e2e8f0; }}
    h1   {{ font-size:1.6em; margin:0 0 4px; color:#f8fafc; }}
    code {{ background:#1e293b; padding:1px 6px; border-radius:4px; font-size:0.9em; }}
  </style>
</head>
<body>
  <h1>Camera Displacement Report</h1>
  <p style="color:#64748b;margin:0 0 28px;">
    {"Source: <code>" + video_name + "</code>" if video_name else ""}
  </p>
  {"".join(camera_blocks)}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    if open_browser:
        abs_path = os.path.abspath(output_path)
        # In WSL there is no Linux browser; convert to a Windows path and use explorer.exe.
        import subprocess, shutil
        if shutil.which("explorer.exe"):
            try:
                win_path = subprocess.check_output(
                    ["wslpath", "-w", abs_path], text=True
                ).strip()
                subprocess.Popen(["explorer.exe", win_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass  # silently skip if conversion fails
        else:
            webbrowser.open(f"file://{abs_path}")

    return output_path
