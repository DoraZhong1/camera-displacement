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


_DARK_BG      = "#0f172a"
_DARK_BG      = "#f0f4f8"
_PANEL_BG     = "#ffffff"
_GRID_COLOR   = "#cbd5e1"
_TEXT_COLOR   = "#0f172a"
_MUTED_COLOR  = "#334155"
_PALETTE = {
    "dx":    "#0284c7",   # blue
    "dy":    "#ea580c",   # orange
    "total": "#dc2626",   # red
    "rot":   "#16a34a",   # green
    "conf":  "#7c3aed",   # violet
}


def _apply_dark_theme(fig, axes_list):
    """Apply a consistent light theme to a figure and all its axes."""
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes_list:
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_MUTED_COLOR, which="both", labelsize=9)
        ax.xaxis.label.set_color(_MUTED_COLOR)
        ax.yaxis.label.set_color(_MUTED_COLOR)
        ax.title.set_color(_TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COLOR)
        ax.grid(True, color=_GRID_COLOR, linewidth=0.8, linestyle="--", alpha=1.0)


def generate_plots(results: List[FrameResult], out_dir: str, prefix: str = "") -> List[str]:
    """Create styled time-series graphs. Returns saved file paths."""
    if not results:
        return []
    os.makedirs(out_dir, exist_ok=True)
    s = _series(results)

    specs = [
        ("dx",    "Horizontal Displacement  Δx (px)",  "displacement_x_vs_time.png"),
        ("dy",    "Vertical Displacement  Δy (px)",    "displacement_y_vs_time.png"),
        ("total", "Total Displacement  √(Δx²+Δy²)  (px)", "total_displacement_vs_time.png"),
        ("rot",   "In-Plane Rotation of Camera (degrees)",  "rotation_vs_time.png"),
        ("conf",  "Tracking Confidence",                "confidence_vs_time.png"),
    ]

    saved = []
    for key, ylabel, fname in specs:
        color = _PALETTE[key]
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.fill_between(s["t"], s[key], alpha=0.15, color=color)
        ax.plot(s["t"], s[key], color=color, linewidth=1.6)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel + "  vs  Time", fontsize=12, fontweight="bold", pad=10)
        if key == "conf":
            ax.set_ylim(-0.02, 1.02)
        _apply_dark_theme(fig, [ax])
        fig.tight_layout(pad=1.8)
        path = os.path.join(out_dir, prefix + fname)
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(path)

    # Overview dashboard (2x2 grid).
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_xy   = axes[0][0]
    ax_tot  = axes[0][1]
    ax_rot  = axes[1][0]
    ax_conf = axes[1][1]

    fig.suptitle("Camera Displacement Overview", fontsize=15, fontweight="bold",
                 color=_TEXT_COLOR, y=0.98)

    # Top-left: Δx and Δy
    ax_xy.fill_between(s["t"], s["dx"], alpha=0.18, color=_PALETTE["dx"])
    ax_xy.fill_between(s["t"], s["dy"], alpha=0.18, color=_PALETTE["dy"])
    ax_xy.plot(s["t"], s["dx"], label="Δx (horizontal)", color=_PALETTE["dx"], linewidth=1.5)
    ax_xy.plot(s["t"], s["dy"], label="Δy (vertical)",   color=_PALETTE["dy"], linewidth=1.5)
    ax_xy.set_title("Horizontal & Vertical Displacement", fontsize=11, fontweight="bold")
    ax_xy.set_ylabel("Displacement (px)", fontsize=10)
    ax_xy.set_xlabel("Time (s)", fontsize=10)
    leg = ax_xy.legend(frameon=True, fontsize=9)
    leg.get_frame().set_facecolor(_PANEL_BG)
    leg.get_frame().set_edgecolor(_GRID_COLOR)
    for txt in leg.get_texts():
        txt.set_color(_TEXT_COLOR)

    # Top-right: total displacement
    ax_tot.fill_between(s["t"], s["total"], alpha=0.18, color=_PALETTE["total"])
    ax_tot.plot(s["t"], s["total"], color=_PALETTE["total"], linewidth=1.5)
    ax_tot.set_title("Total Displacement  √(Δx²+Δy²)", fontsize=11, fontweight="bold")
    ax_tot.set_ylabel("Displacement (px)", fontsize=10)
    ax_tot.set_xlabel("Time (s)", fontsize=10)

    # Bottom-left: rotation
    ax_rot.fill_between(s["t"], s["rot"], alpha=0.18, color=_PALETTE["rot"])
    ax_rot.plot(s["t"], s["rot"], color=_PALETTE["rot"], linewidth=1.5)
    ax_rot.set_title("In-Plane Camera Rotation", fontsize=11, fontweight="bold")
    ax_rot.set_ylabel("Rotation (°)", fontsize=10)
    ax_rot.set_xlabel("Time (s)", fontsize=10)

    # Bottom-right: confidence
    ax_conf.fill_between(s["t"], s["conf"], alpha=0.18, color=_PALETTE["conf"])
    ax_conf.plot(s["t"], s["conf"], color=_PALETTE["conf"], linewidth=1.5)
    ax_conf.set_title("Tracking Confidence", fontsize=11, fontweight="bold")
    ax_conf.set_ylabel("Confidence (0–1)", fontsize=10)
    ax_conf.set_xlabel("Time (s)", fontsize=10)
    ax_conf.set_ylim(-0.02, 1.02)

    _apply_dark_theme(fig, [ax_xy, ax_tot, ax_rot, ax_conf])
    fig.tight_layout(pad=2.0, w_pad=3.0, h_pad=3.0)
    fig.subplots_adjust(top=0.92)

    path = os.path.join(out_dir, prefix + "overview.png")
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
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
        abs_png = os.path.abspath(png_path)
        if not os.path.isfile(abs_png):
            return ""
        with open(abs_png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;display:block;">'

    def _quality_badge(pct: int) -> str:
        if pct >= 80:
            color, bg = "#86efac", "#14532d"
        elif pct >= 50:
            color, bg = "#fde68a", "#78350f"
        else:
            color, bg = "#fca5a5", "#7f1d1d"
        return (
            f'<span style="background:{bg};color:{color};padding:3px 12px;'
            f'border-radius:999px;font-weight:700;font-size:0.85em;letter-spacing:0.04em;">'
            f'{pct}%</span>'
        )

    def _stat_card(label: str, value: str, sub: str = "") -> str:
        return f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                    padding:16px 20px;flex:1;min-width:160px;">
          <div style="color:#64748b;font-size:0.78em;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
          <div style="color:#f8fafc;font-size:1.4em;font-weight:700;line-height:1.2;">{value}</div>
          {f'<div style="color:#64748b;font-size:0.8em;margin-top:4px;">{sub}</div>' if sub else ''}
        </div>"""

    camera_blocks = []
    for label, s, out_dir, _ in camera_results:
        peak_px  = s.get("max_total_displacement_px", 0.0)
        mean_px  = s.get("mean_total_displacement_px", 0.0)
        peak_frame = s.get("max_total_at_frame", 0)
        peak_t   = s.get("max_total_at_timestamp", 0.0)
        ok       = s.get("ok_frames", 0)
        low      = s.get("low_confidence_frames", 0)
        lost     = s.get("lost_frames", 0)
        total    = s.get("frames", 0)
        quality_pct = int(100 * ok / total) if total else 0

        abs_out_dir = os.path.abspath(out_dir)

        # ── Overview image (large, prominent) ─────────────────────────────
        overview_tag = _img_tag(os.path.join(abs_out_dir, "absolute_overview.png"))

        # ── Individual graph cards ─────────────────────────────────────────
        individual_cards = ""
        graph_specs = [
            ("absolute_total_displacement_vs_time.png", "Total Displacement  √(Δx²+Δy²)"),
            ("absolute_displacement_x_vs_time.png",     "Horizontal Displacement  Δx"),
            ("absolute_displacement_y_vs_time.png",     "Vertical Displacement  Δy"),
            ("absolute_rotation_vs_time.png",            "In-Plane Camera Rotation"),
            ("absolute_confidence_vs_time.png",          "Confidence"),
        ]
        for fname, gtitle in graph_specs:
            tag = _img_tag(os.path.join(abs_out_dir, fname))
            if tag:
                individual_cards += f"""
                <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                            padding:14px;margin-bottom:4px;">
                  <div style="color:#94a3b8;font-size:0.78em;text-transform:uppercase;
                              letter-spacing:0.07em;margin-bottom:10px;">{gtitle}</div>
                  {tag}
                </div>"""

        graphs_section = (
            f"""
            <div style="margin-top:28px;">
              <h3 style="color:#94a3b8;font-size:0.82em;text-transform:uppercase;
                         letter-spacing:0.1em;margin:0 0 16px;">Overview Dashboard</h3>
              <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;">
                {overview_tag}
              </div>
            </div>
            <div style="margin-top:24px;">
              <h3 style="color:#94a3b8;font-size:0.82em;text-transform:uppercase;
                         letter-spacing:0.1em;margin:0 0 16px;">Individual Channels</h3>
              {individual_cards}
            </div>"""
            if overview_tag or individual_cards
            else '<p style="color:#475569;font-style:italic;margin-top:20px;">No graphs found — check that the analysis ran successfully.</p>'
        )

        stat_cards = "".join([
            _stat_card("Peak Displacement", f"{peak_px:.2f} px", f"frame {peak_frame} &nbsp;/&nbsp; {peak_t:.2f} s"),
            _stat_card("Mean Displacement", f"{mean_px:.2f} px"),
            _stat_card("Tracking Quality", _quality_badge(quality_pct),
                       f"{ok} OK &nbsp;·&nbsp; {low} low &nbsp;·&nbsp; {lost} lost &nbsp;/ {total}"),
        ])

        camera_blocks.append(f"""
        <section style="background:#1e293b;border:1px solid #334155;border-radius:14px;
                        padding:28px 32px;margin-bottom:32px;">
          <h2 style="margin:0 0 22px;color:#f8fafc;font-size:1.2em;font-weight:700;
                     letter-spacing:0.04em;display:flex;align-items:center;gap:10px;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:#38bdf8;"></span>
            {label}
          </h2>
          <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;">
            {stat_cards}
          </div>
          <div style="margin-top:10px;font-size:0.78em;color:#475569;">
            Output: <code style="color:#7dd3fc;background:#0f172a;padding:1px 6px;border-radius:4px;">{os.path.abspath(out_dir)}</code>
          </div>
          {graphs_section}
        </section>
        """)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Displacement Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #020817;
      color: #e2e8f0;
      min-height: 100vh;
    }}
    .topbar {{
      background: #0f172a;
      border-bottom: 1px solid #1e293b;
      padding: 18px 40px;
      display: flex;
      align-items: center;
      gap: 14px;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .topbar-dot {{
      width: 10px; height: 10px; border-radius: 50%;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      flex-shrink: 0;
    }}
    .topbar h1 {{
      font-size: 1.05em;
      font-weight: 700;
      color: #f8fafc;
      letter-spacing: 0.02em;
    }}
    .topbar .sub {{
      margin-left: auto;
      font-size: 0.82em;
      color: #475569;
    }}
    .topbar code {{
      background: #1e293b;
      color: #7dd3fc;
      padding: 2px 8px;
      border-radius: 5px;
      font-size: 0.88em;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    code {{
      background: #1e293b;
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 0.88em;
      color: #7dd3fc;
    }}
    img {{ border-radius: 8px; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-dot"></div>
    <h1>Camera Displacement Report</h1>
    {"<span class='sub'>Source: <code>" + video_name + "</code></span>" if video_name else ""}
  </div>
  <div class="container">
    {"".join(camera_blocks)}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    if open_browser:
        abs_path = os.path.abspath(output_path)
        # Copy to Windows Downloads so the user can open it easily.
        import subprocess, shutil
        win_downloads = None
        try:
            win_downloads = subprocess.check_output(
                ["wslpath", "-u", "C:\\Users\\smile\\Downloads"], text=True
            ).strip()
        except Exception:
            pass
        if win_downloads and os.path.isdir(win_downloads):
            dest = os.path.join(win_downloads, os.path.basename(abs_path))
            shutil.copy2(abs_path, dest)
            print(f"\nReport copied to Downloads: C:\\Users\\smile\\Downloads\\{os.path.basename(abs_path)}")
            try:
                win_path = subprocess.check_output(
                    ["wslpath", "-w", dest], text=True
                ).strip()
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "", win_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        elif shutil.which("explorer.exe"):
            try:
                win_path = subprocess.check_output(
                    ["wslpath", "-w", abs_path], text=True
                ).strip()
                subprocess.Popen(["explorer.exe", win_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        else:
            webbrowser.open(f"file://{abs_path}")

    return output_path
