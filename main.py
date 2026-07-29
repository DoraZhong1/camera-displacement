#!/usr/bin/env python3
"""Command-line workflow for the Camera Displacement Analyzer.

Interactive workflow (default):

    python main.py

Fully specified / headless run (no GUI):

    python main.py --video clip.mp4 --roi 620 360 120 90 \
        --output out --ppm 12.5 --no-video

Run ``python main.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import os
import sys

# Suppress Qt font/platform noise before any GUI library is imported.
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")
# Silence noisy fontconfig / Pango warnings from OpenCV on WSL.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from typing import Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box as rich_box

from camera_displacement.analyzer import AnalyzerConfig, DisplacementAnalyzer
from camera_displacement.calibration import Calibration
from camera_displacement.roi_selector import select_roi_interactive, validate_roi
from camera_displacement.tracker import count_roi_features
from camera_displacement.video_io import VideoError, VideoLoader, parse_layout
from camera_displacement.reporting import (
    append_mm_to_csv,
    generate_html_report,
    generate_combined_plots,
    load_csv_series,
    generate_comparison_plots,
    generate_comparison_report,
    generate_multi_camera_comparison_report,
    generate_overlay_plots,
    generate_overlay_report,
)

console = Console()

ROI = Tuple[int, int, int, int]


def _prompt(msg: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    resp = input(f"{msg}{suffix}: ").strip()
    return resp if resp else (default or "")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure camera displacement by tracking a stationary reference object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", help="Path to the input video file.")
    p.add_argument(
        "--videos", nargs="+", metavar="VIDEO",
        help=(
            "Two or more videos to analyze and overlay on a single pair of "
            "graphs (total displacement + in-plane rotation). Requires "
            "--layout. Every camera of every video is drawn on both graphs."
        ),
    )
    p.add_argument(
        "--video-labels", nargs="+", metavar="LABEL",
        help="Legend labels for each --videos entry (defaults to the filenames).",
    )
    p.add_argument("--output", default="output", help="Output folder.")
    p.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Reference-object bounding box. Omit for interactive selection.",
    )
    p.add_argument("--baseline-frame", type=int, default=0, help="Baseline frame index.")
    p.add_argument("--detector", choices=["ORB", "SIFT"], default="ORB",
                   help="Feature detector for absolute registration.")
    p.add_argument("--max-features", type=int, default=2000)
    p.add_argument("--min-inliers", type=int, default=12,
                   help="Minimum RANSAC inliers for reliable tracking.")
    p.add_argument("--min-inlier-ratio", type=float, default=0.30)
    p.add_argument("--no-video", action="store_true", help="Skip annotated-video output.")

    # Calibration (choose at most one).
    cal = p.add_argument_group("calibration (optional, choose one)")
    cal.add_argument("--ppm", type=float, help="Manual pixels-per-millimetre.")
    cal.add_argument("--known-dimension", nargs=2, type=float, metavar=("PX", "MM"),
                     help="A known length: PX pixels correspond to MM millimetres.")
    cal.add_argument("--checkerboard", nargs=3, metavar=("COLS", "ROWS", "SQUARE_MM"),
                     help="Inner-corner cols, rows and square size (mm) on baseline frame.")

    p.add_argument("--interactive", action="store_true",
                   help="Force interactive prompts even if flags are given.")
    p.add_argument("--graphs", action="store_true",
                   help="Generate PNG time-series graphs in the output folder (off by default).")

    # Graph filtering.
    p.add_argument("--no-hv-displacement", action="store_true",
                   help="Exclude horizontal and vertical displacement (Δx, Δy) graphs.")
    p.add_argument("--no-confidence", action="store_true",
                   help="Exclude tracking confidence graph.")

    # Multi-camera composite video.
    mc = p.add_argument_group("multi-camera (composite video)")
    mc.add_argument(
        "--layout",
        metavar="COLSxROWS",
        help=(
            "Grid layout of camera feeds inside the video, e.g. '2x2'. "
            "When set, the video is split into individual camera sub-frames "
            "and each is analyzed separately."
        ),
    )
    mc.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of active camera slots to process (left-to-right, "
            "top-to-bottom). Defaults to all slots in the grid. "
            "E.g. use 3 for a 2x2 grid where the bottom-right slot is empty."
        ),
    )
    mc.add_argument(
        "--combine-cameras",
        action="store_true",
        help=(
            "Overlay all cameras on the same graphs instead of one graph set "
            "per camera. Per-camera sections are still included in the report."
        ),
    )

    # Comparison mode: load pre-analyzed CSVs and overlay them on shared graphs.
    cmp = p.add_argument_group("comparison mode (compare pre-analyzed results)")
    cmp.add_argument(
        "--compare", nargs="+", metavar="DIR",
        help=(
            "Two or more output directories to compare. When given, skips "
            "video analysis and only generates comparison plots + report."
        ),
    )
    cmp.add_argument(
        "--compare-labels", nargs="+", metavar="LABEL",
        help="Display labels for each compared directory (same order as --compare).",
    )
    cmp.add_argument(
        "--compare-camera", metavar="CAMERA",
        help=(
            "Camera sub-folder inside each output dir, e.g. camera_0.  "
            "Omit when the CSV is at the root of the output dir."
        ),
    )
    cmp.add_argument(
        "--compare-output", default="out-compare",
        help="Output folder for comparison plots and HTML report.",
    )

    return p.parse_args(argv)


def build_calibration(args, baseline_frame) -> Calibration:
    provided = [bool(args.ppm), bool(args.known_dimension), bool(args.checkerboard)]
    if sum(provided) > 1:
        print("Warning: multiple calibration options given; using the first one.", file=sys.stderr)
    try:
        if args.ppm:
            return Calibration.from_ppm(args.ppm)
        if args.known_dimension:
            px, mm = args.known_dimension
            return Calibration.from_known_dimension(px, mm)
        if args.checkerboard:
            cols, rows, square = args.checkerboard
            return Calibration.from_checkerboard(
                baseline_frame, (int(cols), int(rows)), float(square)
            )
    except (ValueError, Exception) as exc:  # keep going uncalibrated on failure
        print(f"Calibration failed ({exc}). Continuing in pixels only.", file=sys.stderr)
    return Calibration.none()


def interactive_calibration() -> Calibration:
    print("\nCalibration (optional). Choose:")
    print("  [1] Manual pixels-per-mm")
    print("  [2] Known dimension (pixels = millimetres)")
    print("  [Enter] Skip (pixels only)")
    choice = _prompt("Selection", "")
    try:
        if choice == "1":
            return Calibration.from_ppm(float(_prompt("pixels per mm")))
        if choice == "2":
            px = float(_prompt("pixel length"))
            mm = float(_prompt("real length in mm"))
            return Calibration.from_known_dimension(px, mm)
    except ValueError:
        print("Invalid input; continuing in pixels only.", file=sys.stderr)
    return Calibration.none()


def progress_bar(current: int, total: int) -> None:
    total = max(total, current)
    width = 30
    filled = int(width * current / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r  [{bar}] {current}/{total} frames")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _print_summary(outputs, calibration, label: str = "") -> None:
    s = outputs.absolute_summary
    peak_px = s.get("max_total_displacement_px", 0.0)
    mean_px = s.get("mean_total_displacement_px", 0.0)
    peak_frame = s.get("max_total_at_frame", 0)
    peak_t = s.get("max_total_at_timestamp", 0.0)
    peak_rot = s.get("peak_abs_rotation_degrees", 0.0)
    mean_rot = s.get("mean_abs_rotation_degrees", 0.0)
    ok = s.get("ok_frames", 0)
    low = s.get("low_confidence_frames", 0)
    lost = s.get("lost_frames", 0)
    total = s.get("frames", 0)
    quality_pct = int(100 * ok / total) if total else 0

    if quality_pct >= 80:
        quality_color = "green"
    elif quality_pct >= 50:
        quality_color = "yellow"
    else:
        quality_color = "red"

    table = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold white")
    table.add_column("Detail", style="dim")

    table.add_row("Peak displacement",
                  f"{peak_px:.2f} px",
                  f"frame {peak_frame}  /  t = {peak_t:.2f} s")
    table.add_row("Mean displacement", f"{mean_px:.2f} px", "")

    if calibration.is_calibrated:
        table.add_row("Peak displacement",
                      f"{calibration.px_to_mm(peak_px):.3f} mm", "(calibrated)")
        table.add_row("Mean displacement",
                      f"{calibration.px_to_mm(mean_px):.3f} mm", "(calibrated)")

    table.add_row("Peak |rotation|", f"{peak_rot:.3f}°", "")
    table.add_row("Mean |rotation|", f"{mean_rot:.3f}°", "")

    table.add_row(
        "Tracking quality",
        f"[{quality_color}]{quality_pct}% reliable[/{quality_color}]",
        f"{ok} OK / {low} low-conf / {lost} lost  (of {total} frames)",
    )

    title = f"Results — {label}" if label else "Results"
    console.print(Panel(table, title=f"[bold cyan]{title}[/bold cyan]",
                        border_style="cyan", padding=(1, 2)))


def _open_plots_in_windows(png_paths: list) -> None:
    """Open each PNG in the list with the Windows default image viewer."""
    import subprocess
    if not png_paths:
        return
    for path in png_paths:
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", os.path.abspath(path)], text=True
            ).strip()
            subprocess.Popen(
                ["explorer.exe", win_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _select_roi_with_retry(
    frame,
    window_title: str = "Select reference object",
    detector: str = "ORB",
    min_features: int = 12,
) -> Optional[ROI]:
    """Interactive ROI selection that re-prompts if the region has too few features."""
    while True:
        roi = select_roi_interactive(frame, window_title=window_title)
        if roi is None:
            return None
        roi = validate_roi(roi, frame.shape)
        n = count_roi_features(frame, roi, detector=detector)
        if n >= min_features:
            return roi
        print(
            f"\n  Only {n} trackable features found in that region (need at least {min_features})."
        )
        print("  Try selecting an area with more texture/contrast, or a larger box.")
        again = input("  Re-select? [Y/n]: ").strip().lower()
        if again == "n":
            return roi  # proceed anyway at user's request


def _prompt_mm_conversion(outputs_list, calibration) -> Optional[float]:
    """Ask the user for a pixels-per-mm scale and add mm columns to all CSVs.

    Returns the pixels-per-mm value that was applied, or ``None`` when the
    conversion was skipped (or the run was already calibrated).
    """
    if calibration.is_calibrated:
        return None

    console.print("\n[bold]Convert results to millimetres?[/bold]")
    console.print("  Enter the number of pixels that equal [cyan]1 mm[/cyan] in your video,")
    console.print("  or press [dim]Enter[/dim] to skip.")
    raw = input("  Pixels per mm: ").strip()
    if not raw:
        console.print("  [dim]Skipping mm conversion.[/dim]")
        return None
    try:
        ppm = float(raw)
        if ppm <= 0:
            raise ValueError
    except ValueError:
        console.print("  [red]Invalid value — skipping mm conversion.[/red]")
        return None

    console.print(f"\n  Converting at [cyan]{ppm}[/cyan] px/mm …")
    for label, outputs, _ in outputs_list:
        for csv_path in [outputs.absolute_csv, outputs.consecutive_csv]:
            append_mm_to_csv(csv_path, ppm)
            console.print(f"    [green]✓[/green] Updated: {csv_path}")

        s = outputs.absolute_summary
        peak_mm = s.get("max_total_displacement_px", 0.0) / ppm
        mean_mm = s.get("mean_total_displacement_px", 0.0) / ppm
        console.print(f"  [bold]{label}[/bold]  "
                      f"Peak = [bold white]{peak_mm:.3f} mm[/bold white]   "
                      f"Mean = [bold white]{mean_mm:.3f} mm[/bold white]")
    return ppm


def _build_excluded_series(args) -> set:
    """Return the set of series keys to omit from graphs based on CLI flags."""
    excluded: set = set()
    if getattr(args, "no_hv_displacement", False):
        excluded.update(["dx", "dy"])
    if getattr(args, "no_confidence", False):
        excluded.add("conf")
    return excluded


def _discover_cameras(dirs: list) -> list:
    """Return sorted list of camera_* subdirs found in any of the given output directories."""
    cameras: set = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            if entry.startswith("camera_") and os.path.isdir(os.path.join(d, entry)):
                cameras.add(entry)
    return sorted(cameras)


def _run_compare(args) -> int:
    """Load pre-analyzed CSVs from multiple output dirs and produce comparison plots."""
    dirs = args.compare
    labels = list(args.compare_labels or [])
    # Pad missing labels with directory basenames.
    while len(labels) < len(dirs):
        labels.append(os.path.basename(dirs[len(labels)].rstrip("/\\")))

    camera_arg = getattr(args, "compare_camera", None)
    out_dir = args.compare_output
    excluded = _build_excluded_series(args)

    # Determine which cameras to compare: explicit arg or auto-discover all.
    if camera_arg:
        cameras_to_compare = [camera_arg]
    else:
        cameras_to_compare = _discover_cameras(dirs) or [None]

    console.print(
        f"\n[bold]Comparison mode:[/bold] {len(dirs)} recording(s), "
        f"{len(cameras_to_compare)} camera(s)"
    )

    all_camera_sections = []
    total_plots = 0

    for camera in cameras_to_compare:
        cam_out_dir = os.path.join(out_dir, camera) if camera else out_dir
        camera_label = camera or "root"

        datasets = []
        for label, d in zip(labels, dirs):
            csv_dir = os.path.join(d, camera) if camera else d
            csv_path = os.path.join(csv_dir, "absolute_displacement.csv")
            if not os.path.isfile(csv_path):
                console.print(f"  [yellow]CSV not found — skipping:[/yellow] {csv_path}")
                continue
            series = load_csv_series(csv_path)
            datasets.append((label, series))
            console.print(
                f"  [green]Loaded[/green] [{camera_label}] [bold]{label}[/bold]: {csv_path} "
                f"[dim]({len(series['t'])} frames)[/dim]"
            )

        if not datasets:
            console.print(f"  [yellow]No data found for {camera_label} — skipping.[/yellow]")
            continue

        os.makedirs(cam_out_dir, exist_ok=True)
        console.print(f"\n[bold]Generating comparison plots for {camera_label}…[/bold]")
        plots = generate_comparison_plots(datasets, cam_out_dir, excluded_series=excluded)
        all_camera_sections.append((camera_label, datasets, plots))
        total_plots += len(plots)

    if not all_camera_sections:
        console.print("[red]No comparison data found in the given directories.[/red]")
        return 2

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "comparison_report.html")
    title_str = " vs ".join(labels)
    generate_multi_camera_comparison_report(
        all_camera_sections, report_path,
        title=f"Displacement Comparison — {title_str}",
        open_browser=True,
    )

    out_table = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    out_table.add_column("Type", style="dim", no_wrap=True)
    out_table.add_column("Path", style="cyan")
    out_table.add_row(
        "Comparison plots",
        f"{total_plots} PNG files across {len(all_camera_sections)} camera(s) in {out_dir}/",
    )
    out_table.add_row("HTML report", report_path)
    console.print(Panel(
        out_table,
        title=f"[bold]Comparison → {os.path.abspath(out_dir)}[/bold]",
        border_style="green", padding=(1, 2),
    ))

    # Open overview PNGs in Windows image viewer (cap at first 3 to avoid flooding).
    overviews = []
    for cam_label, _, _ in all_camera_sections:
        cam_dir = os.path.join(out_dir, cam_label) if cam_label != "root" else out_dir
        ov = os.path.join(cam_dir, "comparison_overview.png")
        if os.path.isfile(ov):
            overviews.append(ov)
    if overviews:
        _open_plots_in_windows(overviews[:3])

    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    # Comparison mode: load existing CSVs and overlay them — no video needed.
    if getattr(args, "compare", None):
        return _run_compare(args)

    # Overlay mode: analyze several videos and draw every camera on two graphs.
    if getattr(args, "videos", None):
        return _run_multi_video(args)

    # 1. Video selection.
    video_path = args.video
    if not video_path:
        video_path = _prompt("Path to input video")
    if not video_path:
        print("No video provided.", file=sys.stderr)
        return 2

    try:
        loader = VideoLoader(video_path)
    except VideoError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    info = loader.info
    console.print(f"\n[bold green]Loaded:[/bold green] {os.path.basename(video_path)}")
    console.print(f"  [dim]{info.width}x{info.height} @ {info.fps:.2f} fps, "
                  f"~{info.frame_count} frames, {info.duration_seconds:.1f}s[/dim]")

    # 2. Baseline frame.
    baseline_index = args.baseline_frame
    if args.interactive and not args.video:
        baseline_index = int(_prompt("Baseline frame index", str(baseline_index)) or baseline_index)
    baseline = loader.read_frame(baseline_index)
    if baseline is None:
        print(f"Could not read baseline frame {baseline_index}.", file=sys.stderr)
        loader.release()
        return 2

    loader.release()

    # -----------------------------------------------------------------------
    # Multi-camera mode: composite video with a grid of camera feeds.
    # -----------------------------------------------------------------------
    if args.layout:
        return _run_multi_camera(args, video_path, info, baseline, baseline_index)

    # -----------------------------------------------------------------------
    # Single-camera mode (original behaviour).
    # -----------------------------------------------------------------------
    # 3. ROI selection.
    if args.roi:
        roi = tuple(args.roi)  # type: ignore
        roi = validate_roi(roi, baseline.shape)
    else:
        print("\nSelect the stationary reference object in the window that opens...")
        roi = _select_roi_with_retry(baseline, detector=args.detector,
                                     min_features=args.min_inliers)
        if roi is None:
            print("No ROI selected.", file=sys.stderr)
            return 2
    print(f"  ROI = {roi}")

    # 4. Calibration.
    if args.interactive and not any([args.ppm, args.known_dimension, args.checkerboard]):
        calibration = interactive_calibration()
    else:
        calibration = build_calibration(args, baseline)
    console.print(f"  [dim]Calibration: {calibration.describe()}[/dim]")

    # 5. Output folder + run.
    output_dir = args.output
    if args.interactive and not args.video:
        output_dir = _prompt("Output folder", output_dir) or output_dir

    config = AnalyzerConfig(
        video_path=video_path,
        roi=roi,
        output_dir=output_dir,
        baseline_frame_index=baseline_index,
        detector=args.detector,
        max_features=args.max_features,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        calibration=calibration,
        write_annotated_video=not args.no_video,
        write_graphs=True,
        progress=progress_bar,
        excluded_series=_build_excluded_series(args),
    )

    console.print("\n[bold]Analyzing...[/bold]")
    try:
        outputs = DisplacementAnalyzer(config).run()
    except (ValueError, RuntimeError) as exc:
        console.print(f"\n[red]Error during analysis: {exc}[/red]")
        return 1

    _print_summary(outputs, calibration)
    _prompt_mm_conversion([("camera", outputs, output_dir)], calibration)

    # HTML report (written but not auto-opened — use PNGs instead).
    report_path = os.path.join(output_dir, "report.html")
    generate_html_report(
        [("Camera", outputs.absolute_summary, output_dir, outputs.annotated_video)],
        report_path,
        video_name=os.path.basename(video_path),
        open_browser=False,
    )

    # Open the overview graph in Windows Photos.
    overview = os.path.join(output_dir, "absolute_overview.png")
    if os.path.isfile(overview):
        console.print("\n[bold green]Opening overview graph…[/bold green]")
        _open_plots_in_windows([overview])

    # Output file table.
    out_table = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    out_table.add_column("Type", style="dim", no_wrap=True)
    out_table.add_column("Path", style="cyan")
    out_table.add_row("CSV (absolute)", outputs.absolute_csv)
    out_table.add_row("CSV (consecutive)", outputs.consecutive_csv)
    if outputs.absolute_plots or outputs.consecutive_plots:
        out_table.add_row(
            "Graphs",
            f"{len(outputs.absolute_plots) + len(outputs.consecutive_plots)} PNG files in {output_dir}/",
        )
    if outputs.annotated_video:
        out_table.add_row("Annotated video", outputs.annotated_video)
    out_table.add_row("HTML report", report_path)

    console.print(Panel(out_table,
                        title=f"[bold]Output files → {os.path.abspath(output_dir)}[/bold]",
                        border_style="green", padding=(1, 2)))
    return 0


def _slugify(label: str) -> str:
    """Turn a display label into a filesystem-safe folder name."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return safe.strip("_") or "video"


def _run_multi_video(args) -> int:
    """Analyze several videos and overlay every camera on one pair of graphs."""
    video_paths = args.videos
    labels = list(args.video_labels or [])
    while len(labels) < len(video_paths):
        labels.append(os.path.splitext(os.path.basename(video_paths[len(labels)]))[0])

    if not args.layout:
        print("Error: --videos requires --layout (e.g. --layout 3x1).", file=sys.stderr)
        return 2
    if args.roi:
        console.print("[yellow]Note: --roi is ignored in multi-video mode.[/yellow]")

    output_root = args.output
    # Collected across every video: (video_label, camera_label, results/summary).
    plot_data = []
    stats = []
    all_outputs = []
    last_calibration = Calibration.none()

    for video_path, video_label in zip(video_paths, labels):
        console.rule(f"[bold magenta]{video_label}[/bold magenta]  [dim]{video_path}[/dim]")

        try:
            loader = VideoLoader(video_path)
        except VideoError as exc:
            console.print(f"  [red]Skipping — {exc}[/red]")
            continue

        info = loader.info
        console.print(f"  [dim]{info.width}x{info.height} @ {info.fps:.2f} fps, "
                      f"~{info.frame_count} frames, {info.duration_seconds:.1f}s[/dim]")

        baseline_index = args.baseline_frame
        baseline = loader.read_frame(baseline_index)
        loader.release()
        if baseline is None:
            console.print(f"  [red]Could not read baseline frame {baseline_index}; skipping.[/red]")
            continue

        try:
            regions = parse_layout(args.layout, info.width, info.height, args.num_cameras)
        except ValueError as exc:
            console.print(f"  [red]{exc}[/red]")
            return 2

        calibration = build_calibration(args, baseline)
        console.print(f"  [dim]Calibration: {calibration.describe()}[/dim]")

        video_dir = os.path.join(output_root, _slugify(video_label))

        for region in regions:
            cam_label = f"camera_{region.camera_id}"
            console.print(f"\n[bold cyan]{video_label} / {cam_label}[/bold cyan] "
                          f"[dim](crop x={region.x} y={region.y} "
                          f"w={region.width} h={region.height})[/dim]")

            sub_baseline = region.crop(baseline)
            print(f"Select the stationary reference object for "
                  f"{video_label}/{cam_label} in the window that opens...")
            roi = _select_roi_with_retry(
                sub_baseline,
                window_title=f"[{video_label} / {cam_label}] Select reference object",
                detector=args.detector,
                min_features=args.min_inliers,
            )
            if roi is None:
                console.print(f"  [yellow]No ROI selected; skipping {cam_label}.[/yellow]")
                continue
            roi = validate_roi(roi, sub_baseline.shape)
            print(f"  ROI = {roi}")

            cam_output_dir = os.path.join(video_dir, cam_label)
            config = AnalyzerConfig(
                video_path=video_path,
                roi=roi,
                output_dir=cam_output_dir,
                baseline_frame_index=baseline_index,
                detector=args.detector,
                max_features=args.max_features,
                min_inliers=args.min_inliers,
                min_inlier_ratio=args.min_inlier_ratio,
                calibration=calibration,
                write_annotated_video=not args.no_video,
                # Only the two overlay graphs are wanted — skip per-camera PNGs.
                write_graphs=False,
                progress=progress_bar,
                camera_region=region.as_xywh(),
            )

            console.print(f"  [bold]Analyzing {video_label}/{cam_label}...[/bold]")
            try:
                outputs = DisplacementAnalyzer(config).run()
            except (ValueError, RuntimeError) as exc:
                console.print(f"  [red]Error analyzing {cam_label}: {exc}[/red]")
                continue

            _print_summary(outputs, calibration, label=f"{video_label} / {cam_label}")
            plot_data.append((video_label, cam_label, outputs.absolute_results))
            stats.append((video_label, cam_label, outputs.absolute_summary))
            all_outputs.append((f"{video_label}/{cam_label}", outputs, cam_output_dir))
            last_calibration = calibration

    if not plot_data:
        console.print("\n[red]No cameras were successfully analyzed.[/red]")
        return 1

    # A scale entered at the prompt also switches the graphs over to millimetres.
    entered_ppm = _prompt_mm_conversion(all_outputs, last_calibration)
    if entered_ppm:
        last_calibration = Calibration.from_ppm(entered_ppm)

    console.print("\n[bold]Generating overlay graphs…[/bold]")
    os.makedirs(output_root, exist_ok=True)
    plots = generate_overlay_plots(plot_data, output_root, calibration=last_calibration)

    report_path = os.path.join(output_root, "report.html")
    generate_overlay_report(
        stats, plots, report_path,
        title="Camera Displacement — " + " vs ".join(labels),
        open_browser=False,
    )

    out_table = Table(box=rich_box.SIMPLE, show_header=True, padding=(0, 2))
    out_table.add_column("Video", style="bold magenta", no_wrap=True)
    out_table.add_column("Camera", style="bold cyan", no_wrap=True)
    out_table.add_column("Peak (px)", justify="right")
    out_table.add_column("Mean (px)", justify="right")
    out_table.add_column("Peak |rot| (°)", justify="right")
    out_table.add_column("Mean |rot| (°)", justify="right")
    out_table.add_column("Quality", justify="right")

    for video_label, cam_label, s in stats:
        total = s.get("frames", 0)
        ok = s.get("ok_frames", 0)
        quality_pct = int(100 * ok / total) if total else 0
        qcolor = "green" if quality_pct >= 80 else "yellow" if quality_pct >= 50 else "red"
        out_table.add_row(
            video_label,
            cam_label,
            f"{s.get('max_total_displacement_px', 0.0):.2f}",
            f"{s.get('mean_total_displacement_px', 0.0):.2f}",
            f"{s.get('peak_abs_rotation_degrees', 0.0):.3f}",
            f"{s.get('mean_abs_rotation_degrees', 0.0):.3f}",
            f"[{qcolor}]{quality_pct}%[/{qcolor}]",
        )

    console.print(Panel(
        out_table,
        title=f"[bold]Done — {len(plot_data)} camera feed(s) across "
              f"{len(labels)} video(s)[/bold]",
        border_style="green", padding=(1, 2),
    ))
    for path in plots:
        console.print(f"  [green]✓[/green] [cyan]{os.path.abspath(path)}[/cyan]")
    console.print(f"[bold green]HTML report:[/bold green] {os.path.abspath(report_path)}")

    _open_plots_in_windows(plots)
    return 0


def _run_multi_camera(args, video_path, info, baseline, baseline_index) -> int:
    """Run displacement analysis independently on each camera in a composite video."""
    try:
        regions = parse_layout(args.layout, info.width, info.height, args.num_cameras)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    console.print(f"\n[bold]Multi-camera mode:[/bold] layout {args.layout}, "
                  f"{len(regions)} camera(s) to process.")

    if args.roi:
        console.print("[yellow]Note: --roi is ignored in multi-camera mode.[/yellow]")

    # Shared calibration (same for all cameras unless per-camera calibration is needed).
    calibration = build_calibration(args, baseline)
    console.print(f"  [dim]Calibration: {calibration.describe()}[/dim]")

    excluded_series = _build_excluded_series(args)
    output_root = args.output
    all_outputs = []

    for region in regions:
        cam_label = f"camera_{region.camera_id}"
        console.rule(f"[bold cyan]{cam_label}[/bold cyan]  "
                     f"[dim](crop x={region.x} y={region.y} w={region.width} h={region.height})[/dim]")

        sub_baseline = region.crop(baseline)

        print(f"Select the stationary reference object for {cam_label} "
              "in the window that opens...")
        roi = _select_roi_with_retry(
            sub_baseline,
            window_title=f"[{cam_label}] Select reference object",
            detector=args.detector,
            min_features=args.min_inliers,
        )
        if roi is None:
            console.print(f"  [yellow]No ROI selected for {cam_label}; skipping.[/yellow]")
            continue
        roi = validate_roi(roi, sub_baseline.shape)
        print(f"  ROI = {roi}")

        cam_output_dir = os.path.join(output_root, cam_label)

        config = AnalyzerConfig(
            video_path=video_path,
            roi=roi,
            output_dir=cam_output_dir,
            baseline_frame_index=baseline_index,
            detector=args.detector,
            max_features=args.max_features,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            calibration=calibration,
            write_annotated_video=not args.no_video,
            write_graphs=True,
            progress=progress_bar,
            camera_region=region.as_xywh(),
            excluded_series=excluded_series,
        )

        console.print(f"  [bold]Analyzing {cam_label}...[/bold]")
        try:
            outputs = DisplacementAnalyzer(config).run()
        except (ValueError, RuntimeError) as exc:
            console.print(f"  [red]Error analyzing {cam_label}: {exc}[/red]")
            continue

        _print_summary(outputs, calibration, label=cam_label)
        all_outputs.append((cam_label, outputs, cam_output_dir))

    if not all_outputs:
        console.print("\n[red]No cameras were successfully analyzed.[/red]")
        return 1

    _prompt_mm_conversion(all_outputs, calibration)

    # Combined graphs (all cameras on the same axes).
    combined_dir = None
    if args.combine_cameras:
        combined_dir = args.output
        console.print("\n[bold]Generating combined camera plots…[/bold]")
        combined_data = [
            (label, outputs.absolute_results) for label, outputs, _ in all_outputs
        ]
        generate_combined_plots(combined_data, combined_dir, excluded_series=excluded_series)

    # HTML report (all cameras in one page).
    report_path = os.path.join(args.output, "report.html")
    generate_html_report(
        [(label, outputs.absolute_summary, cam_dir, outputs.annotated_video)
         for label, outputs, cam_dir in all_outputs],
        report_path,
        video_name=os.path.basename(video_path),
        open_browser=False,
        combined_dir=combined_dir,
    )

    # Final summary table.
    out_table = Table(box=rich_box.SIMPLE, show_header=True, padding=(0, 2))
    out_table.add_column("Camera", style="bold cyan", no_wrap=True)
    out_table.add_column("Peak (px)", justify="right")
    out_table.add_column("Mean (px)", justify="right")
    out_table.add_column("Peak |rot| (°)", justify="right")
    out_table.add_column("Mean |rot| (°)", justify="right")
    out_table.add_column("Quality", justify="right")
    out_table.add_column("Output dir", style="dim")

    for label, outputs, cam_dir in all_outputs:
        s = outputs.absolute_summary
        peak = s.get("max_total_displacement_px", 0.0)
        mean = s.get("mean_total_displacement_px", 0.0)
        peak_rot = s.get("peak_abs_rotation_degrees", 0.0)
        mean_rot = s.get("mean_abs_rotation_degrees", 0.0)
        total = s.get("frames", 0)
        ok = s.get("ok_frames", 0)
        quality_pct = int(100 * ok / total) if total else 0
        qcolor = "green" if quality_pct >= 80 else "yellow" if quality_pct >= 50 else "red"
        out_table.add_row(
            label,
            f"{peak:.2f}",
            f"{mean:.2f}",
            f"{peak_rot:.3f}",
            f"{mean_rot:.3f}",
            f"[{qcolor}]{quality_pct}%[/{qcolor}]",
            cam_dir,
        )

    console.print(Panel(
        out_table,
        title=f"[bold]Done — {len(all_outputs)}/{len(regions)} cameras analyzed[/bold]",
        border_style="green", padding=(1, 2),
    ))
    console.print(f"[bold green]HTML report:[/bold green] {os.path.abspath(report_path)}")

    # Open overview PNGs in Windows Photos.
    if args.combine_cameras and combined_dir:
        combined_overview = os.path.join(combined_dir, "combined_overview.png")
        plots_to_open = [combined_overview] if os.path.isfile(combined_overview) else []
    else:
        all_overviews = [
            os.path.join(cam_dir, "absolute_overview.png")
            for _, _, cam_dir in all_outputs
        ]
        plots_to_open = [p for p in all_overviews if os.path.isfile(p)]
    if plots_to_open:
        console.print("\n[bold green]Opening graphs in Windows Photos…[/bold green]")
        _open_plots_in_windows(plots_to_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
