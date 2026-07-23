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

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
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
from camera_displacement.reporting import append_mm_to_csv, generate_html_report

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

    table.add_row(
        "Tracking quality",
        f"[{quality_color}]{quality_pct}% reliable[/{quality_color}]",
        f"{ok} OK / {low} low-conf / {lost} lost  (of {total} frames)",
    )

    title = f"Results — {label}" if label else "Results"
    console.print(Panel(table, title=f"[bold cyan]{title}[/bold cyan]",
                        border_style="cyan", padding=(1, 2)))


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


def _prompt_mm_conversion(outputs_list, calibration) -> None:
    """Ask the user for a pixels-per-mm scale and add mm columns to all CSVs."""
    if calibration.is_calibrated:
        return

    console.print("\n[bold]Convert results to millimetres?[/bold]")
    console.print("  Enter the number of pixels that equal [cyan]1 mm[/cyan] in your video,")
    console.print("  or press [dim]Enter[/dim] to skip.")
    raw = input("  Pixels per mm: ").strip()
    if not raw:
        console.print("  [dim]Skipping mm conversion.[/dim]")
        return
    try:
        ppm = float(raw)
        if ppm <= 0:
            raise ValueError
    except ValueError:
        console.print("  [red]Invalid value — skipping mm conversion.[/red]")
        return

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


def main(argv=None) -> int:
    args = parse_args(argv)

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
        write_graphs=args.graphs,
        progress=progress_bar,
    )

    console.print("\n[bold]Analyzing...[/bold]")
    try:
        outputs = DisplacementAnalyzer(config).run()
    except (ValueError, RuntimeError) as exc:
        console.print(f"\n[red]Error during analysis: {exc}[/red]")
        return 1

    _print_summary(outputs, calibration)
    _prompt_mm_conversion([("camera", outputs, output_dir)], calibration)

    # HTML report.
    report_path = os.path.join(output_dir, "report.html")
    generate_html_report(
        [("Camera", outputs.absolute_summary, output_dir, outputs.annotated_video)],
        report_path,
        video_name=os.path.basename(video_path),
    )

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
            write_graphs=args.graphs,
            progress=progress_bar,
            camera_region=region.as_xywh(),
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

    # HTML report (all cameras in one page).
    report_path = os.path.join(args.output, "report.html")
    generate_html_report(
        [(label, outputs.absolute_summary, cam_dir, outputs.annotated_video)
         for label, outputs, cam_dir in all_outputs],
        report_path,
        video_name=os.path.basename(video_path),
    )

    # Final summary table.
    out_table = Table(box=rich_box.SIMPLE, show_header=True, padding=(0, 2))
    out_table.add_column("Camera", style="bold cyan", no_wrap=True)
    out_table.add_column("Peak (px)", justify="right")
    out_table.add_column("Mean (px)", justify="right")
    out_table.add_column("Quality", justify="right")
    out_table.add_column("Output dir", style="dim")

    for label, outputs, cam_dir in all_outputs:
        s = outputs.absolute_summary
        peak = s.get("max_total_displacement_px", 0.0)
        mean = s.get("mean_total_displacement_px", 0.0)
        total = s.get("frames", 0)
        ok = s.get("ok_frames", 0)
        quality_pct = int(100 * ok / total) if total else 0
        qcolor = "green" if quality_pct >= 80 else "yellow" if quality_pct >= 50 else "red"
        out_table.add_row(
            label,
            f"{peak:.2f}",
            f"{mean:.2f}",
            f"[{qcolor}]{quality_pct}%[/{qcolor}]",
            cam_dir,
        )

    console.print(Panel(
        out_table,
        title=f"[bold]Done — {len(all_outputs)}/{len(regions)} cameras analyzed[/bold]",
        border_style="green", padding=(1, 2),
    ))
    console.print(f"[bold green]HTML report:[/bold green] {os.path.abspath(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
