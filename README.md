# Camera Displacement Analyzer

Measures camera movement / instability from a recorded video by tracking a
**physically stationary reference object** across frames. Because the reference
object cannot move relative to the camera's enclosure, any *apparent* motion of
that object between frames is treated as motion of the camera. The tool converts
that apparent motion into estimated **camera displacement** (Δx, Δy, total,
rotation) in pixels, and optionally millimetres.

## Method (and why)

**Absolute displacement (Mode 2 — the primary result): ORB/SIFT feature
matching directly against the baseline frame + RANSAC partial-affine.**
Every frame is registered *directly against the baseline* rather than by summing
frame-to-frame deltas, so tracking error never accumulates as drift. RANSAC
rejects bad matches (lighting/blur/partial occlusion) and its inlier count
provides a natural tracking-confidence signal. Because each frame is matched to
the baseline independently, the tracker **self-heals** after a temporary loss.

**Consecutive movement (Mode 1): Shi-Tomasi corners + Lucas-Kanade optical
flow.** Points seeded in the ROI are tracked forward frame-to-frame, giving the
per-step motion cheaply and accurately.

For each frame both pipelines estimate a partial-affine transform (translation +
rotation + uniform scale), map the baseline ROI centre through it to get the
apparent object shift, and report the **camera displacement as the negation of
that shift**. Total displacement = √(Δx² + Δy²).

The tool distinguishes: **translation** vs **rotation** (separate columns);
**tracking failure** (`LOST`), **temporary movement** vs **permanent
displacement** (visible in the absolute-vs-time graph), and flags
**`LOW_CONFIDENCE`** frames instead of reporting misleading numbers.

## Installation

Requires Python 3.9–3.12 on macOS or Windows.

```bash
# from the project folder
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> Install `opencv-contrib-python` (in `requirements.txt`), **not**
> `opencv-python`, so that SIFT and the interactive ROI window are available.

## Usage

### Interactive (recommended)

```bash
python main.py
```

You will be prompted to: pick the video → a window shows the baseline frame →
drag a box around the stationary reference object (press **ENTER**) → optionally
enter calibration → results are written to the output folder.

### Non-interactive / headless

```bash
python main.py --video clip.mp4 --roi 620 360 120 90 \
    --output results --baseline-frame 0 --detector ORB --ppm 12.5
```

Useful flags (`python main.py --help` for all):

| Flag | Meaning |
|------|---------|
| `--roi X Y W H` | Reference-object box (skip the GUI) |
| `--baseline-frame N` | Use frame N as the baseline |
| `--detector ORB\|SIFT` | Feature detector for absolute registration |
| `--min-inliers`, `--min-inlier-ratio` | Reliability thresholds |
| `--no-video` | Skip the annotated video (faster) |
| `--ppm V` | Manual pixels-per-mm calibration |
| `--known-dimension PX MM` | Known length: PX pixels = MM millimetres |
| `--checkerboard COLS ROWS SQUARE_MM` | Calibrate from a checkerboard on the baseline frame |
| `--interactive` | Force prompts even with flags |

## Outputs

Written to the chosen output folder:

- `absolute_displacement.csv` — **primary**, each frame vs baseline.
- `consecutive_displacement.csv` — each frame vs previous frame.
- Graphs (PNG): horizontal, vertical, total displacement, rotation, and
  tracking confidence vs time, plus an `overview` composite (for each mode).
- `annotated.mp4` — baseline box, current (shifted) box, tracked points, a live
  Δx/Δy/total/rotation/confidence readout and an unreliable-tracking warning.

CSV columns: `frame_number, timestamp_seconds, displacement_x_pixels,
displacement_y_pixels, total_displacement_pixels, rotation_degrees, scale,
number_of_matched_features, number_of_inlier_features, tracking_confidence,
tracking_status`. When calibrated, `*_mm` columns are added.

`tracking_status` ∈ `OK`, `LOW_CONFIDENCE` (below the inlier threshold — treat
with caution), `LOST` (registration failed; no displacement reported).

## Calibration and physical units

A single 2-D video measures displacement **in the image plane**. A
pixels-per-millimetre factor (however obtained) is only physically accurate at
the **depth of the reference object**. Computing true 3-D camera translation in
millimetres additionally requires camera intrinsic calibration **and** a known
camera-to-object distance. The mm figures here are image-plane estimates valid
at the reference object's depth — treat them accordingly.

## Handling difficult conditions

- **Lighting changes / noise / mild blur / focus drift:** feature matching plus
  RANSAC tolerates these; low-quality frames simply yield fewer inliers.
- **Partial obstruction / low-feature objects:** confidence drops and the frame
  is marked `LOW_CONFIDENCE`; enlarge the ROI or try `--detector SIFT`.
- **Temporary tracking loss:** flagged `LOST`; the next good frame recovers
  automatically because matching is always against the baseline.
- **Slight scale/perspective change:** captured by the `scale` column; the
  partial-affine model absorbs small changes.

## Module layout

```
camera_displacement/
  video_io.py      load video, frame iteration, metadata
  roi_selector.py  ROI selection (GUI) + validation + masks
  tracker.py       ORB baseline + LK consecutive tracking, transform, confidence
  calibration.py   pixels <-> mm (manual / known dimension / checkerboard)
  reporting.py     CSV export + graphs + summary stats
  annotate.py      annotated output video
  analyzer.py      single-pass orchestration
main.py            CLI workflow
```

## Limitations / assumptions

- The reference object is roughly planar, near the centre, and textured enough
  to yield features. Featureless objects reduce reliability (flagged, not
  silently wrong).
- Rotation sign convention: camera rotation is reported as the negation of the
  reference object's apparent rotation; magnitude is what matters most.
- Very large/fast motion between frames can exceed the optical-flow window
  (Mode 1); the absolute mode (Mode 2) is unaffected as long as enough baseline
  features are still visible.
