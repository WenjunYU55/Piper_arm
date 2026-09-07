# Offline distortion verification — 2026-09-07

Worktree: `fix/recorded-session-recovery`, HEAD
`d1941c89920f1313972eed75ab507df2c04f8d4e`. Existing uncommitted recovery and GUI
work was preserved. This follow-up does not change camera settings, motion,
stationary capture, sampling policy, ROS parameters or planner behaviour.

The pending converter already corrected recorded Brown–Conrady distortion.
This follow-up shares that mapping with a diagnostic utility, verifies it against
an independent polynomial projection, rejects inadequate target depth after
remapping, and records input/output calibration explicitly in `validation.json`.
RGB is remapped linearly; aligned depth and binary masks use nearest neighbour.
The output keeps K and uses pinhole projection. Depth remains axial Z, rounded
to millimetres for BundleSDF. Original raw data is untouched.

## What the recorded data shows

The retained first frame has colour calibration
`D=[0.16904977,-0.51425225,-0.00214692,0.00131909,0.47541139]`.
The largest pixel displacement is **7.8462 px**, but the centred target has
**0.0476 px p95 / 0.0774 px maximum** displacement. Lens correction is appropriate,
but its small effect on this target cannot explain the previous large mesh wings.

For the visually selected wooden-board ROI `(100,80)-(250,180)`, 15,000 valid raw
aligned-depth samples give plane RMS **0.4058 mm** with recorded distortion versus
**0.4110 mm** if distortion is ignored. Corrected p95 absolute residual is
**0.8008 mm**. This is a local single-frame consistency measurement, not absolute
accuracy: the board is not a certified flat calibration reference. It neither
rules out errors elsewhere nor calibrates depth bias.

The independently recorded pose audit found up to 74.25 degrees of variation in
`T_base_camera @ T_camera_object` for the nominally stationary target. That is a
tracking/TF/hand-eye consistency diagnostic, not ground truth. Multi-view alignment
needs separate investigation; arbitrary flattening would conceal the problem.

Evidence is retained outside Git at:

`/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-distortion-diagnostic/`

It contains `raw.png`, `rectified.png`, `comparison.png` and `distortion.json`.

## Reproduce

```bash
SESSION=/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-real-scan-142623-headless/native-session.incomplete
python3 tools/bundlesdf/distortion.py "$SESSION" \
  --output /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-distortion-diagnostic \
  --plane-roi 100 80 250 180
python3 tools/bundlesdf/dataset.py "$SESSION" --allow-incomplete \
  --output /tmp/piper-distortion-converted
tools/bundlesdf/python_native.sh tools/bundlesdf/check_upstream_loader.py \
  /tmp/piper-distortion-converted \
  --upstream /home/prl/.local/share/piper-bundlesdf/upstream
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction tests/gui reconstruction/test_gui_support.py
python3 -m flake8 tools/bundlesdf/distortion.py tools/bundlesdf/dataset.py \
  tests/reconstruction/test_distortion.py tests/reconstruction/test_recording.py \
  --max-line-length=110
```

Use new output directories on rerun; existing evidence is never overwritten.
Choose a single planar patch in raw pixel coordinates for `--plane-roi`. Do not
include the object and background together. Omitting the ROI reports pixel
rectification only.

Executed results: baseline recording suite **60 passed**; geometry/recording suite
**65 passed**; final complete reconstruction/GUI suite **253 passed**, no failures;
changed-file lint passed. The first full-suite attempt without sourcing the
workspace failed collection because `piper_mobile_manipulation` was unavailable;
the properly sourced command above passed. All AI YAMLs parse.

The unmodified reader at upstream
`ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1` loaded all **141 frames, 640×480**.
All **424** converted RGB/depth/mask PNGs and K file are byte-identical to the
previous completed actual GPU run's input. The newly added metadata is additive.
No new GPU reconstruction was run for byte-identical input; the prior mesh's
visual-quality failure remains explicitly recorded. This follow-up does not
claim a better mesh, camera calibration, physical testing or a new ROS build.

## Camera-specific interpretation

Use each stream's own intrinsics and distortion model, as described in the
[official RealSense projection documentation](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/).
Do not apply colour coefficients to the native depth grid or undistort an already
rectified stream a second time.

The [L515 user guide](https://www.realsenseai.com/download/13061/) describes
scan-time motion distortion and multipath corner rounding on pages 25–26, and
near-object saturation versus laser power/gain on page 8. These are different
from optical RGB distortion and are not corrected by remapping pixels.
Before changing settings or estimating any depth-bias correction, measure a
stationary matte planar reference at several distances and image positions,
using native depth calibration. No such physical calibration was performed here.
