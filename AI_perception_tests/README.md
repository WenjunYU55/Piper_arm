# AI Perception Tests

This folder is for offline perception experiments only. It is separate from the main ROS active scan pipeline and is intended for testing saved RGB-D captures before integrating AI perception models into ROS.

These tools:

- read saved RGB-D capture folders from disk
- check whether expected capture files are present
- organise a small real L515 validation/debug set
- write offline analysis outputs under `outputs/`
- prepare the project for future GroundingDINO, SAM2, and VLM testing

These tools do not:

- control the PiPER arm
- publish ROS topics
- command `/piper/servo_cmd`
- perform real arm movement
- perform real pushing
- require PyTorch, SAM2, GroundingDINO, YOLO, or VLM dependencies

## Purpose

This is not a training database. GroundingDINO and SAM2 are pretrained models and do not need to be trained here.

The `test_sets/real_l515_baseline` workflow is a small validation/debug set for roughly 10 to 30 real L515 captures. Use it to check the current static analyser and, later, to test pretrained GroundingDINO and SAM2 on real PiPER scene images before any ROS integration.

## Expected Capture Files

Each capture folder is expected to contain:

- `rgb.png`
- `depth.npy`
- `detection_mask.png`
- `camera_info.yaml`
- `target_3d.yaml`
- `metadata.yaml`

Optional files:

- `scan_quality.yaml`
- `occlusion_status.yaml`

## Basic Setup

No dependencies are installed automatically. If needed, install the minimal dependencies manually:

```bash
python3 -m pip install -r /home/prl/Piper_arm/AI_perception_tests/requirements_basic.txt
```

## One-Capture Static Analysis

```bash
/home/prl/Piper_arm/AI_perception_tests/run_static_analysis.sh /path/to/capture_folder
```

Example:

```bash
/home/prl/Piper_arm/AI_perception_tests/run_static_analysis.sh /home/prl/Piper_arm/L515_camera/captures/capture_YYYYMMDD_HHMMSS
```

The analyser writes `analysis.yaml` to:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_folder_name>/
```

## Real L515 Baseline Workflow

1. Capture real L515 snapshots:

```bash
/home/prl/Piper_arm/L515_camera/capture_snapshot.sh
```

2. Create or update the manifest:

```bash
/home/prl/Piper_arm/AI_perception_tests/create_manifest.sh
```

This scans:

```text
/home/prl/Piper_arm/L515_camera/captures
```

and writes:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/manifest.yaml
```

Capture folders are symlinked into category folders where possible:

```text
test_sets/real_l515_baseline/clear_cube
test_sets/real_l515_baseline/partial_occlusion
test_sets/real_l515_baseline/heavy_occlusion
test_sets/real_l515_baseline/hand_blocker
test_sets/real_l515_baseline/edge_cases
test_sets/real_l515_baseline/lost_target
test_sets/real_l515_baseline/unknown
```

3. Optionally edit `manifest.yaml` manually to set:

- `category`
- `expected_state`
- `target`
- `occluder`
- `notes`

Default labels are intentionally conservative:

```yaml
category: unknown
expected_state: unknown
target: green cube
occluder: unknown
```

4. Run batch static analysis:

```bash
/home/prl/Piper_arm/AI_perception_tests/batch_static_analysis.sh
```

This writes per-capture outputs under:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_name>/
```

and updates `manifest.yaml` with the static analyser decision and key metrics.

5. Generate the summary report:

```bash
/home/prl/Piper_arm/AI_perception_tests/summarize_test_set.sh
```

This prints a summary and writes:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/summary.yaml
```

6. Later, use this same small test set for offline GroundingDINO and SAM2 validation. Do that offline first; do not add those models to ROS launch files until they have been validated on saved captures.

## Offline Temporal Tracking

After a heavy detector and SAM2 provide the first target mask, run lightweight tracking over a saved
active-scan sequence:

```bash
python3 /home/prl/Piper_arm/AI_perception_tests/run_temporal_tracking.py \
  /home/prl/Piper_arm/datasets/active_scan/scan_YYYYMMDD_HHMMSS
```

The runner accepts the existing `frames/view_NNN_rgb.png`, depth, mask, and metadata layout. By default,
only the first saved mask initializes tracking. Later masks are evaluation references and are not fed back
into the tracker. Outputs are written under:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/temporal_tracking/<scan_name>/
```

The lightweight tracker uses forward/backward pyramidal Lucas-Kanade optical flow, RANSAC affine mask
propagation, mask-area gates, aligned depth validity, and persistent local foreground checks. It requests a
heavy refresh when confidence drops, tracking fails, the scene changes, or the refresh interval expires.
It does not import or run GroundingDINO or SAM2 itself.

To test refresh state transitions with already saved masks:

```bash
python3 /home/prl/Piper_arm/AI_perception_tests/run_temporal_tracking.py \
  /path/to/scan --simulate-saved-mask-refresh
```

This is simulation only; it is not model inference. Reports and overlays remain offline, dry-run, and unable
to command robot motion.

To execute event-driven GroundingDINO/SAM2 initialization and refresh using the isolated AI environment:

```bash
HF_HOME=/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/hf_cache \
MPLCONFIGDIR=/tmp/piper_grounded_sam2_mpl \
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  /home/prl/Piper_arm/AI_perception_tests/run_temporal_tracking.py \
  /path/to/scan --execute-heavy-refresh --heavy-device cpu
```

This runs heavy perception on frame 0 and only when the lightweight policy requests a refresh. Heavy event
inputs, GroundingDINO boxes, SAM2 masks, labels, confidence, and trigger reasons are stored alongside the
temporal report. `--execute-heavy-refresh` remains offline and cannot publish ROS or motion commands.

Run the synthetic unit tests:

```bash
cd /home/prl/Piper_arm/AI_perception_tests
python3 -m unittest -v test_temporal_tracking.py
```

Validated offline behavior:

- `scan_20260616_122317`: 29/29 post-initialization frames remained `TRACKING` with no refresh request or lost frame.
- `scan_20260612_143052`: tracking failure requested refresh and expired to `LOST` when no refresh was supplied.
- the same recovery scan with simulated saved-mask refresh returned to tracking after one refresh with no lost frame.
- `scan_20260623_111328`: tracked all 40 frames; detected a marker entering at frame 6, confirmed
  persistence and requested one heavy refresh at frame 8, then cleared the obstacle state at frame 24.
- event-driven heavy validation on that scan initialized the target at frame 0, classified the frame-8
  blocker as `whiteboard marker`, associated its semantic mask with closer-depth evidence, refreshed the
  target mask once, and completed with no lost frames or unsafe obstacle classification.

The purpose-built `scan_20260623_111328` sequence validates real-camera obstacle persistence and clearing;
the earlier scans validate stable propagation and recovery state transitions.

## Isolated Live Heavy-Refresh Worker

GroundingDINO and SAM2 remain outside ROS Foxy. A lightweight Foxy bridge snapshots the latest RGB,
aligned depth, and tracked mask into an atomic filesystem job whenever
`/piper/heavy_refresh_request` is received. A separate Python 3.10 process runs the heavy models and
returns an atomic mask response. The bridge then publishes that mask on `/piper/heavy_target_mask`.

Start the bridge and worker in separate terminals after the camera, then start the tracker with its heavy
mask input (the order between bridge and worker is unimportant):

```bash
/home/prl/Piper_arm/L515_camera/run_heavy_refresh_bridge.sh
/home/prl/Piper_arm/L515_camera/run_heavy_model_worker.sh
/home/prl/Piper_arm/L515_camera/run_temporal_tracking_readonly.sh /piper/heavy_target_mask
```

Both sides use `/tmp/piper_heavy_refresh` by default. Override it consistently with
`PIPER_HEAVY_REFRESH_SPOOL`. The worker is CPU-only, imports no ROS modules, and publishes no topics or
motion commands. Status is available on `/piper/heavy_refresh_status`.

Heavy refresh responses also publish categorized obstacle masks:

```text
/piper/heavy_obstacle_mask
/piper/candidate_movable_obstacle_mask
/piper/unsafe_obstacle_mask
```

Candidate-movable masks are limited to conservatively approved classes such as markers, pens, paper,
and tissue. Hands, people, fingers, wires, cables, unknown objects, and generic tools remain unsafe
blockers and must never become manipulation targets.

## SAM2 Video Benchmark

Benchmark pretrained SAM2.1 Hiera Tiny video propagation without ROS or motion commands:

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  /home/prl/Piper_arm/AI_perception_tests/benchmark_sam2_video.py \
  /home/prl/Piper_arm/datasets/active_scan/scan_20260623_111328
```

Validated CPU result on the 40-frame scan:

- mean reference-mask IoU: 0.917
- minimum reference-mask IoU: 0.833
- propagation rate: 0.113 FPS (353.5 seconds for 40 frames)
- peak resident memory: 2652 MB

The mask quality supports SAM2 video as the planned Jetson tracker, including multi-object target and
obstacle propagation. CPU throughput is not suitable for the live camera loop, so the lightweight
adaptive appearance/depth tracker remains the current live path.

Run the boundary test without loading either heavy model:

```bash
cd /home/prl/Piper_arm/AI_perception_tests
python3 -m unittest -v test_heavy_model_worker.py
```

## Optional Read-Only ROS Tracker

The validated tracking core is shared with an optional ROS node. It is not included in existing system
launch files and publishes no motion commands. With the camera and perception pipeline running, start it
with an HSV one-shot seed for live testing:

```bash
/home/prl/Piper_arm/L515_camera/run_temporal_tracking_readonly.sh
```

Outputs:

```text
/piper/temporal_target_mask
/piper/temporal_tracking_debug_image
/piper/temporal_tracking_status
/piper/heavy_refresh_request
```

The node accepts a seed mask only during initialization or after it requests a heavy refresh. Continuous
HSV masks therefore do not reinitialize tracking every frame. The default production-facing seed topic is
`/piper/heavy_target_mask`; the helper overrides it with `/piper/detection_mask` only for fallback testing.
