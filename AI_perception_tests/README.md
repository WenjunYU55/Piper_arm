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
