# Offline Grounded-SAM-2 / GroundingDINO Test

This folder contains an offline-only test scaffold for saved L515 captures.

It uses the official IDEA-Research Grounded-SAM-2 repository as the intended backend:

```text
https://github.com/IDEA-Research/Grounded-SAM-2
```

The current scripts run the GroundingDINO detector bundled inside Grounded-SAM-2 on saved `rgb.png` files, optionally refine detections with SAM2 masks, and write an offline manipulation-readiness report. They do not use ROS, publish ROS topics, publish `/piper/servo_cmd`, move the PiPER arm, modify `piper_ros_foxy`, train a model, or integrate anything into ROS launch files.

## Local Install Layout

The Grounded-SAM-2 checkout and Python environment are local helper dependencies:

```text
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/Grounded-SAM-2
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310
```

These are ignored by git so third-party code, environments, and model weights are not committed into your project repository.

By default, the scripts look for:

```text
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/Grounded-SAM-2/grounding_dino
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/weights/groundingdino_swint_ogc.pth
```

You can override those paths:

```bash
export GROUNDED_SAM2_PYTHON=/path/to/python
export GROUNDINGDINO_REPO_DIR=/path/to/Grounded-SAM-2/grounding_dino
export GROUNDINGDINO_CONFIG_PATH=/path/to/GroundingDINO_SwinT_OGC.py
export GROUNDINGDINO_CHECKPOINT_PATH=/path/to/groundingdino_swint_ogc.pth
export GROUNDINGDINO_DEVICE=cpu
```

Use `GROUNDINGDINO_DEVICE=cuda` only if your machine has a working CUDA PyTorch install.

Do not install GroundingDINO, SAM2, PyTorch, or Transformers into the ROS 2 Foxy workspace Python environment. Keep them in the isolated AI environment so version changes do not break the working ROS pipeline. The known fragile dependency is `transformers`; preserve the pinned version that works in your local AI environment.

The validated package versions, upstream revision, and `transformers==4.44.2` pin are recorded in `requirements_ai.txt`. The CPU setup command is documented in the repository-level `README.md`. GPU installs need a host-specific PyTorch/CUDA selection.

Run `./fetch_ai_assets.sh` to check out the tested Grounded-SAM-2 revision and download the GroundingDINO Swin-T and SAM2.1 Hiera Tiny checkpoints expected by these scripts.

## Check Environment

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/check_env.sh
```

This checks Python, torch, CUDA, `CUDA_HOME`, GroundingDINO imports, SAM2 imports, and model/config paths. It does not install anything.

## Run One Capture

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/run_one.sh \
  /home/prl/Piper_arm/L515_camera/captures/capture_YYYYMMDD_HHMMSS \
  "green cube . cube . box . hand . pen . tissue . paper tissue . paper . fruit . tool . wire . unknown object ."
```

Outputs are written to:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_name>/groundingdino/
```

Files:

- `groundingdino_boxes.yaml`
- `groundingdino_debug.png`

`groundingdino_boxes.yaml` includes raw detections plus:

- `summary.best_target_detection`
- `summary.obstacle_candidates`
- `summary.unsafe_candidates`
- `summary.candidate_safe_class_detections`

## Run Batch Test

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/batch_run.sh
```

Default prompt:

```text
green cube . cube . box . hand . pen . tissue . paper tissue . paper . fruit . tool . wire . unknown object .
```

Keep blocker prompts object-specific. Generic terms such as `occluder` and `blocker`
produced false positives in clear scenes; use mask overlap and depth ordering for the
geometric occlusion decision instead.

Each capture uses two GroundingDINO passes:

1. The full frame finds the target. A valid tracked-mask component is used only when that pass misses the target.
2. A bounded crop around the selected target runs the obstacle-only prompt at a separate threshold.

The obstacle prompt uses `|`-separated groups so visually competing classes are evaluated independently:

```text
pen . | hand . finger . | wire . cable . | tissue . paper tissue . paper .
```

Generic `tool` and `unknown object` are not crop prompts. A geometrically detected region that does not
match a specific group remains unknown and blocked. Crop detections are remapped into full-image
coordinates and duplicate boxes are suppressed across groups.
SAM2 then rejects masks that are not target-local, lack depth support, or duplicate the target mask.
Full-frame non-target detections remain diagnostic and do not become manipulation obstacles.

Batch summary is written to:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/groundingdino_results.yaml
```

Per-capture outputs are written to:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_name>/groundingdino/
```

The batch result does not modify `manifest.yaml`. The manifest remains the human-labelled ground truth file.

## Run SAM2 Refinement

After GroundingDINO has produced boxes, refine one capture:

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  /home/prl/Piper_arm/AI_perception_tests/groundingdino_test/sam2_refine_on_capture.py \
  /home/prl/Piper_arm/L515_camera/captures/capture_YYYYMMDD_HHMMSS
```

Outputs are written to:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_name>/sam2/
```

Files:

- `sam2_masks.yaml`
- `sam2_overlay.png`
- `mask_*.png`

If SAM2 is not importable or the checkpoint is missing, the script writes `status: sam2_unavailable` instead of crashing the whole workflow.

SAM2 refinement is target-centric:

- GroundingDINO supplies the preferred target box.
- A valid saved tracked-target component can supply a fallback box when GroundingDINO misses a partly visible target.
- GroundingDINO obstacle boxes are retained only when they are near the target box.
- A closer-depth mask is generated only in a small region around a trusted target mask.
- Depth-only obstacle masks are classified as unknown and unsafe; they cannot authorize manipulation.
- When a target-local semantic mask explains at least half of the same depth region, the depth evidence
  inherits that semantic class instead of being treated as a second unknown obstacle.

Run the full manifest:

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  /home/prl/Piper_arm/AI_perception_tests/groundingdino_test/batch_sam2_refine.py
```

Batch summary is written to:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/sam2_results.yaml
```

## Create Readiness Summary

Combine the labelled manifest, GroundingDINO results, and SAM2 results:

```bash
python3 /home/prl/Piper_arm/AI_perception_tests/groundingdino_test/summarize_ai_results.py
```

Output:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/ai_readiness_summary.yaml
```

The readiness report is conservative:

- `safe_to_consider_manipulation` is always `false` in this offline stage.
- manifest occluder labels are evaluation ground truth and are not treated as runtime model evidence.
- candidate-movable classes such as pens, markers, paper, and tissue are advisory labels only.
- humans/body parts, wires, cables, generic tools, and unknown occluders are blocked.
- candidate-movable model labels require at least `0.45` confidence; weaker labels remain unknown and blocked.
- a depth-confirmed candidate may use a `0.40` semantic threshold when at least half of the same obstacle
  region is independently supported by closer depth.
- missing target masks, poor depth, or missing model outputs are blocked.

## Notes

This is not a training database. It is a small validation/debug workflow for checking pretrained models on real L515 RGB snapshots.

SAM2 is optional in this scaffold. Mask generation must pass offline validation before any ROS integration.
