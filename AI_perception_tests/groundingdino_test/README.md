# Offline Grounded-SAM-2 / GroundingDINO Test

This folder contains an offline-only test scaffold for saved L515 captures.

It uses the official IDEA-Research Grounded-SAM-2 repository as the intended backend:

```text
https://github.com/IDEA-Research/Grounded-SAM-2
```

The current scripts run the GroundingDINO detector bundled inside Grounded-SAM-2 on saved `rgb.png` files. They do not use ROS, publish ROS topics, publish `/piper/servo_cmd`, move the PiPER arm, modify `piper_ros_foxy`, train a model, or integrate anything into ROS launch files.

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

## Check Environment

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/check_env.sh
```

This checks Python, torch, CUDA, `CUDA_HOME`, GroundingDINO imports, SAM2 imports, and model/config paths. It does not install anything.

## Run One Capture

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/run_one.sh \
  /home/prl/Piper_arm/L515_camera/captures/capture_YYYYMMDD_HHMMSS \
  "green cube . cube . box . hand . leaf . branch . stem . tool . wire . unknown object ."
```

Outputs are written to:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_name>/groundingdino/
```

Files:

- `groundingdino_boxes.yaml`
- `groundingdino_debug.png`

## Run Batch Test

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/batch_run.sh
```

Default prompt:

```text
green cube . cube . box . hand . leaf . branch . stem . fruit . tool . wire . unknown object .
```

Batch summary is written to:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/groundingdino_results.yaml
```

Per-capture outputs are written to:

```text
/home/prl/Piper_arm/AI_perception_tests/outputs/<capture_name>/groundingdino/
```

## Notes

This is not a training database. It is a small validation/debug workflow for checking pretrained models on real L515 RGB snapshots.

SAM2 is installed as part of the Grounded-SAM-2 environment, but this scaffold currently runs the bundled GroundingDINO detector only. Mask generation from SAM2 should be added as a separate offline step before any ROS integration.
