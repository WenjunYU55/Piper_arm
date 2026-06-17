# Offline GroundingDINO Test

This folder contains an offline-only GroundingDINO test scaffold for saved L515 captures.

It reads `rgb.png` files from capture folders listed in:

```text
/home/prl/Piper_arm/AI_perception_tests/test_sets/real_l515_baseline/manifest.yaml
```

It does not:

- use ROS
- publish ROS topics
- publish `/piper/servo_cmd`
- move the PiPER arm
- modify `piper_ros_foxy`
- train a model
- add SAM2
- integrate GroundingDINO into ROS launch files

The purpose is to test whether pretrained GroundingDINO can detect the green cube and possible occluders in real saved L515 snapshots.

## Intended Backend

Use the official IDEA-Research/GroundingDINO repository:

```text
https://github.com/IDEA-Research/GroundingDINO
```

The scripts expect the official Python inference API:

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
```

## Expected Local Paths

By default, the scripts look for:

```text
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/GroundingDINO
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/weights/groundingdino_swint_ogc.pth
```

You can override those paths:

```bash
export GROUNDINGDINO_REPO_DIR=/path/to/GroundingDINO
export GROUNDINGDINO_CONFIG_PATH=/path/to/GroundingDINO_SwinT_OGC.py
export GROUNDINGDINO_CHECKPOINT_PATH=/path/to/groundingdino_swint_ogc.pth
export GROUNDINGDINO_DEVICE=cpu
```

Use `GROUNDINGDINO_DEVICE=cuda` only if your environment is already configured for CUDA.

## Check Environment

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/check_env.sh
```

This checks Python, torch, CUDA, `CUDA_HOME`, GroundingDINO imports, and model/config paths. It does not install anything.

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

## Manual Install Notes

If GroundingDINO is missing, install it manually from the official repository in a Python environment you control. Do not install it from these scripts.

Typical manual steps are:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
python3 -m pip install -e .
```

Then download the official pretrained checkpoint and set `GROUNDINGDINO_CHECKPOINT_PATH` to that `.pth` file.

Run `check_env.sh` again before running inference.
