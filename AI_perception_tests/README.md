# AI Perception Tests

This directory contains the offline and worker-side validation code for the production
GroundingDINO and multi-object SAM2 perception pipeline. None of these tools command the arm.

## Production architecture

The active pipeline uses:

- GroundingDINO for initialization and event-driven semantic refresh.
- SAM2 image segmentation for full-resolution target and obstacle masks.
- SAM2 video propagation for continuous multi-object tracking.
- ROS geometry nodes for 2D geometry, RGB-D projection, 3D tracking, scan quality, occlusion,
  and target-cloud accumulation.

The former HSV detector and CPU optical-flow mask tracker have been removed. A saved or live
tracked target mask may still provide a generic fallback box when GroundingDINO misses the target;
that fallback does not perform HSV segmentation.

## CUDA environment

Prepare the isolated environment once:

```bash
cd /home/prl/Piper_arm
./AI_perception_tests/groundingdino_test/setup_gpu_env.sh
```

The model workers use this environment so ROS 2 Foxy remains isolated from the Python 3.10 AI
dependencies.

## Worker tests

Run the ROS-free heavy-worker tests with the system Python:

```bash
cd /home/prl/Piper_arm/AI_perception_tests
python3 -m unittest -v test_heavy_model_worker.py
```

Run SAM2 tests with the isolated environment:

```bash
cd /home/prl/Piper_arm/AI_perception_tests
groundingdino_test/envs/grounded_sam2_py310/bin/python test_sam2_live_worker.py
```

The SAM2 tests cover multi-object propagation, empty-target recovery, semantic reseeding, and
reduced-resolution inference with native-resolution mask output.

## Offline captures

Capture directories retain this compatibility layout:

```text
rgb.png
depth.npy
detection_mask.png
camera_info.yaml
target_3d.yaml
metadata.yaml
```

`detection_mask.png` is a historical file-format name. New live captures source it from
`/piper/sam2_target_mask`.

Run static analysis over one capture:

```bash
python3 /home/prl/Piper_arm/AI_perception_tests/static_scene_analyzer.py /path/to/capture
```

GroundingDINO/SAM2-specific setup and offline commands are documented in
`groundingdino_test/README.md`.

## Live system

Start the complete read-only pipeline from the repository root:

```bash
./L515_camera/run_gpu_vision_pipeline.sh
```

Primary outputs include:

```text
/piper/sam2_target_mask
/piper/sam2_obstacle_mask
/piper/sam2_object_ids
/piper/sam2_tracking_status
/piper/target_3d
/piper/target_cloud
```
