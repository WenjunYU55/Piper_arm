# L515 Camera Helpers

This folder contains helper files for testing the Intel RealSense L515 before target handoff is ready.

Files:

- `realsense_l515_version_notes.md`: L515-specific SDK/firmware guidance from RealSense release notes.
- `fetch_realsense_sources.sh`: clones the selected librealsense and realsense-ros source tags.
- `install_realsense_build_deps.sh`: installs system build dependencies that need sudo.
- `build_realsense_ws.sh`: applies the local L515/Foxy patch and builds the RealSense source workspace.
- `source_l515_environment.sh`: sources ROS 2 Foxy plus the local RealSense and PiPER overlays.
- `check_l515_ros.sh`: checks whether RealSense ROS, this ROS package, and camera topics are visible.
- `start_l515_camera.sh`: starts the RealSense ROS camera node with aligned depth enabled.
- `run_heavy_refresh_bridge.sh`: snapshots heavy-refresh requests into filesystem jobs and publishes returned masks.
- `run_heavy_model_worker.sh`: runs GroundingDINO/SAM2 in the isolated Python 3.10 environment.
- `run_sam2_live_bridge.sh`: spools live RGB frames and publishes GPU SAM2 masks back into ROS.
- `run_sam2_live_worker.sh`: runs incremental SAM2.1 video propagation in the isolated CUDA environment.
- `run_gpu_vision_pipeline.sh`: starts the complete read-only CUDA vision pipeline.
- `run_gpu_geometry.sh`: converts the SAM2 target mask into 2D/3D tracking and occlusion inputs.
- `run_target_cloud.sh`: accumulates the masked L515 depth into a target point cloud.
- `capture_hand_eye_sample.py`: captures a strict full-board ChArUco hand-eye sample.
- `solve_hand_eye.py`: solves and independently validates PiPER eye-in-hand calibration.
- `run_hand_eye_tf.sh`: publishes the accepted dynamic `base_link` to camera TF.
- `run_fixed_board_validation.sh`: interactively checks fixed-board repeatability across arm poses.
- `view_l515_camera.sh`: opens a simple image viewer for a camera topic.
- `view_l515_showimage.sh`: opens ROS 2 `image_tools/showimage` for a camera topic.
- `view_l515_opencv.sh`: opens a direct OpenCV viewer for a camera topic.
- `view_l515_rviz.sh`: opens RViz with color, aligned depth, and detection debug image displays.

Heavy-refresh mask topics:

```text
/piper/heavy_target_mask
/piper/heavy_obstacle_mask
/piper/candidate_movable_obstacle_mask
/piper/unsafe_obstacle_mask
/piper/sam2_target_mask
```

## Complete GPU vision pipeline

Prepare the isolated CUDA environment once:

```bash
./AI_perception_tests/groundingdino_test/setup_gpu_env.sh
```

Start the complete system with one command:

```bash
./L515_camera/run_gpu_vision_pipeline.sh
```

GroundingDINO detects the requested target and obstacles only at initialization and event-triggered
semantic refreshes. SAM2 creates their masks and then tracks every labelled object continuously. Both
models are required to run on CUDA; the workers fail instead of silently falling back to CPU. The
rolling SAM2 state resets every eight frames using the latest masks, bounding GPU memory on the
validated RTX 3090. Live SAM2 inference defaults to 384 pixels wide and its masks are restored to the
native 640x480 RGB-D resolution with nearest-neighbour resizing. Override this with
`PIPER_SAM2_INFERENCE_WIDTH`; use `640` for native-resolution live inference.

View the output:

```bash
./L515_camera/view_l515_opencv.sh /piper/sam2_target_mask
./L515_camera/view_l515_opencv.sh /piper/sam2_obstacle_mask
./L515_camera/view_l515_opencv.sh /piper/sam2_object_ids
```

Request one full-resolution GroundingDINO/SAM2 cloud capture while the camera is stationary, then
save the accumulated L515 target cloud:

```bash
export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: capture}"
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: save}"
```

The delayed full-resolution mask is matched to its original cached RGB-D frame and eroded by one
pixel before projection, reducing background leakage. Live upscaled masks still accumulate by
default. For a refinement-only high-quality cloud, start with
`PIPER_CLOUD_ACCUMULATE_LIVE=false` and issue `capture` once at each stationary viewpoint.

Clouds are written under `datasets/target_clouds`. Points are fused in `base_link` by default and a
timestamped camera-to-base transform is mandatory. The explicit production command is:

```bash
export ROS_LOCALHOST_ONLY=0 L515_ROS_LOCALHOST_ONLY=0
PIPER_CLOUD_FRAME=base_link PIPER_CLOUD_REQUIRE_TF=true \
  ./L515_camera/run_gpu_vision_pipeline.sh
```

Status and performance are published on `/piper/sam2_tracking_status`; cloud status is published on
`/piper/target_cloud_status`. This pipeline is read-only and does not publish real arm commands.

## Eye-in-hand calibration

The deployed calibration is:

```text
calibration/hand_eye/session_20260701_local/calibration_result.yaml
```

It is accepted from 12 fitting and 3 held-out validation samples. Do not use
`session_20260629_resample3`, which was rejected. Reproduce the solve from the compact committed
sample metadata:

```bash
python3 L515_camera/solve_hand_eye.py \
  L515_camera/calibration/hand_eye/session_20260701_local
```

Start the runtime TF publisher after the PiPER driver and camera:

```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
./L515_camera/run_hand_eye_tf.sh
```

The publisher refuses any calibration whose status is not `accepted`. It reads
`/joint_states_single`, computes PiPER modified-DH mode-0 FK, and publishes dynamic
`base_link -> camera_link`; RealSense supplies the remaining static optical-frame transform.

Validate the physical chain with the fixed ChArUco board left stationary:

```bash
./L515_camera/run_fixed_board_validation.sh
```

Stop the arm at each substantially different viewpoint, press Enter to average ten strict full-board
detections, collect at least three poses, then enter `q`. The accepted physical test used five poses
and measured maximum drift of 8.63 mm and 0.59 degrees against limits of 15 mm and 1.5 degrees.
Neither the TF publisher nor validator commands arm motion.

The ROS 2 package itself remains in:

```text
/home/prl/Piper_arm/piper_ros_foxy/src/piper_mobile_manipulation
```

The GPU geometry launch publishes one atomic obstacle snapshot per SAM2 frame on
`/piper/obstacle_instances_3d`. Each record contains camera and `base_link` geometry;
`scene_blocked` remains true for invalid, unsafe, unknown, or non-whitelisted objects.
Writing tools use one canonical `pen` class. Marker label variants normalize to
`pen`; it is the only whitelisted class.

The geometry launch also maintains a stationary target landmark in `base_link`:

- `/piper/target_landmark` is the conservative world-frame reference point.
- `/piper/target_landmark_projection` is its predicted pixel in the current view.
- `/piper/target_landmark_status` reports agreement and whether a rescan is needed.

Meaningful viewpoint changes request a fresh full target mask. Target cloud points
are fused in `base_link`; a timestamped transform is required, so no camera-frame
fallback is used for multi-view accumulation.

To validate a fixed marker after starting the normal perception and hand-eye TF terminals:

```bash
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh
ros2 run piper_mobile_manipulation obstacle_repeatability_validator.py \
  --ros-args -p scenario:=clear_view -p expected_label:=pen
```

At each stationary GUI-positioned viewpoint, capture one sample. Collect 5–8, then finalize:

```bash
ros2 service call /obstacle_repeatability_validator/capture_sample std_srvs/srv/Trigger '{}'
ros2 service call /obstacle_repeatability_validator/finalize std_srvs/srv/Trigger '{}'
```

Reports are written under `/tmp/piper_obstacle_validation` by default. For unsafe-object and
unknown-object scenarios, also pass `-p expect_scene_blocked:=true` and use a distinct scenario
name. Neither node publishes any motion command.

Run:

```bash
cd /home/prl/Piper_arm/L515_camera
./check_l515_ros.sh
```

Fetch source code from GitHub:

```bash
./fetch_realsense_sources.sh
```

Default source pair:

```text
librealsense v2.50.0
realsense-ros 4.0.4
```

Then, in separate terminals:

```bash
./start_l515_camera.sh
./view_l515_camera.sh
```

`realsense_ws/src`, `build`, `install`, and `log` are generated locally and intentionally not committed. Run `fetch_realsense_sources.sh` and `build_realsense_ws.sh` to recreate them on a new machine.

If `rqt_image_view` prints DDS deserialization errors, use the lighter viewers:

```bash
./view_l515_showimage.sh /camera/color/image_raw
./view_l515_opencv.sh /camera/color/image_raw
```

To source the same environment manually:

```bash
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh
```

For RViz:

```bash
./view_l515_rviz.sh
```
