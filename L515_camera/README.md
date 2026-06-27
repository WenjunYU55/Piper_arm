# L515 Camera Helpers

This folder contains helper files for testing the Intel RealSense L515 before target handoff is ready.

Files:

- `l515_camera_notes.md`: test workflow and topic notes.
- `realsense_l515_version_notes.md`: L515-specific SDK/firmware guidance from RealSense release notes.
- `fetch_realsense_sources.sh`: clones the selected librealsense and realsense-ros source tags.
- `install_realsense_build_deps.sh`: installs system build dependencies that need sudo.
- `build_realsense_ws.sh`: applies the local L515/Foxy patch and builds the RealSense source workspace.
- `source_l515_environment.sh`: sources ROS 2 Foxy plus the local RealSense and PiPER overlays.
- `check_l515_ros.sh`: checks whether RealSense ROS, this ROS package, and camera topics are visible.
- `check_l515_detection_connector.sh`: checks the camera input topics and the processed `/piper` connector topics for future base/arm integration.
- `start_l515_camera.sh`: starts the RealSense ROS camera node with aligned depth enabled.
- `run_l515_perception.sh`: starts the L515 perception-only pipeline after the RealSense camera driver is already running.
- `run_temporal_tracking_readonly.sh`: starts optional lightweight mask tracking with HSV as a one-shot seed by default.
- `run_heavy_refresh_bridge.sh`: snapshots heavy-refresh requests into filesystem jobs and publishes returned masks.
- `run_heavy_model_worker.sh`: runs GroundingDINO/SAM2 in the isolated Python 3.10 environment.
- `run_sam2_live_bridge.sh`: spools live RGB frames and publishes GPU SAM2 masks back into ROS.
- `run_sam2_live_worker.sh`: runs incremental SAM2.1 video propagation in the isolated CUDA environment.
- `run_gpu_vision_pipeline.sh`: starts the complete read-only CUDA vision pipeline.
- `run_gpu_geometry.sh`: converts the SAM2 target mask into 2D/3D tracking and occlusion inputs.
- `run_target_cloud.sh`: accumulates the masked L515 depth into a target point cloud.
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

Clouds are written under `datasets/target_clouds`. By default points remain in the camera frame. A
multi-view arm scan requires a valid camera-to-base TF and these settings:

```bash
PIPER_CLOUD_FRAME=piper_base_link PIPER_CLOUD_REQUIRE_TF=true \
  ./L515_camera/run_gpu_vision_pipeline.sh
```

Status and performance are published on `/piper/sam2_tracking_status`; cloud status is published on
`/piper/target_cloud_status`. This pipeline is read-only and does not publish real arm commands.

The ROS 2 package itself remains in:

```text
/home/prl/Piper_arm/piper_ros_foxy/src/piper_mobile_manipulation
```

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
./run_l515_perception.sh
./check_l515_detection_connector.sh
./view_l515_camera.sh
```

The perception pipeline now ends with a fake visual-servo command topic. It does not move the arm:

```bash
ros2 topic echo /piper/servo_cmd
```

`realsense_ws/src`, `build`, `install`, and `log` are generated locally and intentionally not committed. Run `fetch_realsense_sources.sh` and `build_realsense_ws.sh` to recreate them on a new machine.

For simple object recognition/tracking, set `target_color` in:

```text
/home/prl/Piper_arm/piper_ros_foxy/src/piper_mobile_manipulation/config/detection_params.yaml
```

Supported presets are `green`, `red`, `blue`, `yellow`, `orange`, `purple`, and `custom`. Use `custom` with `hsv_lower`/`hsv_upper` when the preset is not tight enough for the object.

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
