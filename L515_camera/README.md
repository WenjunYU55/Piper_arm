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
```

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
