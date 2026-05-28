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
- `view_l515_camera.sh`: opens a simple image viewer for a camera topic.
- `view_l515_showimage.sh`: opens ROS 2 `image_tools/showimage` for a camera topic.
- `view_l515_opencv.sh`: opens a direct OpenCV viewer for a camera topic.
- `view_l515_rviz.sh`: opens RViz with color, aligned depth, and detection debug image displays.

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
