# Intel RealSense L515 camera bringup for PiPER arm-side perception

This file is for the camera-only development stage. It does not use target handoff and it does not move the PiPER arm.

## Goal

Run the L515, detect a colored test object in RGB, convert the 2D detection to a 3D point using depth, then track the filtered target position and velocity.

Pipeline:

```text
L515 RGB image
  -> l515_object_detector_node
  -> /piper/detection_2d

L515 depth image + CameraInfo + /piper/detection_2d
  -> depth_to_3d_node
  -> /piper/target_3d

/piper/target_3d
  -> target_tracker_node
  -> /piper/tracked_target
```

## Recommended test environment

- Indoor, shaded, or greenhouse lighting.
- Keep the target roughly 0.3 m to 1.0 m from the L515.
- Use a strong colored test object first, such as a green, red, or blue card/ball.
- Avoid strong sunlight for the first tests.

## Check whether RealSense ROS is installed

From `/home/prl/Piper_arm/L515_camera`:

```bash
./check_l515_ros.sh
```

If `realsense2_camera` is missing, install or build Intel RealSense ROS for ROS 2 Foxy before running the camera launch.

See `realsense_l515_version_notes.md` before installing. For L515, the RealSense release page still points to SDK `v2.50.0` as validated and `v2.54.2` as supported but not validated. The best first source-build candidate is currently `librealsense v2.50.0` plus `realsense-ros 4.0.4`; use `realsense-ros 3.2.3` as a fallback.

## Start the L515

Typical RealSense ROS 2 command:

```bash
source /opt/ros/foxy/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  device_type:=l515 \
  enable_color:=true \
  enable_depth:=true \
  enable_confidence:=false \
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=640x480x30 \
  align_depth.enable:=true
```

Or use the helper:

```bash
cd /home/prl/Piper_arm/L515_camera
./start_l515_camera.sh
```

Then inspect topics:

```bash
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo --once /camera/color/camera_info
```

Topic names vary by RealSense launch settings. If your topics are different, update:

- `/home/prl/Piper_arm/piper_ros_foxy/src/piper_mobile_manipulation/config/detection_params.yaml`
- `/home/prl/Piper_arm/piper_ros_foxy/src/piper_mobile_manipulation/config/camera_params.yaml`

Use aligned depth for this first detector. The detector pixel `(u, v)` comes from the RGB image, so the depth image should be `/camera/aligned_depth_to_color/image_raw` and intrinsics should come from `/camera/color/camera_info`.

With the helper launch, the expected camera optical frame is `camera_color_optical_frame`. If you change `camera_name`, update `config/frames.yaml` and the camera topics together.

If your RealSense wrapper rejects `align_depth.enable`, you likely have an older ROS2 legacy wrapper. Check `realsense_l515_version_notes.md` and adapt the launch arguments to the wrapper version installed on the robot.

## Run arm-side perception only

```bash
cd /home/prl/Piper_arm/piper_ros_foxy
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch piper_mobile_manipulation perception_only.launch.py
```

Or use the helper from this folder:

```bash
cd /home/prl/Piper_arm/L515_camera
./run_l515_perception.sh
```

Watch outputs:

```bash
ros2 topic echo /piper/detection_2d
ros2 topic echo /piper/target_3d
ros2 topic echo /piper/tracked_target
```

## HSV tuning

The first detector uses HSV thresholding. Default config is green:

```yaml
hsv_lower: [30, 50, 50]
hsv_upper: [90, 255, 255]
```

Use a simple colored target first. Later this detector can be replaced with YOLO, segmentation, or a plant/object-specific detector.

## Safety

This camera stage does not command the real arm. Use it to prove:

- RGB frames arrive.
- Depth frames arrive.
- CameraInfo arrives.
- 2D detection is stable.
- Median depth is valid.
- 3D target points stay in the expected camera frame.
- Tracker output is smooth and does not chase raw depth noise.
