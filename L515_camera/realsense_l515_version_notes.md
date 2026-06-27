# L515 RealSense Version Notes

Checked against the RealSense SDK release page on 2026-05-28.

## Practical recommendation for this robot

For the L515 on ROS 2 Foxy, prefer a known L515-compatible setup instead of blindly installing the newest SDK:

- L515 firmware should be `1.5.8.1` or later.
- SDK `v2.50.0` is the last release called out as validated for L515.
- SDK `v2.54.2` is called out as supporting L515 but not validated.
- The latest release page checked was `v2.58.1`; it still points back to `v2.50.0` as the validated L515 SDK and `v2.54.2` as supported-but-not-validated.
- ROS 2 Foxy is EOL, and the current RealSense ROS wrapper docs say Debian package install is not supported for Foxy. Use a source build or an already-working local wrapper install.

## Best source-build candidates

Primary candidate:

```text
librealsense:  v2.50.0
realsense-ros: 4.0.4
```

Why: `realsense-ros` `4.0.4` explicitly supports LibRealSense `v2.50.0`, ROS 2 Foxy, and L515. It is newer than `3.2.3` and its release notice says it fixed required packages that caused Debian build errors.

Fallback candidate:

```text
librealsense:  v2.50.0
realsense-ros: 3.2.3
```

Why: `3.2.3` also explicitly supports LibRealSense `v2.50.0`, Foxy, and L515/L535. Use this if `4.0.4` has build or launch incompatibilities on the robot.

Experimental candidate:

```text
librealsense:  v2.51.1
realsense-ros: 4.51.1
```

Why: `4.51.1` still lists Foxy and L515 and includes several align-depth fixes, including align-depth enable/disable and a crash fix when IMU and aligned depth are active. This is useful if aligned depth is broken on the older pair, but it moves away from the L515-validated `v2.50.0` baseline.

## What this changes in our setup

The perception stack now assumes:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/aligned_depth_to_color/image_raw
camera_color_optical_frame
```

Aligned depth matters because SAM2 masks use color-image pixels. Projecting those pixels with unaligned depth can produce wrong 3D target points.

The camera launch keeps the RealSense wrapper's default `camera_name:=camera` so the topics stay under `/camera`. It also passes `device_type:=l515` to avoid accidentally binding to another RealSense camera on the same host.

## Suggested bringup order

1. Check the environment:

```bash
cd /home/prl/Piper_arm/L515_camera
./check_l515_ros.sh
```

2. Start the camera in terminal 1:

```bash
cd /home/prl/Piper_arm/L515_camera
./start_l515_camera.sh
```

3. Start perception in terminal 2:

```bash
cd /home/prl/Piper_arm/L515_camera
./run_gpu_vision_pipeline.sh
```

4. Watch outputs:

```bash
ros2 topic echo /piper/detection_2d
ros2 topic echo /piper/target_3d
ros2 topic echo /piper/tracked_target
```

## ROS wrapper parameter compatibility

Newer `realsense-ros` wrappers use:

```bash
align_depth.enable:=true
depth_module.profile:=640x480x30
```

Older ROS2 legacy wrappers may use older parameter names. If `start_l515_camera.sh` fails because `align_depth.enable` is unknown, inspect the wrapper help/log output and try the legacy launch arguments used by your installed wrapper.

## Install/build note

Do not install multiple librealsense versions at the same time. The RealSense ROS wrapper documentation explicitly warns against mixing install methods because it can create workspace conflicts.

On this machine, the source build needs `libusb-1.0-0-dev`. Without it, librealsense falls back to its bundled libusb build, which can fail with:

```text
fatal error: config.h: No such file or directory
```

Install build dependencies from a terminal where you can enter the sudo password:

```bash
cd /home/prl/Piper_arm/L515_camera
./install_realsense_build_deps.sh
```

Then build:

```bash
./build_realsense_ws.sh
```
