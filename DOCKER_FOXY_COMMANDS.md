# Docker Foxy Commands

This project is built for Ubuntu 20.04 and ROS 2 Foxy. The host machine is Ubuntu 22.04, so run ROS
workflows inside the local Docker image instead of installing Foxy directly on the host.

Host project path:

```bash
/home/wenjun/prl/Piper_arm
```

Docker image:

```bash
piper-arm-foxy:local
```

For the full operator reference, including what each script does, also read:

```bash
/home/wenjun/prl/Piper_arm/OPERATOR_COMMANDS.md
```

## Start A Foxy Terminal

Run this on the host for every terminal that needs ROS/Foxy. This version also enables GUI viewers and
shares the heavy-model spool directory with the host AI worker.

```bash
mkdir -p /tmp/piper_heavy_refresh
xhost +local:root

cd /home/wenjun/prl/Piper_arm
sudo docker run --rm -it \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/piper_heavy_refresh:/tmp/piper_heavy_refresh \
  -v "$PWD":/workspace/Piper_arm \
  -w /workspace/Piper_arm \
  piper-arm-foxy:local \
  bash
```

Inside Docker, source the Foxy and project overlays:

```bash
source L515_camera/source_l515_environment.sh
```

## Verify The Install

Inside Docker:

```bash
./verify_installation.sh
```

Expected result:

```text
Installation verification passed. Hardware connectivity is not tested.
```

## Rebuild The Docker Image

Run this on the host after editing `Dockerfile.foxy`:

```bash
cd /home/wenjun/prl/Piper_arm
sudo docker build -f Dockerfile.foxy -t piper-arm-foxy:local .
```

## Rebuild ROS Workspaces

Inside Docker:

```bash
source /opt/ros/foxy/setup.bash
cd /workspace/Piper_arm/piper_ros_foxy
colcon build --symlink-install
cd /workspace/Piper_arm
```

Build the RealSense/L515 workspace:

```bash
cd /workspace/Piper_arm
./L515_camera/build_realsense_ws.sh
```

## Read-Only Camera And Perception Runtime

Use separate host terminals. Start Docker in each ROS terminal with the command from "Start A Foxy
Terminal", then run `source L515_camera/source_l515_environment.sh` inside Docker.

Terminal 1, Docker: start the L515 camera.

```bash
./L515_camera/start_l515_camera.sh
```

Terminal 2, Docker: run the heavy-refresh bridge.

```bash
./L515_camera/run_heavy_refresh_bridge.sh
```

Terminal 3, host only: run the AI model worker. Do not run this one inside Docker; it uses the host
Python 3.10 Grounded-SAM2 environment and has no ROS imports.

```bash
cd /home/wenjun/prl/Piper_arm
./L515_camera/run_heavy_model_worker.sh
```

Terminal 4, Docker: run temporal tracking after the heavy target mask is available.

```bash
./L515_camera/run_temporal_tracking_readonly.sh /piper/heavy_target_mask
```

Terminal 5, Docker: view the tracking debug image.

```bash
./L515_camera/view_l515_opencv.sh /piper/temporal_tracking_debug_image
```

## Real PiPER Arm Runtime

These commands can enable or move the physical arm. Keep the workspace clear and use the physical
emergency-stop/power procedure for emergencies. `disable_piper.sh` is only a software disable request.

Use separate host terminals. Start Docker in each terminal with the command from "Start A Foxy Terminal".

Terminal 1, Docker: start the PiPER driver and CAN interface.

```bash
./start_piper.sh
```

Defaults:

```text
CAN interface: can0
CAN bitrate: 1000000
ROS_DOMAIN_ID: 42
Auto-enable: false
```

Useful overrides:

```bash
PIPER_CAN_PORT=can1 ./start_piper.sh
PIPER_CAN_BITRATE=1000000 ./start_piper.sh
PIPER_ROS_DOMAIN_ID=42 ./start_piper.sh
```

Terminal 2, Docker: enable the real arm after the driver is running.

```bash
./enable_piper.sh
```

Disable the arm through the ROS service:

```bash
./disable_piper.sh
```

Show arm status:

```bash
ros2 topic echo /arm_status
```

Show joint feedback:

```bash
ros2 topic echo /joint_states_single
```

Move to all-zero joint target:

```bash
./reset_piper.sh
```

Move to the saved reset/home pose:

```bash
./reset_arm.sh
```

Start the manual PiPER GUI:

```bash
./start_gui.sh
```

Record measured joint bounds:

```bash
./calibrate_bounds.sh
```

## Real Arm Shutdown

1. Disable the arm:

```bash
./disable_piper.sh
```

2. Stop GUI/reset/manual command programs with `Ctrl+C`.
3. Stop the `start_piper.sh` terminal with `Ctrl+C`.
4. Use the physical power/emergency-stop procedure when required.

## Useful L515 Checks

Check USB speed from the host:

```bash
lsusb -t
```

The L515 should appear under `5000M` or `10000M` for USB 3. If it appears under `480M`, it is running as
USB 2.

Check USB type through RealSense inside Docker:

```bash
rs-enumerate-devices | grep -i "Usb Type\|Physical Port\|Serial Number"
```

View camera topics:

```bash
./L515_camera/view_l515_opencv.sh /camera/color/image_raw
./L515_camera/view_l515_opencv.sh /camera/aligned_depth_to_color/image_raw
```

RViz:

```bash
./L515_camera/view_l515_rviz.sh
```

## AI Environment Checks

Check the host Python 3.10 AI environment:

```bash
cd /home/wenjun/prl/Piper_arm
GROUNDED_SAM2_PYTHON=AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  ./AI_perception_tests/groundingdino_test/check_env.sh
```

## Common ROS Commands

List built ROS packages inside Docker:

```bash
ros2 pkg list | grep -E 'piper|realsense'
```

List active topics:

```bash
ros2 topic list
```

Echo a topic:

```bash
ros2 topic echo /piper/heavy_refresh_status
```

Check Docker image exists on the host:

```bash
sudo docker images piper-arm-foxy:local
```

## Notes

- The host remains Ubuntu 22.04 with ROS Humble. Foxy runs inside Docker.
- Every ROS terminal should be a Docker terminal.
- The heavy AI worker should run on the host, not inside Docker.
- Use `--net=host` so ROS 2 DDS discovery works with host networking.
- Use `--privileged` so camera/USB/CAN access is available to the container.
- Use `-e DISPLAY=$DISPLAY` and `-v /tmp/.X11-unix:/tmp/.X11-unix` for GUI viewers.
- The L515 perception workflow is read-only. The real PiPER arm scripts are separate and can move hardware.
- Hardware connectivity is not proven by `verify_installation.sh`; connect the L515 and CAN hardware before live tests.
