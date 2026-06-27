# Operator Commands

This file is the command reference for running the PiPER + L515 system from a fresh clone.

There are two separate workflows:

1. Read-only L515 perception. This does not move the robot.
2. Real PiPER arm operation. These commands can enable or move the physical arm.

Run commands from the repository root unless a section says otherwise:

```bash
cd /home/prl/Piper_arm
```

If the repository is cloned somewhere else, replace `/home/prl/Piper_arm` with that path.

## Safety rules

- The L515 perception scripts are read-only and do not publish real arm motion commands.
- `start_piper.sh` starts the real PiPER driver and CAN interface, but does not auto-enable the arm by default.
- After `enable_piper.sh` succeeds, joint commands can move the real robot.
- `reset_piper.sh`, `reset_arm.sh`, and `start_gui.sh` can move the real robot if the driver is running and enabled.
- `disable_piper.sh` requests software disable through the PiPER ROS service. It is not a physical emergency stop.
- For emergency stop, use the robot's physical power/emergency-stop procedure.

## Common environment

Most scripts source their environment automatically. For manual ROS commands, use:

```bash
export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
```

Check that ROS can see the camera:

```bash
ros2 topic list | grep camera
```

Check PiPER-related topics:

```bash
ros2 topic list | grep piper
```

## Read-only L515 perception runtime

### GPU SAM2 live tracking

Install the CUDA AI environment once:

```bash
cd /home/prl/Piper_arm
./AI_perception_tests/groundingdino_test/setup_gpu_env.sh
```

Start the complete read-only pipeline:

```bash
./L515_camera/run_gpu_vision_pipeline.sh
```

View the GPU-propagated mask:

```bash
./L515_camera/view_l515_opencv.sh /piper/sam2_target_mask
```

GroundingDINO identifies the target and obstacles at startup and on tracking, occlusion, scene-change,
or periodic refresh events. SAM2 creates the initial masks and propagates all labelled objects between
those events. Both inference workers require CUDA. Tracking status, measured FPS, object labels, IDs,
mask areas, and device are published on `/piper/sam2_tracking_status`. This workflow does not command
arm motion.

Useful outputs:

```text
/piper/sam2_target_mask
/piper/sam2_obstacle_mask
/piper/sam2_unsafe_obstacle_mask
/piper/sam2_candidate_movable_obstacle_mask
/piper/sam2_object_ids
/piper/target_3d
/piper/target_cloud
```

Live tracking uses a 384-pixel-wide SAM2 input by default and publishes masks restored to the native
640x480 camera resolution. Set `PIPER_SAM2_INFERENCE_WIDTH=640` before startup to disable reduction.

For the highest-quality cloud, start the pipeline with live-mask accumulation disabled, request one
full-resolution capture at each stationary viewpoint, then save or clear the accumulated cloud:

```bash
PIPER_CLOUD_ACCUMULATE_LIVE=false ./L515_camera/run_gpu_vision_pipeline.sh

export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: capture}"
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: save}"
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: clear}"
```

Wait for `/piper/target_cloud_status` to report
`mask_source: full_resolution_refinement` before moving to the next viewpoint or saving.

Saved PLY files are written to `datasets/target_clouds`. Camera-frame accumulation works for a fixed
L515. Multi-view accumulation requires a published camera-to-base transform and
`PIPER_CLOUD_FRAME=piper_base_link PIPER_CLOUD_REQUIRE_TF=true`.

Use separate terminals.

Terminal 1: start the RealSense L515 camera.

```bash
cd /home/prl/Piper_arm
./L515_camera/start_l515_camera.sh
```

What it does:

- Starts the RealSense ROS camera node.
- Publishes color, depth, aligned depth, camera info, metadata, and IMU topics.
- Does not move the arm.

Expected camera topics include:

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
/camera/aligned_depth_to_color/camera_info
```

Terminal 2: start the read-only heavy-refresh filesystem bridge.

```bash
cd /home/prl/Piper_arm
./L515_camera/run_heavy_refresh_bridge.sh
```

What it does:

- Listens for `/piper/heavy_refresh_request`.
- Saves camera snapshots/jobs into `/tmp/piper_heavy_refresh`.
- Publishes returned heavy model masks to ROS.
- Does not run GroundingDINO/SAM2 itself.
- Does not move the arm.

Main topics:

```text
/piper/heavy_refresh_request
/piper/heavy_refresh_status
/piper/heavy_target_mask
/piper/heavy_obstacle_mask
/piper/candidate_movable_obstacle_mask
/piper/unsafe_obstacle_mask
```

Terminal 3: start the isolated heavy-model worker.

```bash
cd /home/prl/Piper_arm
./L515_camera/run_heavy_model_worker.sh
```

What it does:

- Runs in the isolated Python 3.10 GroundingDINO/SAM2 environment.
- Reads jobs from `/tmp/piper_heavy_refresh`.
- Writes target and obstacle mask results back to the spool directory.
- Has no ROS imports.
- Cannot command arm motion.

The production GPU pipeline requires CUDA:

```bash
export PIPER_HEAVY_DEVICE=cuda
./L515_camera/run_heavy_model_worker.sh
```

Terminal 4: monitor heavy-refresh status.

```bash
cd /home/prl/Piper_arm
export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
ros2 topic echo /piper/heavy_refresh_status
```

What it does:

- Shows whether heavy-refresh jobs are queued, processing, done, or failed.
- Useful for checking if the bridge and worker are communicating.
- Does not move the arm.

Terminal 5: start temporal mask tracking.

```bash
cd /home/prl/Piper_arm
./L515_camera/run_temporal_tracking_readonly.sh /piper/heavy_target_mask
```

What it does:

- Uses `/piper/heavy_target_mask` as the seed mask from GroundingDINO/SAM2.
- Tracks the target with lightweight CPU tracking between heavy refreshes.
- Publishes debug/status/mask topics.
- Requests periodic or event-driven heavy refreshes.
- Does not move the arm.

Main topics:

```text
/piper/temporal_target_mask
/piper/temporal_tracking_debug_image
/piper/temporal_tracking_status
/piper/heavy_refresh_request
```

Terminal 6: view the tracking debug image.

```bash
cd /home/prl/Piper_arm
./L515_camera/view_l515_opencv.sh /piper/temporal_tracking_debug_image
```

What it does:

- Opens an OpenCV image viewer.
- Shows the live debug overlay from the temporal tracker.
- Does not move the arm.

Optional: view heavy snapshot masks.

```bash
./L515_camera/view_l515_opencv.sh /piper/heavy_target_mask
./L515_camera/view_l515_opencv.sh /piper/heavy_obstacle_mask
./L515_camera/view_l515_opencv.sh /piper/candidate_movable_obstacle_mask
./L515_camera/view_l515_opencv.sh /piper/unsafe_obstacle_mask
```

Black screens usually mean that topic has not published a non-empty image yet.

## Camera-only checks

Check the L515 ROS environment:

```bash
cd /home/prl/Piper_arm/L515_camera
./check_l515_ros.sh
```

View the raw camera image:

```bash
cd /home/prl/Piper_arm
./L515_camera/view_l515_opencv.sh /camera/color/image_raw
```

View aligned depth:

```bash
./L515_camera/view_l515_opencv.sh /camera/aligned_depth_to_color/image_raw
```

Open RViz:

```bash
./L515_camera/view_l515_rviz.sh
```

## Real PiPER arm commands

These commands are intentionally separate from the read-only perception workflow.

Terminal 1: start the PiPER driver.

```bash
cd /home/prl/Piper_arm
./start_piper.sh
```

What it does:

- Sources ROS 2 Foxy and the PiPER workspace.
- Checks the PiPER ROS package exists.
- Checks Python runtime dependencies.
- Checks the CAN interface, default `can0`.
- Resets and activates CAN.
- Launches `piper start_single_piper.launch.py`.
- Does not auto-enable the arm by default.

Useful environment overrides:

```bash
PIPER_CAN_PORT=can1 ./start_piper.sh
PIPER_CAN_BITRATE=1000000 ./start_piper.sh
PIPER_ROS_DOMAIN_ID=42 ./start_piper.sh
```

Terminal 2: enable the real arm.

```bash
cd /home/prl/Piper_arm
./enable_piper.sh
```

What it does:

- Calls `/enable_srv` with `enable_request: true`.
- After this succeeds, joint commands can move the real robot.

Disable the real arm through the ROS service:

```bash
./disable_piper.sh
```

What it does:

- Calls `/enable_srv` with `enable_request: false`.
- This is a software disable request, not a physical emergency stop.

## Reset and home commands

Move to all-zero joint target:

```bash
cd /home/prl/Piper_arm
./reset_piper.sh
```

What it does:

- Runs `reset_piper.py`.
- Waits for `/joint_states_single`.
- Publishes a joint target to `/joint_ctrl_single`.
- Target is all joints at `0.0`, with gripper command included.
- Can move the real arm if the driver is enabled.

Move to your saved reset/home pose:

```bash
./reset_arm.sh
```

What it does:

- Runs `reset_arm.py`.
- Waits for `/joint_states_single`.
- Publishes a saved joint target to `/joint_ctrl_single`.
- Can move the real arm if the driver is enabled.

Current saved pose:

```text
joint1 = -1.55024828
joint2 = -0.040347972
joint3 =  0.03410302
joint4 =  0.018979072
joint5 =  0.320917268
joint6 =  1.07777754
joint7 =  0.01981
```

## Manual GUI

Start the PiPER manual GUI:

```bash
cd /home/prl/Piper_arm
./start_gui.sh
```

What it does:

- Opens the Tkinter PiPER control GUI.
- Shows feedback from `/joint_states_single` and `/arm_status`.
- Can call `/enable_srv`.
- Can publish joint targets to `/joint_ctrl_single`.
- Uses `piper_joint_bounds.json` for measured joint limits when available.
- Can move the real arm if the driver is enabled.

Use the GUI only when the arm workspace is clear.

## Joint bounds calibration

Record measured joint bounds:

```bash
cd /home/prl/Piper_arm
./calibrate_bounds.sh
```

What it does:

- Runs `piper_calibrate_bounds.py`.
- Reads live feedback from `/joint_states_single`.
- Does not command motion itself.
- Prompts you to manually move each joint to min/max positions.
- Writes `piper_joint_bounds.json`.

This file is used by:

```text
start_piper.sh
piper_gui_native.py
```

## Useful ROS inspection commands

List camera topics:

```bash
ros2 topic list | grep camera
```

List PiPER topics:

```bash
ros2 topic list | grep piper
```

List PiPER services:

```bash
ros2 service list | grep enable
```

Show arm status:

```bash
ros2 topic echo /arm_status
```

Show joint feedback:

```bash
ros2 topic echo /joint_states_single
```

Show temporal tracker status:

```bash
ros2 topic echo /piper/temporal_tracking_status
```

Show heavy-refresh status:

```bash
ros2 topic echo /piper/heavy_refresh_status
```

## Shutdown order

For read-only perception:

1. Close viewer windows.
2. Stop temporal tracker with `Ctrl+C`.
3. Stop heavy model worker with `Ctrl+C`.
4. Stop heavy-refresh bridge with `Ctrl+C`.
5. Stop the L515 camera with `Ctrl+C`.

For real arm operation:

1. Disable the arm:

   ```bash
   ./disable_piper.sh
   ```

2. Stop GUI/reset/manual command programs.
3. Stop `start_piper.sh` terminal with `Ctrl+C`.
4. Use the physical power/emergency-stop procedure when required.

## Command summary

| Command | Purpose | Can move real arm? |
|---|---|---|
| `./L515_camera/start_l515_camera.sh` | Start L515 camera | No |
| `./L515_camera/run_heavy_refresh_bridge.sh` | ROS/filesystem bridge for heavy refresh | No |
| `./L515_camera/run_heavy_model_worker.sh` | Isolated GroundingDINO/SAM2 worker | No |
| `./L515_camera/run_temporal_tracking_readonly.sh /piper/heavy_target_mask` | Live mask tracking | No |
| `./L515_camera/view_l515_opencv.sh <topic>` | Image viewer | No |
| `./start_piper.sh` | Start PiPER driver/CAN | Not by itself |
| `./enable_piper.sh` | Enable real arm | Enables motion |
| `./disable_piper.sh` | Disable real arm through ROS service | Stops accepting normal commands after success |
| `./reset_piper.sh` | Move to zero joint target | Yes, if enabled |
| `./reset_arm.sh` | Move to saved home/reset pose | Yes, if enabled |
| `./start_gui.sh` | Manual GUI control | Yes, if enabled |
| `./calibrate_bounds.sh` | Record joint bounds from feedback | Does not command motion itself |
