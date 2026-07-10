# Operator Commands

Command reference for running the PiPER + L515 perception stack from:

```bash
cd /home/prl/Piper_arm
```

Current GitHub `main` is the reverted supervised dry-run workflow version:

```text
5d995cf Revert "Add base-frame perception recovery workflow"
```

This means the newer base-frame recovery commit `fa32da6` is not active in GitHub `main`.

## Safety rules

- Do not enable real robot motion during perception/active-scan validation.
- `./start_piper.sh` starts the PiPER driver and CAN interface, but should not auto-enable the arm.
- `./enable_piper.sh`, `./reset_piper.sh`, `./reset_arm.sh`, and GUI joint commands can move the real robot.
- `disable_piper.sh` is only a software disable request. Use the physical emergency stop/power procedure for emergencies.
- Keep the workspace clear before enabling the arm.

## Standard environment

Use this in every manual ROS terminal:

```bash
cd /home/prl/Piper_arm
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
source install/setup.bash
```

```bash
source piper_ros_foxy/install/setup.bash
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

Use `ROS_LOCALHOST_ONLY=1` for single-machine validation. Do not mix `0` and `1` across terminals.

## Copy-paste one-line startup commands

These are the preferred commands when pasting into terminals. Use one separate terminal per command.

PiPER driver:

```bash
cd /home/prl/Piper_arm && PIPER_ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 ./start_piper.sh
```

Hand-eye TF:

```bash
bash -lc 'cd /home/prl/Piper_arm && source /opt/ros/foxy/setup.bash && source install/setup.bash && source piper_ros_foxy/install/setup.bash && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ./L515_camera/run_hand_eye_tf.sh'
```

Full GPU perception pipeline:

```bash
bash -lc 'cd /home/prl/Piper_arm && source /opt/ros/foxy/setup.bash && source install/setup.bash && source piper_ros_foxy/install/setup.bash && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ./L515_camera/run_gpu_vision_pipeline.sh'
```

Full GPU perception pipeline, reusing an already-running camera:

```bash
bash -lc 'cd /home/prl/Piper_arm && source /opt/ros/foxy/setup.bash && source install/setup.bash && source piper_ros_foxy/install/setup.bash && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && PIPER_REUSE_EXISTING_CAMERA=1 ./L515_camera/run_gpu_vision_pipeline.sh'
```

Active scan debug:

```bash
bash -lc 'cd /home/prl/Piper_arm && source /opt/ros/foxy/setup.bash && source install/setup.bash && source piper_ros_foxy/install/setup.bash && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ros2 launch piper_mobile_manipulation active_scan_debug.launch.py'
```

Open a sourced shell for manual ROS checks:

```bash
bash -lc 'cd /home/prl/Piper_arm && source /opt/ros/foxy/setup.bash && source install/setup.bash && source piper_ros_foxy/install/setup.bash && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && exec bash'
```

## Clean reset

Stop running terminals with `Ctrl+C`, then use one sourced terminal.

Reset ROS CLI discovery:

```bash
ros2 daemon stop
```

Check old processes:

```bash
pgrep -af 'realsense2_camera|rs_launch.py|run_gpu_vision_pipeline|run_heavy_model_worker|heavy_model_worker.py|run_sam2_live_worker|sam2_live_worker.py|run_heavy_refresh_bridge|sam2_live_bridge_node|target_tracker_node|scan_viewpoint_planner|viewpoint_reachability_filter|active_scan_debug'
```

If old vision/planning processes remain, stop their terminals or kill the specific stale process. Avoid killing PiPER unless you intend to restart the driver.

Archive stale perception queues:

```bash
mkdir -p /tmp/piper_reset_backup
```

```bash
mv /tmp/piper_heavy_refresh /tmp/piper_reset_backup/piper_heavy_refresh_$(date +%s) 2>/dev/null || true
```

```bash
mv /tmp/piper_sam2_live /tmp/piper_reset_backup/piper_sam2_live_$(date +%s) 2>/dev/null || true
```

Restart ROS CLI daemon:

```bash
ros2 daemon start
```

## Clean rebuild after reverting code

Use this if ROS nodes crash with message type-support errors such as:

```text
undefined symbol: piper_mobile_manipulation__msg__scene_object__convert_to_py
UnsupportedTypeSupport: Could not import 'rosidl_typesupport_c'
```

That means the source code and generated ROS build/install files do not match. Stop the relevant ROS nodes, then run:

```bash
cd /home/prl/Piper_arm/piper_ros_foxy
```

```bash
rm -rf build/piper_mobile_manipulation install/piper_mobile_manipulation log
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
colcon build --packages-select piper_mobile_manipulation --symlink-install
```

Then open new terminals and source the environment again before starting nodes.

## Recommended startup sequence

Use separate terminals.

### Terminal 1 — PiPER driver

```bash
cd /home/prl/Piper_arm
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
source install/setup.bash
```

```bash
source piper_ros_foxy/install/setup.bash
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
PIPER_ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 ./start_piper.sh
```

Expected:

```text
auto_enable is False
```

Do not run `enable_piper.sh` for dry-run perception validation.

### Terminal 2 — hand-eye TF

```bash
cd /home/prl/Piper_arm
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
source install/setup.bash
```

```bash
source piper_ros_foxy/install/setup.bash
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
./L515_camera/run_hand_eye_tf.sh
```

Expected:

```text
Publishing base_link -> camera_link
```

The final optical-frame check is still `base_link -> camera_color_optical_frame` after the camera is running, because RealSense provides the static camera-frame-to-optical-frame TF.

### Terminal 3 — full GPU perception pipeline

```bash
cd /home/prl/Piper_arm
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
source install/setup.bash
```

```bash
source piper_ros_foxy/install/setup.bash
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
./L515_camera/run_gpu_vision_pipeline.sh
```

This starts:

```text
L515 camera if no reusable camera is active
heavy_refresh_bridge_node
heavy_model_worker with PIPER_HEAVY_DEVICE=cuda
sam2_live_worker with PIPER_SAM2_DEVICE=cuda
sam2_live_bridge_node
GPU geometry / depth-to-3D nodes
target cloud node
```

Do not also start `run_heavy_model_worker.sh` manually while this is running.

If the camera is already running and publishing, reuse it:

```bash
PIPER_REUSE_EXISTING_CAMERA=1 ./L515_camera/run_gpu_vision_pipeline.sh
```

### Terminal 4 — active scan debug

```bash
cd /home/prl/Piper_arm
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
source install/setup.bash
```

```bash
source piper_ros_foxy/install/setup.bash
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
ros2 launch piper_mobile_manipulation active_scan_debug.launch.py
```

This is dry-run/debug. It does not enable real robot motion.

## Camera-first debugging

If tracking does not work, verify the camera before debugging SAM2 or heavy refresh.

Check publishers:

```bash
ros2 topic info /camera/color/image_raw --verbose
```

```bash
ros2 topic info /camera/aligned_depth_to_color/image_raw --verbose
```

```bash
ros2 topic info /camera/color/camera_info --verbose
```

Expected:

```text
Publisher count: 1
```

Check image rate:

```bash
ros2 topic hz /camera/color/image_raw
```

If publisher count is `0` while RealSense processes exist, stop the GPU pipeline and start only the camera:

```bash
./L515_camera/start_l515_camera.sh
```

Then check:

```bash
ros2 topic hz /camera/color/image_raw
```

If camera works alone, restart the full pipeline with camera reuse:

```bash
PIPER_REUSE_EXISTING_CAMERA=1 ./L515_camera/run_gpu_vision_pipeline.sh
```

## Verification commands

Check live nodes:

```bash
ros2 node list | grep -E 'camera|heavy|sam2|target|scan|reach'
```

Expected core vision nodes:

```text
/camera/camera
/heavy_refresh_bridge_node
/sam2_live_bridge_node
/sam2_target_tracker
/object_frame_broadcaster
/sam2_depth_to_3d
/target_cloud_node
```

Check camera-to-base TF:

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

Check heavy refresh:

```bash
ros2 topic echo /piper/heavy_refresh_status
```

Check SAM2 status:

```bash
ros2 topic echo /piper/sam2_tracking_status
```

Check target 3D:

```bash
ros2 topic echo /piper/target_3d
```

Check filtered/predicted tracker output:

```bash
ros2 topic echo /piper/tracked_target
```

Check persistent object TF:

```bash
ros2 run tf2_ros tf2_echo base_link tracked_object_frame
```

Check predicted object TF:

```bash
ros2 run tf2_ros tf2_echo base_link predicted_object_frame
```

Check target cloud:

```bash
ros2 topic echo /piper/target_cloud_status
```

Check active scan outputs:

```bash
ros2 topic echo /piper/scan_viewpoints
```

```bash
ros2 topic echo /piper/reachable_scan_viewpoints
```

Note: in the reverted current GitHub version, the newer `/piper/target/raw_base`, `/piper/target/filtered_base`, `/piper/target/predicted_base`, and `tracked_object_frame` outputs from `fa32da6` are not active.

## Forcing a heavy refresh

Use this only after camera topics are publishing.

```bash
ros2 topic pub --once /piper/heavy_refresh_request std_msgs/msg/String "{data: '{\"request_id\":\"manual_initial_cube\",\"reason\":\"manual_initial_cube\",\"tracking\":{\"tracking_confidence\":0.0},\"dry_run\":true,\"real_arm_motion\":false}'}"
```

Then monitor:

```bash
ros2 topic echo /piper/heavy_refresh_status
```

Good signs:

```text
queued
published
sam2_seed_queued
```

Check output masks:

```bash
ros2 topic echo /piper/heavy_target_mask
```

```bash
ros2 topic echo /piper/sam2_target_mask
```

## Viewing masks and point clouds

View RGB:

```bash
./L515_camera/view_l515_opencv.sh /camera/color/image_raw
```

View SAM2 target mask:

```bash
./L515_camera/view_l515_opencv.sh /piper/sam2_target_mask
```

View heavy target mask:

```bash
./L515_camera/view_l515_opencv.sh /piper/heavy_target_mask
```

Open RViz:

```bash
./L515_camera/view_l515_rviz.sh
```

Use fixed frame:

```text
base_link
```

Useful displays:

```text
/piper/target_cloud
/piper/target_landmark
/piper/supervised_workflow_markers
TF
```

## Target cloud capture

At each stationary viewpoint:

```bash
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: capture}"
```

Check status:

```bash
ros2 topic echo /piper/target_cloud_status
```

Save:

```bash
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: save}"
```

Clear:

```bash
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: clear}"
```

Saved PLY files are written under:

```text
datasets/target_clouds
```

## Supervised cube workflow

This coordinator is dry-run only. The operator performs any proposed movement manually.

Start after PiPER, hand-eye TF, and GPU perception are stable:

```bash
./L515_camera/run_supervised_cube_workflow.sh
```

Monitor:

```bash
ros2 topic echo --full-length /piper/supervised_workflow_status
```

```bash
ros2 topic echo --full-length /piper/removal_plan
```

Start workflow:

```bash
ros2 service call /supervised_cube_workflow/start std_srvs/srv/Trigger '{}'
```

Approve a dry-run proposal after review:

```bash
ros2 service call /supervised_cube_workflow/approve_plan std_srvs/srv/Trigger '{}'
```

Confirm manual action complete:

```bash
ros2 service call /supervised_cube_workflow/confirm_action_complete std_srvs/srv/Trigger '{}'
```

Capture a scan view:

```bash
ros2 service call /supervised_cube_workflow/capture_view std_srvs/srv/Trigger '{}'
```

Finish:

```bash
ros2 service call /supervised_cube_workflow/finish_scan std_srvs/srv/Trigger '{}'
```

Abort:

```bash
ros2 service call /supervised_cube_workflow/abort std_srvs/srv/Trigger '{}'
```

## Real PiPER arm commands

Start driver:

```bash
./start_piper.sh
```

Enable real arm:

```bash
./enable_piper.sh
```

Disable real arm:

```bash
./disable_piper.sh
```

Move to all-zero joint target:

```bash
./reset_piper.sh
```

Move to saved home/reset pose:

```bash
./reset_arm.sh
```

Start GUI:

```bash
./start_gui.sh
```

Only use real motion commands when the arm workspace is clear.

## GitHub version commands

Show recent commits:

```bash
git log --oneline --decorate --graph -10
```

Current rollback commit:

```text
5d995cf Revert "Add base-frame perception recovery workflow"
```

Reverted commit:

```text
fa32da6 Add base-frame perception recovery workflow
```

Previous supervised dry-run version:

```text
dac547e Add supervised dry-run cube workflow
```

View a file from a previous commit:

```bash
git show dac547e:path/to/file
```

Temporarily browse an old version:

```bash
git checkout dac547e
```

Return to current main:

```bash
git checkout main
```
