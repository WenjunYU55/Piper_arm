# PiPER L515 Active Scan Notes

## Purpose

This update adds a dry-run active perception layer for the arm-mounted Intel RealSense L515. The goal is to plan and inspect scan viewpoints around the current object of interest, visualize the scan state, and save synchronized RGB-D samples for later perception development.

The current implementation is intentionally passive. It does not move the PiPER arm, does not push objects, and does not publish real servo commands.

## Current Safety State

- Real arm motion remains disabled by default.
- Real pushing remains disabled by default.
- The active scan nodes do not publish `/piper/servo_cmd`.
- The active scan launch files do not start the camera or perception stack; those are started separately.
- The current detector remains the simple green HSV detector.
- YOLO, SAM2, GroundingDINO, and VLM detection are not active integrations in the working scan pipeline.

## Added Active Scan Components

- `scan_viewpoint_planner_node.py`
  - Subscribes to the target object position and scan status inputs.
  - Publishes candidate scan viewpoints on `/piper/scan_viewpoints`.
  - Publishes scan coverage state on `/piper/scan_coverage`.
  - Plans a dry-run arc around the object, targeting about 250 degrees when configured.

- `viewpoint_reachability_filter_node.py`
  - Subscribes to planned viewpoints and status topics.
  - Publishes conservative reachable viewpoints on `/piper/reachable_scan_viewpoints`.
  - Rejects viewpoints when distance, height change, arm status, target status, or dry-run safety checks fail.

- `active_scan_debug_overlay_node.py`
  - Builds a visual debug image on `/piper/active_scan_debug_image`.
  - Overlays detector state, target distance, planned viewpoint count, reachable viewpoint count, scan coverage, scan quality, useful coverage, and safety status.

- `scan_capture_node.py`
  - Saves synchronized RGB, depth, mask, and metadata samples.
  - Publishes `/piper/scan_capture_status` and `/piper/scan_summary`.
  - Writes scan sessions under `/home/prl/Piper_arm/datasets/active_scan`.

- `scan_quality_node.py`
  - Scores each live RGB-D-mask view using mask size, edge margin, depth validity, and depth noise.
  - Publishes per-view quality on `/piper/scan_quality`.
  - Publishes readable quality details on `/piper/scan_quality_debug`.
  - Publishes approximate useful dry-run coverage on `/piper/useful_scan_coverage`.
  - Does not command the arm or publish `/piper/servo_cmd`.

- `occlusion_checker_node.py`
  - Uses aligned depth, the HSV mask, target 3D depth, and scan quality to detect closer-depth occlusion near the object.
  - Publishes structured status on `/piper/occlusion_status`.
  - Publishes readable status on `/piper/occlusion_debug`.
  - Does not label occluders, plan pushes, command the arm, or publish `/piper/servo_cmd`.

## Added Launch Files

- `active_scan_debug.launch.py`
  - Starts the scan planner, reachability filter, debug overlay, scan quality node, and occlusion checker.

- `active_scan_capture_debug.launch.py`
  - Starts the scan planner, reachability filter, debug overlay, scan quality node, occlusion checker, and scan capture node.
  - Does not launch the camera or perception nodes.

## Typical Runtime Flow

Terminal 1:

```bash
export ROS_DOMAIN_ID=42
/home/prl/Piper_arm/L515_camera/start_l515_camera.sh
```

Terminal 2:

```bash
export ROS_DOMAIN_ID=42
cd /home/prl/Piper_arm/L515_camera
./run_l515_perception.sh
```

Terminal 3:

```bash
export ROS_DOMAIN_ID=42
source /home/prl/Piper_arm/piper_ros_foxy/install/setup.bash
ros2 launch piper_mobile_manipulation active_scan_capture_debug.launch.py
```

Terminal 4:

```bash
export ROS_DOMAIN_ID=42
source /home/prl/Piper_arm/piper_ros_foxy/install/setup.bash
TOPIC=/piper/active_scan_debug_image
/home/prl/Piper_arm/L515_camera/view_l515_opencv.sh "$TOPIC"
```

## Useful Checks

```bash
ros2 topic echo /piper/scan_capture_status
ros2 topic echo /piper/scan_summary
ros2 topic echo /piper/scan_quality
ros2 topic echo /piper/scan_quality_debug
ros2 topic echo /piper/occlusion_status
ros2 topic echo /piper/occlusion_debug
ros2 topic echo /piper/useful_scan_coverage
ros2 topic echo /piper/scan_viewpoints
ros2 topic echo /piper/reachable_scan_viewpoints
ros2 topic list | grep servo
ls -R /home/prl/Piper_arm/datasets/active_scan
```

## Build

```bash
cd /home/prl/Piper_arm/piper_ros_foxy
colcon build --packages-select piper_mobile_manipulation
source install/setup.bash
```

## Current Limitation

The saved scan frames are captured from the fixed current camera pose at a timed interval. They are not yet captured from physically different arm viewpoints. Real viewpoint execution should only be added after the planner, reachability checks, quality scoring, and dry-run dataset capture are stable.
