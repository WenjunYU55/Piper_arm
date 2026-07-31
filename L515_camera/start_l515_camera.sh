#!/usr/bin/env bash
set -eo pipefail

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

if ! ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
  echo "realsense2_camera is not available in the current ROS environment."
  echo "Build and source $ROOT/L515_camera/realsense_ws first:"
  echo "  cd $ROOT/L515_camera"
  echo "  ./install_realsense_build_deps.sh"
  echo "  ./build_realsense_ws.sh"
  exit 1
fi

echo "Starting L515 with RGB, depth, and aligned depth-to-color enabled."
echo "Using ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Close-range preset: ${L515_VISUAL_PRESET:-5} (5=Short Range, 3=Low Ambient Light)."
echo "Leave this running, then start the GPU vision pipeline components as needed."
ros2 launch realsense2_camera rs_launch.py \
  device_type:=l515 \
  enable_color:=true \
  enable_depth:=true \
  enable_confidence:=false \
  enable_infra:=false \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_fisheye1:=false \
  enable_fisheye2:=false \
  enable_pose:=false \
  enable_gyro:=false \
  enable_accel:=false \
  depth_module.profile:=640,480,30 \
  rgb_camera.profile:=640,480,30 \
  depth_module.global_time_enabled:=true \
  rgb_camera.global_time_enabled:=true \
  color_qos:=SENSOR_DATA \
  color_info_qos:=SENSOR_DATA \
  depth_qos:=SENSOR_DATA \
  depth_info_qos:=SENSOR_DATA \
  infra_qos:=SENSOR_DATA \
  infra_info_qos:=SENSOR_DATA \
  align_depth.enable:=true \
  clip_distance:=-1.0 \
  pointcloud.enable:=false \
  pointcloud.stream_index_filter:=0 \
  initial_reset:=false &

launch_pid=$!
trap 'kill "$launch_pid" 2>/dev/null || true; wait "$launch_pid" 2>/dev/null || true' EXIT INT TERM

# realsense-ros 4.0.4 exposes sensor controls only after the device is opened.
# The L515 firmware also initializes global time disabled, so apply all required
# controls to the live sensors rather than relying only on launch overrides.
preset="${L515_VISUAL_PRESET:-5}"
case "$preset" in
  3|5) ;;
  *)
    echo "L515_VISUAL_PRESET must be 5 (Short Range) or 3 (Low Ambient Light)." >&2
    exit 2
    ;;
esac

configured=false
for _ in $(seq 1 40); do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    wait "$launch_pid"
    exit $?
  fi
  if ros2 param set /camera/camera depth_module.visual_preset "$preset" >/dev/null 2>&1 &&
     ros2 param set /camera/camera depth_module.global_time_enabled true >/dev/null 2>&1 &&
     ros2 param set /camera/camera rgb_camera.global_time_enabled true >/dev/null 2>&1 &&
     ros2 param get /camera/camera depth_module.global_time_enabled 2>/dev/null | grep -q 'Boolean value is: True' &&
     ros2 param get /camera/camera rgb_camera.global_time_enabled 2>/dev/null | grep -q 'Boolean value is: True'; then
    configured=true
    break
  fi
  sleep 0.5
done

if [[ "$configured" != true ]]; then
  echo "Camera started, but its visual preset and global-time controls could not be applied." >&2
  echo "Check that /camera/camera exposes the depth_module and rgb_camera sensor options." >&2
  exit 1
fi

echo "Applied L515 visual preset $(ros2 param get /camera/camera depth_module.visual_preset | sed 's/^Integer value is: //')."
echo "Enabled host-corrected global timestamps for L515 depth and RGB streams."
wait "$launch_pid"
