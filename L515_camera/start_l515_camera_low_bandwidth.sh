#!/usr/bin/env bash
set -eo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

profile_path="${PIPER_CAMERA_PROFILE_PATH:-$ROOT/L515_camera/rgb_profile.conf}"
saved_color_width=640
saved_color_height=480
saved_color_fps=30
if [ -r "$profile_path" ]; then
  IFS=, read -r saved_color_width saved_color_height saved_color_fps < "$profile_path"
fi
color_width="${PIPER_CAMERA_COLOR_WIDTH:-$saved_color_width}"
color_height="${PIPER_CAMERA_COLOR_HEIGHT:-$saved_color_height}"
color_fps="${PIPER_CAMERA_COLOR_FPS:-$saved_color_fps}"
case "${color_width},${color_height},${color_fps}" in
  640,480,15|640,480,30|960,540,15|960,540,30|1280,720,15|1280,720,30|1920,1080,15|1920,1080,30) ;;
  *)
    echo "Unsupported L515 RGB profile: ${color_width}x${color_height}@${color_fps}." >&2
    exit 2
    ;;
esac
rgb_profile="${color_width},${color_height},${color_fps}"

if ! ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
  echo "realsense2_camera is not available in the current ROS environment."
  echo "Build and source $ROOT/L515_camera/realsense_ws first:"
  echo "  cd $ROOT/L515_camera"
  echo "  ./build_realsense_ws.sh"
  exit 1
fi

echo "Starting L515 with reduced depth bandwidth."
echo "Use this when the USB controller or DDS path resets during the normal launch."
echo "RGB profile: ${color_width}x${color_height}@${color_fps}; depth profile: 320x240@30."
echo "Close-range preset: ${L515_VISUAL_PRESET:-5} (5=Short Range, 3=Low Ambient Light)."
ros2 launch realsense2_camera rs_launch.py \
  device_type:=l515 \
  enable_color:=true \
  enable_depth:=true \
  enable_confidence:=true \
  enable_infra:=false \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_fisheye1:=false \
  enable_fisheye2:=false \
  enable_pose:=false \
  enable_gyro:=false \
  enable_accel:=false \
  depth_module.profile:=320,240,30 \
  rgb_camera.profile:="$rgb_profile" \
  depth_module.global_time_enabled:=true \
  rgb_camera.global_time_enabled:=true \
  color_qos:=SENSOR_DATA \
  color_info_qos:=SENSOR_DATA \
  depth_qos:=SENSOR_DATA \
  depth_info_qos:=SENSOR_DATA \
  infra_qos:=SENSOR_DATA \
  infra_info_qos:=SENSOR_DATA \
  align_depth.enable:=true \
  enable_sync:=true \
  clip_distance:=-1.0 \
  pointcloud.enable:=false \
  pointcloud.stream_index_filter:=0 \
  initial_reset:=false &

launch_pid=$!
trap 'kill "$launch_pid" 2>/dev/null || true; wait "$launch_pid" 2>/dev/null || true' EXIT INT TERM

preset="${L515_VISUAL_PRESET:-5}"
case "$preset" in
  3|5) ;;
  *)
    echo "L515_VISUAL_PRESET must be 5 (Short Range) or 3 (Low Ambient Light)." >&2
    exit 2
    ;;
esac

configured=false
camera_param() {
  # Foxy's direct-discovery parameter CLI can ignore its ROS spin bound while
  # the target node is still starting.  An OS timeout prevents a missing or
  # late camera from turning parameter discovery into an unbounded process.
  timeout --signal=TERM --kill-after=1s 2s \
    ros2 param "$@"
}
camera_stream_ready() {
  # Avoid changing controls while the depth/confidence/RGB streams are still
  # claiming their USB interfaces.
  (
    set +o pipefail
    timeout --signal=TERM --kill-after=1s 3s \
      ros2 topic echo --qos-profile sensor_data \
        /camera/color/camera_info sensor_msgs/msg/CameraInfo 2>/dev/null |
      head -n 1 | grep -q .
  )
}
for _ in $(seq 1 40); do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    wait "$launch_pid"
    exit $?
  fi
  if camera_stream_ready &&
     camera_param get --no-daemon --spin-time 0.5 /camera/camera depth_module.global_time_enabled >/dev/null 2>&1 &&
     camera_param get --no-daemon --spin-time 0.5 /camera/camera rgb_camera.global_time_enabled >/dev/null 2>&1 &&
     camera_param set --no-daemon --spin-time 0.5 /camera/camera depth_module.visual_preset "$preset" >/dev/null 2>&1 &&
     camera_param set --no-daemon --spin-time 0.5 /camera/camera depth_module.global_time_enabled true >/dev/null 2>&1 &&
     camera_param set --no-daemon --spin-time 0.5 /camera/camera rgb_camera.global_time_enabled true >/dev/null 2>&1 &&
     camera_param get --no-daemon --spin-time 0.5 /camera/camera depth_module.global_time_enabled 2>/dev/null | grep -q 'Boolean value is: True' &&
     camera_param get --no-daemon --spin-time 0.5 /camera/camera rgb_camera.global_time_enabled 2>/dev/null | grep -q 'Boolean value is: True'; then
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

echo "Applied L515 visual preset $(camera_param get --no-daemon --spin-time 0.5 /camera/camera depth_module.visual_preset | sed 's/^Integer value is: //')."
echo "Enabled host-corrected global timestamps for L515 depth and RGB streams."
wait "$launch_pid"
