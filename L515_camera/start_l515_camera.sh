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
echo "Leave this running, then start ./run_l515_perception.sh in another terminal."
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
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=640x480x30 \
  color_qos:=SENSOR_DATA \
  color_info_qos:=SENSOR_DATA \
  depth_qos:=SENSOR_DATA \
  depth_info_qos:=SENSOR_DATA \
  infra_qos:=SENSOR_DATA \
  infra_info_qos:=SENSOR_DATA \
  align_depth.enable:=true \
  pointcloud.enable:=false \
  pointcloud.stream_index_filter:=0 \
  initial_reset:=false
