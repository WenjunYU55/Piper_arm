#!/usr/bin/env bash
set -eo pipefail

export ROS_DOMAIN_ID=42
L515_REQUIRE_REALSENSE=1
L515_REQUIRE_PIPER=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

OUT_DIR=${1:-/home/prl/Piper_arm/datasets/plant_tracking/bags/l515_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$OUT_DIR")"

echo "Recording L515 dataset bag to ${OUT_DIR}"
ros2 bag record -o "$OUT_DIR" \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info \
  /piper/detection_debug_image
