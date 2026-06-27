#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

echo "Starting L515 SAM2 target cloud accumulation."
echo "Publish 'save' or 'clear' on /piper/target_cloud_request."
exec ros2 run piper_mobile_manipulation target_cloud_node.py --ros-args \
  -p mask_topic:=/piper/sam2_target_mask \
  -p target_frame:="${PIPER_CLOUD_FRAME:-camera_color_optical_frame}" \
  -p require_transform:="${PIPER_CLOUD_REQUIRE_TF:-false}"
