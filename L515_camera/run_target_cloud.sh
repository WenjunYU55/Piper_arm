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
  -p accumulate_live_masks:="${PIPER_CLOUD_ACCUMULATE_LIVE:-false}" \
  -p refined_match_tolerance_sec:="${PIPER_CLOUD_REFINED_MATCH_TOLERANCE_SEC:-0.15}" \
  -p pixel_stride:="${PIPER_CLOUD_PIXEL_STRIDE:-1}" \
  -p voxel_size_m:="${PIPER_CLOUD_VOXEL_SIZE_M:-0.001}" \
  -p target_frame:="${PIPER_CLOUD_FRAME:-base_link}" \
  -p require_transform:="${PIPER_CLOUD_REQUIRE_TF:-true}"
