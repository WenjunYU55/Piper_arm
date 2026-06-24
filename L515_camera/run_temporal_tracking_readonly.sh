#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

PARAMS_FILE="$ROOT/piper_ros_foxy/src/piper_mobile_manipulation/config/temporal_tracking_params.yaml"
SEED_MASK_TOPIC="${1:-/piper/detection_mask}"

echo "Starting read-only temporal mask tracking."
echo "Seed mask topic: ${SEED_MASK_TOPIC}"
echo "Real arm motion: disabled"

exec ros2 run piper_mobile_manipulation temporal_mask_tracker_node.py --ros-args \
  --params-file "$PARAMS_FILE" \
  -p seed_mask_topic:="$SEED_MASK_TOPIC" \
  -p dry_run:=true \
  -p enable_real_arm_motion:=false
