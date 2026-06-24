#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

PARAMS_FILE="$ROOT/piper_ros_foxy/src/piper_mobile_manipulation/config/scan_capture_params.yaml"
CAPTURE_INTERVAL_SEC="${1:-0.5}"
MAX_FRAMES="${2:-40}"

echo "Starting dry-run temporal RGB-D capture."
echo "Interval: ${CAPTURE_INTERVAL_SEC}s; frames: ${MAX_FRAMES}"
echo "Real arm motion: disabled"

exec ros2 run piper_mobile_manipulation scan_capture_node.py --ros-args \
  --params-file "$PARAMS_FILE" \
  -p capture_interval_sec:="$CAPTURE_INTERVAL_SEC" \
  -p max_frames_per_scan:="$MAX_FRAMES" \
  -p require_valid_target:=false \
  -p require_mask:=false \
  -p dry_run:=true \
  -p enable_real_arm_motion:=false
