#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

RECOVERY_ENABLED="${PIPER_TIMESTAMP_AUTO_RECOVERY:-1}"
case "$RECOVERY_ENABLED" in
  0) ROS_RECOVERY_ENABLED=false ;;
  1) ROS_RECOVERY_ENABLED=true ;;
  *)
    echo "PIPER_TIMESTAMP_AUTO_RECOVERY must be 0 or 1." >&2
    exit 2
    ;;
esac

exec ros2 run piper_mobile_manipulation camera_timestamp_watchdog_node.py --ros-args \
  --params-file "$ROOT/piper_ros_foxy/install/piper_mobile_manipulation/share/piper_mobile_manipulation/config/camera_timestamp_watchdog_params.yaml" \
  -p enable_recovery_request:="$ROS_RECOVERY_ENABLED" \
  -p recovery_request_path:="${PIPER_VISION_RECOVERY_REQUEST:-/tmp/piper_vision_recovery/request.yaml}"
