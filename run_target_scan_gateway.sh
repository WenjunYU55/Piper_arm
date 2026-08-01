#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/source_piper_foxy_environment.sh"
export ROS_DOMAIN_ID="${PIPER_TRACKED_ROBOT_ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=0
# The gateway is the only PiPER process placed on the tracked robot network.
# Leave its external Fast DDS profile under deployment control.
if [ -n "${PIPER_GATEWAY_FASTDDS_PROFILE:-}" ]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="$PIPER_GATEWAY_FASTDDS_PROFILE"
else
  unset FASTRTPS_DEFAULT_PROFILES_FILE
fi

exec ros2 launch piper_mobile_manipulation target_scan_gateway.launch.py \
  piper_base_frame:="${PIPER_GATEWAY_BASE_FRAME:-piper_base_link}" \
  mission_spool_root:="${PIPER_MISSION_SPOOL_ROOT:-/tmp/piper_target_scan_missions}"
