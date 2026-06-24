#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPOOL_DIR="${PIPER_HEAVY_REFRESH_SPOOL:-/tmp/piper_heavy_refresh}"

# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

echo "Starting read-only Foxy heavy-refresh filesystem bridge."
echo "Spool directory: ${SPOOL_DIR}"
echo "Real arm motion: disabled"

exec ros2 run piper_mobile_manipulation heavy_refresh_bridge_node.py --ros-args \
  -p spool_dir:="$SPOOL_DIR" \
  -p dry_run:=true \
  -p enable_real_arm_motion:=false
