#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

L515_REQUIRE_REALSENSE=1
L515_REQUIRE_PIPER=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

RVIZ_CONFIG="$ROOT/L515_camera/l515_camera.rviz"

if ! command -v rviz2 >/dev/null 2>&1; then
  echo "rviz2 is not installed. Install it with:"
  echo "  sudo apt-get install -y ros-foxy-rviz2"
  exit 1
fi

echo "Opening RViz with ${RVIZ_CONFIG}"
rviz2 -d "${RVIZ_CONFIG}"
