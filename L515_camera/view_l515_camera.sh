#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

TOPIC=${1:-/camera/color/image_raw}

if ! ros2 pkg prefix rqt_image_view >/dev/null 2>&1; then
  echo "rqt_image_view is not installed. Install it with:"
  echo "  sudo apt-get install -y ros-foxy-rqt-image-view"
  exit 1
fi

echo "Opening rqt_image_view on ${TOPIC}"
ros2 run rqt_image_view rqt_image_view "${TOPIC}"
