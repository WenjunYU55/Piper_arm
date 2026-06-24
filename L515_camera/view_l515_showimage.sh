#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

TOPIC=${1:-/camera/color/image_raw}

if ! ros2 pkg prefix image_tools >/dev/null 2>&1; then
  echo "image_tools is not installed. Install it with:"
  echo "  sudo apt-get install -y ros-foxy-image-tools"
  exit 1
fi

echo "Opening image_tools/showimage on ${TOPIC}"
ros2 run image_tools showimage --ros-args -r image:="${TOPIC}"
