#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

TOPIC=${1:-/camera/color/image_raw}

echo "Opening OpenCV viewer on ${TOPIC}"
python3 "$ROOT/L515_camera/view_l515_opencv.py" "${TOPIC}"
