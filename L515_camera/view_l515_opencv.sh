#!/usr/bin/env bash
set -euo pipefail

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

TOPIC=${1:-/camera/color/image_raw}

echo "Opening OpenCV viewer on ${TOPIC}"
python3 /home/prl/Piper_arm/L515_camera/view_l515_opencv.py "${TOPIC}"
