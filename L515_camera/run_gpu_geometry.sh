#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

echo "Starting SAM2 mask geometry, RGB-D projection, tracking, and occlusion monitoring."
exec ros2 launch piper_mobile_manipulation gpu_geometry.launch.py
