#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="${PIPER_ARM_ROOT:-$(cd .. && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

python3 "$ROOT/L515_camera/view_sam2_overlay.py"
