#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT/L515_camera/realsense_ws"

# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

PATCH_FILE="$ROOT/L515_camera/patches/realsense-ros-4.0.4-l515-foxy.patch"
if [ -f "$PATCH_FILE" ]; then
  if git -C src/realsense-ros apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "Applying RealSense ROS L515 Foxy patch."
    git -C src/realsense-ros apply "$PATCH_FILE"
  else
    echo "RealSense ROS L515 Foxy patch is already applied or not applicable."
  fi
fi

colcon build \
  --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF
