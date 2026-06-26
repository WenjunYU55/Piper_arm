#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT/L515_camera/realsense_ws"

# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

for PATCH_FILE in \
  "$ROOT/L515_camera/patches/realsense-ros-4.0.4-l515-foxy.patch" \
  "$ROOT/L515_camera/patches/realsense-ros-4.0.4-no-default-profile-fallback.patch"
do
  if [ -f "$PATCH_FILE" ]; then
    PATCH_NAME="$(basename "$PATCH_FILE")"
    if git -C src/realsense-ros apply --check "$PATCH_FILE" >/dev/null 2>&1; then
      echo "Applying $PATCH_NAME."
      git -C src/realsense-ros apply "$PATCH_FILE"
    else
      echo "$PATCH_NAME is already applied or not applicable."
    fi
  fi
done

colcon build \
  --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF
