#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="$ROOT/L515_camera/realsense_ws/install/librealsense2"
SOURCE="$ROOT/L515_camera/realsense_ws/src/realsense-ros"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
# The production patch must be installed, not a test-only alternate implementation.
git -C "$SOURCE" apply --reverse --check "$ROOT/L515_camera/patches/realsense-ros-4.0.4-aligned-depth-guard.patch"
g++ -std=c++11 "$ROOT/L515_camera/test_aligned_depth_guard.cpp" \
  -I "$SOURCE/realsense2_camera/include" -I "$SDK/include" \
  -L "$SDK/lib" -Wl,-rpath,"$SDK/lib" -lrealsense2 -o "$BUILD/test_guard"
"$BUILD/test_guard"
