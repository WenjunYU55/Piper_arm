#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/rootless_common.sh"

PYTHONPATH="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy" \
  python3 -m piper_tesseract_foxy.model_builder \
  --xacro "$ROOT/piper_ros_foxy/src/piper_description/urdf/piper_description.xacro" \
  --calibration "$ROOT/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml" \
  --manifest "$COLLISION_MANIFEST_HOST" \
  --output "$RUNTIME/piper_planning.urdf"

"${ROOTLESS_BWRAP[@]}" /opt/tesseract/bin/python -c \
  "import tesseract_robotics; from tesseract_robotics.planning import Robot; print('tesseract import: OK', getattr(tesseract_robotics, '__version__', 'unknown')); Robot.from_files('/workspace/motion_planning/tesseract/.runtime/piper_planning.urdf', '$COLLISION_SRDF_CONTAINER'); print('PiPER model load: OK')"
