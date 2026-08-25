#!/usr/bin/env bash
set -euo pipefail

"$(dirname "${BASH_SOURCE[0]}")/smoke_rootless_worker.sh"
source "$(dirname "${BASH_SOURCE[0]}")/rootless_common.sh"

"${ROOTLESS_BWRAP[@]}" /opt/tesseract/bin/python \
  -m piper_tesseract_foxy.qualification \
  --urdf /workspace/motion_planning/tesseract/.runtime/piper_planning.urdf \
  --srdf "$COLLISION_SRDF_CONTAINER" \
  --manifest "$COLLISION_MANIFEST_CONTAINER" \
  --calibration /workspace/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml \
  --home-pose /workspace/piper_home_pose.json \
  --suite core

exec "${ROOTLESS_BWRAP[@]}" /opt/tesseract/bin/python \
  -m piper_tesseract_foxy.qualification \
  --urdf /workspace/motion_planning/tesseract/.runtime/piper_planning.urdf \
  --srdf "$COLLISION_SRDF_CONTAINER" \
  --manifest "$COLLISION_MANIFEST_CONTAINER" \
  --calibration /workspace/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml \
  --home-pose /workspace/piper_home_pose.json \
  --suite compact
