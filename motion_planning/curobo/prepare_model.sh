#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CUROBO_PYTHON="${PIPER_CUROBO_PYTHON:-}"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
OUTPUT_ROOT="${PIPER_CUROBO_OUTPUT:-$RUNTIME_ROOT/piper_curobo_model}"
CALIBRATION="${PIPER_HAND_EYE_CALIBRATION:-$ROOT/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml}"
# shellcheck disable=SC1091
source "$ROOT/motion_planning/tesseract/floor_profile.sh"

if [ -z "$CUROBO_PYTHON" ] || [ ! -x "$CUROBO_PYTHON" ]; then
  echo "PIPER_CUROBO_PYTHON must name the explicit executable cuRobo interpreter." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
chmod 700 "$OUTPUT_ROOT"
URDF="$OUTPUT_ROOT/piper_planning.urdf"
CONFIG="$OUTPUT_ROOT/piper_curobo.yml"
DESCRIPTION_ROOT="$ROOT/piper_ros_foxy/src/piper_description"
MANIFEST="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_MANIFEST_NAME"
SRDF="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_SRDF_NAME"

PYTHONPATH="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy" \
  /usr/bin/python3 -m piper_tesseract_foxy.model_builder \
  --xacro "$DESCRIPTION_ROOT/urdf/piper_description.xacro" \
  --calibration "$CALIBRATION" \
  --manifest "$MANIFEST" \
  --output "$URDF"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$CUROBO_PYTHON" -m motion_planning.curobo.generate_robot_config \
  --urdf "$URDF" \
  --srdf "$SRDF" \
  --collision-manifest "$MANIFEST" \
  --description-root "$DESCRIPTION_ROOT" \
  --output "$CONFIG"

chmod 600 "$URDF" "$CONFIG"
printf '%s\n' "$CONFIG"
