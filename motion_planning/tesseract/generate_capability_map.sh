#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNTIME="${PIPER_TESSERACT_RUNTIME:-$ROOT/motion_planning/tesseract/.runtime}"
ROOTFS="$RUNTIME/rootfs"
OUTPUT_DIR="${PIPER_CAPABILITY_OUTPUT_DIR:-$RUNTIME/capability_map}"
WORKERS="${PIPER_CAPABILITY_WORKERS:-8}"
CHECKPOINTS="${PIPER_CAPABILITY_CHECKPOINTS:-100000 250000 500000 1000000 2000000}"

if [[ ! -x "$ROOTFS/opt/tesseract/bin/python" ]]; then
  echo "Rootless Tesseract is not installed. Run setup_rootless_worker.sh first." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$RUNTIME"
chmod 700 "$OUTPUT_DIR"

TABLETOP_MANIFEST="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml"
GROUND_MANIFEST="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model_ground.yaml"
SRDF="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/piper_bunker.srdf"
XACRO="$ROOT/piper_ros_foxy/src/piper_description/urdf/piper_description.xacro"
CALIBRATION="$ROOT/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml"
JOINT_BOUNDS="$ROOT/piper_joint_bounds.json"
URDF="$RUNTIME/piper_capability_map.urdf"

PYTHONPATH="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy:$ROOT/piper_ros_foxy/src/piper_mobile_manipulation" \
  python3 -m piper_tesseract_foxy.model_builder \
  --xacro "$XACRO" \
  --calibration "$CALIBRATION" \
  --manifest "$TABLETOP_MANIFEST" \
  --output "$URDF"

bwrap --unshare-user --uid 1000 --gid 1000 --unshare-pid --unshare-net \
  --die-with-parent --new-session \
  --ro-bind "$ROOTFS" / \
  --ro-bind "$ROOT" /workspace \
  --proc /proc --dev /dev \
  --tmpfs /tmp \
  --bind "$OUTPUT_DIR" /tmp/output \
  --ro-bind "$URDF" /tmp/piper_capability_map.urdf \
  --setenv HOME /home/planner \
  --setenv LANG C.UTF-8 \
  --unsetenv AMENT_PREFIX_PATH \
  --unsetenv COLCON_PREFIX_PATH \
  --unsetenv ROS_PACKAGE_PATH \
  --unsetenv ROS_DISTRO \
  --unsetenv ROS_VERSION \
  --unsetenv ROS_PYTHON_VERSION \
  --setenv PYTHONPATH /workspace/piper_ros_foxy/src/piper_tesseract_foxy:/workspace/piper_ros_foxy/src/piper_mobile_manipulation \
  --setenv TESSERACT_RESOURCE_PATH /workspace/piper_ros_foxy/src \
  /opt/tesseract/bin/python -m piper_tesseract_foxy.capability_map_generator \
  --project-root /workspace \
  --urdf /tmp/piper_capability_map.urdf \
  --srdf /workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/piper_bunker.srdf \
  --manifest /workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml \
  --joint-bounds /workspace/piper_joint_bounds.json \
  --output-dir /tmp/output \
  --workers "$WORKERS" \
  --checkpoints $CHECKPOINTS \
  --source /workspace/piper_ros_foxy/src/piper_description/urdf/piper_description.xacro \
  --source /workspace/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml \
  --source /workspace/piper_joint_bounds.json \
  --source /workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/piper_bunker.srdf \
  --source /workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml \
  --source /workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model_ground.yaml \
  --source /workspace/piper_ros_foxy/src/piper_tesseract_foxy/piper_tesseract_foxy/model_builder.py \
  --source /workspace/piper_ros_foxy/src/piper_tesseract_foxy/piper_tesseract_foxy/worker.py
