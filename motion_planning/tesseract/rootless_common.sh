#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNTIME="${PIPER_TESSERACT_RUNTIME:-$ROOT/motion_planning/tesseract/.runtime}"
ROOTFS="$RUNTIME/rootfs"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
SPOOL="${PIPER_TESSERACT_SPOOL:-$RUNTIME_ROOT/piper_tesseract_plans}"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/floor_profile.sh"
COLLISION_MANIFEST_HOST="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_MANIFEST_NAME"
COLLISION_SRDF_HOST="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_SRDF_NAME"
COLLISION_MANIFEST_CONTAINER="/workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_MANIFEST_NAME"
COLLISION_SRDF_CONTAINER="/workspace/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_SRDF_NAME"

if [[ ! -x "$ROOTFS/opt/tesseract/bin/python" ]]; then
  echo "Rootless worker is not installed. Run setup_rootless_worker.sh first." >&2
  exit 2
fi

mkdir -p "$SPOOL"/{requests,processing,responses,failed}
chmod 700 "$SPOOL" "$SPOOL"/{requests,processing,responses,failed}
mkdir -p "$ROOTFS/workspace" "$ROOTFS/spool" "$ROOTFS/home/planner"

ROOTLESS_BWRAP=(
  bwrap --unshare-user --uid 1000 --gid 1000 --unshare-pid --unshare-net
  --die-with-parent --new-session
  --ro-bind "$ROOTFS" /
  --ro-bind "$ROOT" /workspace
  --bind "$SPOOL" /spool
  --proc /proc --dev /dev
  --tmpfs /tmp
  --setenv HOME /home/planner
  --setenv LANG C.UTF-8
  --unsetenv AMENT_PREFIX_PATH
  --unsetenv COLCON_PREFIX_PATH
  --unsetenv ROS_PACKAGE_PATH
  --unsetenv ROS_DISTRO
  --unsetenv ROS_VERSION
  --unsetenv ROS_PYTHON_VERSION
  --setenv PYTHONPATH /workspace/piper_ros_foxy/src/piper_tesseract_foxy:/workspace/piper_ros_foxy/src/piper_mobile_manipulation
  --setenv TESSERACT_RESOURCE_PATH /workspace/piper_ros_foxy/src
  --setenv PIPER_TESSERACT_URDF /workspace/motion_planning/tesseract/.runtime/piper_planning.urdf
  --setenv PIPER_FLOOR_PROFILE "$FLOOR_PROFILE"
  --setenv PIPER_TESSERACT_SRDF "$COLLISION_SRDF_CONTAINER"
  --setenv PIPER_TESSERACT_COLLISION_MANIFEST "$COLLISION_MANIFEST_CONTAINER"
  --setenv PIPER_TESSERACT_SPOOL /spool
)
