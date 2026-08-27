#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CUROBO_PYTHON="${PIPER_CUROBO_PYTHON:-}"
CUROBO_CUDA_HOME="${PIPER_CUROBO_CUDA_HOME:-}"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
SPOOL="${PIPER_CUROBO_SPOOL:-$RUNTIME_ROOT/piper_curobo_plans}"
LOCK_FILE="$RUNTIME_ROOT/piper_curobo_worker.lock"
MODEL_ROOT="$RUNTIME_ROOT/piper_curobo_model"
ROBOT_CONFIG="${PIPER_CUROBO_ROBOT_CONFIG:-$MODEL_ROOT/piper_curobo.yml}"
# shellcheck disable=SC1091
source "$ROOT/motion_planning/tesseract/floor_profile.sh"

if [ -z "$CUROBO_PYTHON" ] || [ ! -x "$CUROBO_PYTHON" ]; then
  echo "PIPER_CUROBO_PYTHON must name the explicit executable cuRobo interpreter." >&2
  exit 2
fi

if [ -n "$CUROBO_CUDA_HOME" ]; then
  if [ ! -x "$CUROBO_CUDA_HOME/bin/nvcc" ]; then
    echo "PIPER_CUROBO_CUDA_HOME does not contain an executable nvcc." >&2
    exit 2
  fi
  export CUDA_HOME="$CUROBO_CUDA_HOME"
  export PATH="$CUROBO_CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUROBO_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ -z "${PIPER_CUROBO_ROBOT_CONFIG:-}" ]; then
  PIPER_CUROBO_OUTPUT="$MODEL_ROOT" \
    "$ROOT/motion_planning/curobo/prepare_model.sh"
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A cuRobo worker already owns $LOCK_FILE; refusing a duplicate worker." >&2
  exit 3
fi

mkdir -p "$SPOOL"/{requests,processing,responses,failed}
chmod 700 "$SPOOL" "$SPOOL"/{requests,processing,responses,failed}

export PIPER_CUROBO_SPOOL="$SPOOL"
export PIPER_CUROBO_ROBOT_CONFIG="$ROBOT_CONFIG"
export PIPER_CUROBO_SRDF="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_SRDF_NAME"
export PIPER_CUROBO_COLLISION_MANIFEST="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/$COLLISION_MANIFEST_NAME"
if [ "$FLOOR_PROFILE" = "ground" ]; then
  export PIPER_CUROBO_FLOOR_Z_M=-0.466
else
  export PIPER_CUROBO_FLOOR_Z_M=0.005
fi
# Deliberately replace, rather than extend, the login-shell PYTHONPATH.  This
# worker runs in Python 3.10 and must never inherit Foxy's Python 3.8 packages.
export PYTHONPATH="$ROOT:$ROOT/piper_ros_foxy/src/piper_tesseract_foxy"

exec "$CUROBO_PYTHON" -m motion_planning.curobo.worker
