#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PIPER_TESSERACT_IMAGE:-localhost/piper-tesseract:0.35.0.6-dev}"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
SPOOL="${PIPER_TESSERACT_SPOOL:-$RUNTIME_ROOT/piper_tesseract_plans}"
LOCK_FILE="$RUNTIME_ROOT/piper_tesseract_worker.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A Tesseract worker already owns $LOCK_FILE; refusing a duplicate worker." >&2
  exit 3
fi

if ! command -v podman >/dev/null 2>&1; then
  source "$(dirname "${BASH_SOURCE[0]}")/rootless_common.sh"
  PYTHONPATH="$ROOT/piper_ros_foxy/src/piper_tesseract_foxy" \
    python3 -m piper_tesseract_foxy.model_builder \
    --xacro "$ROOT/piper_ros_foxy/src/piper_description/urdf/piper_description.xacro" \
    --calibration "$ROOT/L515_camera/calibration/hand_eye/session_20260701_local/calibration_result.yaml" \
    --manifest "$ROOT/piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml" \
    --output "$RUNTIME/piper_planning.urdf"
  # Keep this shell alive as Bubblewrap's direct parent.  The GUI launches this
  # wrapper from a short-lived Python startup thread; exec'ing bwrap there made
  # --die-with-parent observe that thread's exit and kill the worker with
  # SIGKILL even though the GUI process remained alive.
  "${ROOTLESS_BWRAP[@]}" \
    /opt/tesseract/bin/python -m piper_tesseract_foxy.worker
fi
mkdir -p "$SPOOL"/{requests,processing,responses,failed}
chmod 700 "$SPOOL" "$SPOOL"/{requests,processing,responses,failed}

exec podman run --rm \
  --name piper-tesseract-worker \
  --network none \
  --read-only \
  --cap-drop all \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --cpus 4 \
  --memory 4g \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --volume "$SPOOL:/spool:rw" \
  "$IMAGE"
