#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AI_DIR="$ROOT/AI_perception_tests/groundingdino_test"
PYTHON_BIN="${GROUNDED_SAM2_PYTHON:-$AI_DIR/envs/grounded_sam2_py310/bin/python}"
SPOOL_DIR="${PIPER_HEAVY_REFRESH_SPOOL:-/tmp/piper_heavy_refresh}"
DEVICE="${PIPER_HEAVY_DEVICE:-cpu}"

export HF_HOME="${HF_HOME:-$AI_DIR/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/piper_grounded_sam2_mpl}"

echo "Starting isolated heavy-model worker."
echo "Python: ${PYTHON_BIN}"
echo "Device: ${DEVICE}"
echo "Spool directory: ${SPOOL_DIR}"
echo "Hugging Face model resolution: local cache only"
echo "This process has no ROS imports and cannot command arm motion."

exec "$PYTHON_BIN" "$ROOT/AI_perception_tests/heavy_model_worker.py" \
  --spool-dir "$SPOOL_DIR" --device "$DEVICE" \
  --task-id "${PIPER_MISSION_TASK_ID:-}" \
  --mission-sha256 "${PIPER_MISSION_SHA256:-}" \
  --target-label "${PIPER_TARGET_LABEL:-green cube}" \
  --target-profile "${PIPER_TARGET_PROFILE:-green_cube}" \
  --target-prompt "${PIPER_TARGET_PROMPT:-green cube .}"
