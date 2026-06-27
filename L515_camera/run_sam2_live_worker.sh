#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AI_DIR="$ROOT/AI_perception_tests/groundingdino_test"
PYTHON_BIN="${GROUNDED_SAM2_PYTHON:-$AI_DIR/envs/grounded_sam2_py310/bin/python}"
SPOOL_DIR="${PIPER_SAM2_LIVE_SPOOL:-/tmp/piper_sam2_live}"

echo "Starting isolated CUDA SAM2 live worker."
echo "Python: $PYTHON_BIN"
echo "Spool directory: $SPOOL_DIR"
exec "$PYTHON_BIN" "$ROOT/AI_perception_tests/sam2_live_worker.py" \
  --spool-dir "$SPOOL_DIR" \
  --device "${PIPER_SAM2_DEVICE:-cuda}" \
  --max-session-frames "${PIPER_SAM2_MAX_SESSION_FRAMES:-8}"
