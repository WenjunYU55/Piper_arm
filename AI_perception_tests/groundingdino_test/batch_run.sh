#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${GROUNDED_SAM2_PYTHON:-$SCRIPT_DIR/envs/grounded_sam2_py310/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/piper_grounded_sam2_mpl}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRIPT_DIR/hf_cache/transformers}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/batch_groundingdino.py" "$@"
