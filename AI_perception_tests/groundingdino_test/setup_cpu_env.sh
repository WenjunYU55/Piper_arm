#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON310:-python3.10}"
ENV_DIR="${GROUNDED_SAM2_ENV_DIR:-$SCRIPT_DIR/envs/grounded_sam2_py310}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required. Install an isolated Python 3.10 interpreter first." >&2
  echo "Do not install GroundingDINO/SAM2 into ROS Foxy's Python 3.8 environment." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
SAM2_BUILD_CUDA=0 "$ENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements_ai.txt"
"$SCRIPT_DIR/fetch_ai_assets.sh"
GROUNDED_SAM2_PYTHON="$ENV_DIR/bin/python" "$SCRIPT_DIR/check_env.sh"

echo "CPU AI environment ready: $ENV_DIR"
