#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$ROOT/reconstruction/.venv"
BASE_PYTHON="${PIPER_RECONSTRUCTION_BASE_PYTHON:-$ROOT/AI_perception_tests/groundingdino_test/envs/python310_base/bin/python3.10}"

if [ ! -x "$BASE_PYTHON" ]; then
  BASE_PYTHON="$(command -v python3.10 || true)"
fi
if [ -z "$BASE_PYTHON" ] || [ ! -x "$BASE_PYTHON" ]; then
  echo "Python 3.10 is required for the isolated Open3D environment." >&2
  exit 2
fi

"$BASE_PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/reconstruction/requirements.txt"
"$VENV/bin/python" -c 'import cv2, numpy, open3d, yaml; print("reconstruction environment ready")'
