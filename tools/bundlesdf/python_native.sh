#!/usr/bin/env bash
# Explicit interpreter boundary, matching the cuRobo worker pattern.
set -euo pipefail
BUNDLESDF_RUNTIME="${PIPER_BUNDLESDF_RUNTIME:-$HOME/.local/share/piper-bundlesdf}"
BUNDLESDF_PYTHON="${PIPER_BUNDLESDF_PYTHON:-$HOME/.venvs/piper-bundlesdf/bin/python}"
BUNDLESDF_CUDA="${PIPER_BUNDLESDF_CUDA_HOME:-$HOME/.local/cuda-12.8}"
test -x "$BUNDLESDF_PYTHON" || { echo 'BundleSDF environment missing; run tools/bundlesdf/setup_native.sh.' >&2; exit 2; }
export CUDA_HOME="$BUNDLESDF_CUDA"
export PATH="$BUNDLESDF_CUDA/bin:$(dirname "$BUNDLESDF_PYTHON"):$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
# Replace ROS environment paths, never append Foxy Python/native library paths.
unset PYTHONHOME
export PYTHONPATH="$BUNDLESDF_RUNTIME/upstream"
export LD_LIBRARY_PATH="$BUNDLESDF_RUNTIME/native/usr/lib:$BUNDLESDF_RUNTIME/native/usr/lib/x86_64-linux-gnu:$BUNDLESDF_CUDA/lib64"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec "$BUNDLESDF_PYTHON" "$@"
