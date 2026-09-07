#!/usr/bin/env bash
# Reuse the already-isolated perception interpreter without starting its live worker.
set -euo pipefail
BUNDLESDF_PROJECT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUNDLESDF_SAM2_PYTHON="${GROUNDED_SAM2_PYTHON:-$BUNDLESDF_PROJECT/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python}"
test -x "$BUNDLESDF_SAM2_PYTHON" || { echo 'Isolated SAM2 Python missing; set GROUNDED_SAM2_PYTHON to the existing Grounded-SAM2 environment.' >&2; exit 2; }
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$BUNDLESDF_SAM2_PYTHON" "$@"
