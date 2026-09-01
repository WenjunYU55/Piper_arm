#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/motion_planning/tesseract/rootless_common.sh"

exec "${ROOTLESS_BWRAP[@]}" \
  /opt/tesseract/bin/python \
  /workspace/motion_planning/tesseract/validate_benchmark_responses.py \
  --input /spool/validation_input.json \
  --output /spool/validation_output.json
