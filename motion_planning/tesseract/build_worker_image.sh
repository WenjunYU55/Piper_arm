#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGE="${PIPER_TESSERACT_IMAGE:-localhost/piper-tesseract:0.35.0.6-dev}"

if ! command -v podman >/dev/null 2>&1; then
  echo "Podman is required. Install it interactively, then rerun this command." >&2
  exit 2
fi

podman build --pull=never --tag "$IMAGE" --file "$ROOT/motion_planning/tesseract/Dockerfile" "$ROOT"
podman image inspect "$IMAGE" --format '{{.Digest}} {{.Id}}'
