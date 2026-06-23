#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/Grounded-SAM-2"
REVISION=b7a9c29f196edff0eb54dbe14588d7ae5e3dde28

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch origin "$REVISION"
git -C "$REPO_DIR" checkout --detach "$REVISION"

mkdir -p "$SCRIPT_DIR/weights" "$SCRIPT_DIR/checkpoints"
curl --fail --location --continue-at - \
  --output "$SCRIPT_DIR/weights/groundingdino_swint_ogc.pth" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
curl --fail --location --continue-at - \
  --output "$SCRIPT_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

echo "Pinned model sources and required checkpoints are ready."
