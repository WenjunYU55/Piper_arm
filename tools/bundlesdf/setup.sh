#!/usr/bin/env bash
# Keep the complete upstream Python/CUDA toolchain outside the ROS workspace.
set -euo pipefail
BUNDLESDF_CHECKOUT="${1:-$HOME/.local/share/piper-bundlesdf/upstream}"
BUNDLESDF_PIN=ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1
if [ -e "$BUNDLESDF_CHECKOUT" ]; then
  test "$(git -C "$BUNDLESDF_CHECKOUT" rev-parse HEAD)" = "$BUNDLESDF_PIN" || { echo 'Existing checkout does not match pin; choose a new location.' >&2; exit 2; }
  test -z "$(git -C "$BUNDLESDF_CHECKOUT" status --porcelain)" || { echo 'Existing checkout is modified; choose a new location.' >&2; exit 2; }
else
  git clone https://github.com/NVlabs/BundleSDF.git "$BUNDLESDF_CHECKOUT"
  git -C "$BUNDLESDF_CHECKOUT" checkout --detach "$BUNDLESDF_PIN"
fi
printf 'Pinned BundleSDF at %s\nLicense: %s/LICENSE.txt (research/evaluation only)\n' "$BUNDLESDF_PIN" "$BUNDLESDF_CHECKOUT"
command -v docker >/dev/null || { echo 'Docker is missing. Install Docker with NVIDIA Container Toolkit, then rerun this script. No ROS Python packages were modified.' >&2; exit 3; }
docker info >/dev/null || { echo 'Docker daemon unavailable; configure Docker access, then rerun.' >&2; exit 3; }
docker build --network host -t "piper-bundlesdf:$BUNDLESDF_PIN" -f "$BUNDLESDF_CHECKOUT/docker/dockerfile" "$BUNDLESDF_CHECKOUT/docker"
# Retain editable-install metadata inside the final image. Native build outputs
# stay in this external checkout and never enter the ROS tree or Git.
BUNDLESDF_BUILD_CONTAINER="$(docker create --gpus all -v "$BUNDLESDF_CHECKOUT:$BUNDLESDF_CHECKOUT" -w "$BUNDLESDF_CHECKOUT" "piper-bundlesdf:$BUNDLESDF_PIN" bash -e build.sh)"
trap 'docker rm -f "$BUNDLESDF_BUILD_CONTAINER" >/dev/null 2>&1 || true' EXIT
docker start -a "$BUNDLESDF_BUILD_CONTAINER"
test "$(docker inspect -f '{{.State.ExitCode}}' "$BUNDLESDF_BUILD_CONTAINER")" = 0
docker commit "$BUNDLESDF_BUILD_CONTAINER" "piper-bundlesdf:$BUNDLESDF_PIN"
