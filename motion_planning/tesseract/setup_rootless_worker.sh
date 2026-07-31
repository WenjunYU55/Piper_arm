#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNTIME="${PIPER_TESSERACT_RUNTIME:-$ROOT/motion_planning/tesseract/.runtime}"
ROOTFS="$RUNTIME/rootfs"
CACHE="$RUNTIME/cache"
ROOTFS_URL="https://partner-images.canonical.com/oci/noble/20260719/ubuntu-noble-oci-amd64-root.tar.gz"
ROOTFS_SHA256="5e63dd4d2c0c774251d87b5df27da4cdacca47545f2d46fb34c7741d221ee62a"
ARCHIVE="$CACHE/ubuntu-noble-oci-amd64-root.tar.gz"
MICROMAMBA="${PIPER_MICROMAMBA:-$(command -v micromamba || true)}"

if [[ -z "$MICROMAMBA" || ! -x "$MICROMAMBA" ]]; then
  echo "micromamba is required (set PIPER_MICROMAMBA if it is not in PATH)." >&2
  exit 2
fi
if ! command -v bwrap >/dev/null 2>&1; then
  echo "bubblewrap is required." >&2
  exit 2
fi

mkdir -p "$CACHE" "$RUNTIME"
if [[ ! -f "$ARCHIVE" ]]; then
  curl --fail --location --output "$ARCHIVE.part" "$ROOTFS_URL"
  mv "$ARCHIVE.part" "$ARCHIVE"
fi
printf '%s  %s\n' "$ROOTFS_SHA256" "$ARCHIVE" | sha256sum --check

if [[ ! -x "$ROOTFS/bin/bash" ]]; then
  mkdir -p "$ROOTFS"
  tar --extract --gzip --file "$ARCHIVE" --directory "$ROOTFS" --no-same-owner
fi

mkdir -p "$ROOTFS/opt/tesseract" "$ROOTFS/opt/micromamba-root" "$ROOTFS/tmp"

# Setup deliberately shares the network only while resolving the exact pinned
# userspace. The runtime launcher below always creates a new network namespace.
if [[ ! -d "$ROOTFS/opt/tesseract/lib/python3.10/site-packages/tesseract_robotics" ]]; then
  bwrap \
    --unshare-user --uid 0 --gid 0 --unshare-pid --die-with-parent \
    --bind "$ROOTFS" / \
    --ro-bind "$MICROMAMBA" /usr/local/bin/micromamba \
    --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --ro-bind /etc/ssl/certs /etc/ssl/certs \
    --proc /proc --dev /dev \
    --setenv HOME /root \
    --setenv MAMBA_ROOT_PREFIX /opt/micromamba-root \
    --setenv LANG C.UTF-8 \
    /usr/local/bin/micromamba create --yes --prefix /opt/tesseract \
      --channel conda-forge --strict-channel-priority \
      python=3.10.20 pip numpy=2.2.6 scipy=1.15.2 pyyaml=6.0.2

  bwrap \
    --unshare-user --uid 0 --gid 0 --unshare-pid --die-with-parent \
    --bind "$ROOTFS" / \
    --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --ro-bind /etc/ssl/certs /etc/ssl/certs \
    --proc /proc --dev /dev \
    --setenv HOME /root --setenv LANG C.UTF-8 \
    /opt/tesseract/bin/pip install --no-cache-dir \
      tesseract-robotics-nanobind==0.35.0.6
fi

MAMBA_ROOT_PREFIX="$ROOTFS/opt/micromamba-root" \
  "$MICROMAMBA" list --prefix "$ROOTFS/opt/tesseract" --json \
  > "$RUNTIME/conda-packages.json"
bwrap \
  --unshare-user --uid 0 --gid 0 --unshare-pid --unshare-net --die-with-parent \
  --ro-bind "$ROOTFS" / --proc /proc --dev /dev \
  --setenv HOME /root --setenv LANG C.UTF-8 \
  /opt/tesseract/bin/python -m pip freeze --all \
  > "$RUNTIME/python-requirements.lock"

{
  printf 'rootfs_url=%s\n' "$ROOTFS_URL"
  printf 'rootfs_sha256=%s\n' "$ROOTFS_SHA256"
  printf 'tesseract_binding=0.35.0.6\n'
  printf 'conda_packages_sha256=%s\n' "$(sha256sum "$RUNTIME/conda-packages.json" | cut -d' ' -f1)"
  printf 'python_requirements_sha256=%s\n' "$(sha256sum "$RUNTIME/python-requirements.lock" | cut -d' ' -f1)"
  "$ROOTFS/opt/tesseract/bin/python" --version 2>&1 || true
} > "$RUNTIME/runtime.lock"

echo "Rootless Tesseract userspace prepared at $ROOTFS"
echo "Run motion_planning/tesseract/smoke_rootless_worker.sh next."
