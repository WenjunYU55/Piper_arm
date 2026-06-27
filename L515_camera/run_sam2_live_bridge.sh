#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPOOL_DIR="${PIPER_SAM2_LIVE_SPOOL:-/tmp/piper_sam2_live}"

# shellcheck disable=SC1091
source "$ROOT/L515_camera/source_l515_environment.sh"

echo "Starting read-only SAM2 live frame bridge."
echo "Spool directory: $SPOOL_DIR"
exec ros2 run piper_mobile_manipulation sam2_live_bridge_node.py --ros-args \
  -p spool_dir:="$SPOOL_DIR" \
  -p frame_rate_hz:="${PIPER_SAM2_FPS:-10.0}" \
  -p auto_initial_mask:="${PIPER_SAM2_USE_HEAVY_INITIALIZER:-true}" \
  -p allow_fallback_seed:="${PIPER_SAM2_ALLOW_FALLBACK_SEED:-false}"
