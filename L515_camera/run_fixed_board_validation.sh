#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
VALIDATION_ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
source "$SCRIPT_DIR/source_l515_environment.sh"
export ROS_DOMAIN_ID="$VALIDATION_ROS_DOMAIN_ID"
export ROS_LOCALHOST_ONLY="$VALIDATION_ROS_LOCALHOST_ONLY"

exec python3 "$SCRIPT_DIR/validate_fixed_board.py" \
  --output "${PIPER_BOARD_VALIDATION_OUTPUT:-$SCRIPT_DIR/calibration/hand_eye/session_20260701_local/fixed_board_validation.yaml}" \
  "$@"
