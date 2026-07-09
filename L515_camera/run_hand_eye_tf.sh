#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAND_EYE_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
HAND_EYE_ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
source "$SCRIPT_DIR/source_l515_environment.sh"
export ROS_DOMAIN_ID="$HAND_EYE_ROS_DOMAIN_ID"
export ROS_LOCALHOST_ONLY="$HAND_EYE_ROS_LOCALHOST_ONLY"

exec python3 "$SCRIPT_DIR/publish_hand_eye_tf.py" \
  --calibration "${PIPER_HAND_EYE_CALIBRATION:-$SCRIPT_DIR/calibration/hand_eye/session_20260701_local/calibration_result.yaml}" \
  --joint-topic "${PIPER_HAND_EYE_JOINT_TOPIC:-/joint_states_single}" \
  --base-frame "${PIPER_HAND_EYE_BASE_FRAME:-base_link}" \
  --camera-frame "${PIPER_HAND_EYE_CAMERA_FRAME:-camera_link}" \
  --calibration-frame "${PIPER_HAND_EYE_CALIBRATION_FRAME:-camera_color_optical_frame}"
