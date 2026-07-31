#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAND_EYE_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
source "$SCRIPT_DIR/source_l515_environment.sh"
export ROS_DOMAIN_ID="$HAND_EYE_ROS_DOMAIN_ID"

# Use the same local UDP-only Fast DDS participant policy as the native GUI.
# Foxy shared-memory discovery can retain graph endpoints while failing to
# deliver the RealSense parameter reply and latched /tf_static sample to a
# process started later.  ROS_LOCALHOST_ONLY must be 0 so Fast DDS retains the
# XML participant transports; the profile itself permits loopback only.
HAND_EYE_FASTDDS_PROFILE="${PIPER_HAND_EYE_FASTDDS_PROFILE:-$SCRIPT_DIR/../fastdds_gui_udp_only.xml}"
if [ ! -f "$HAND_EYE_FASTDDS_PROFILE" ]; then
  echo "Missing hand-eye Fast DDS profile: $HAND_EYE_FASTDDS_PROFILE" >&2
  exit 1
fi
export FASTRTPS_DEFAULT_PROFILES_FILE="$HAND_EYE_FASTDDS_PROFILE"
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
export ROS_LOCALHOST_ONLY=0

exec python3 "$SCRIPT_DIR/publish_hand_eye_tf.py" \
  --calibration "${PIPER_HAND_EYE_CALIBRATION:-$SCRIPT_DIR/calibration/hand_eye/session_20260701_local/calibration_result.yaml}" \
  --joint-topic "${PIPER_HAND_EYE_JOINT_TOPIC:-/joint_states_single}" \
  --base-frame "${PIPER_HAND_EYE_BASE_FRAME:-base_link}" \
  --camera-frame "${PIPER_HAND_EYE_CAMERA_FRAME:-camera_link}" \
  --calibration-frame "${PIPER_HAND_EYE_CALIBRATION_FRAME:-camera_color_optical_frame}"
