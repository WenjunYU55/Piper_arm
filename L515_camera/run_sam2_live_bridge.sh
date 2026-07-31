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
  -p auto_initial_mask:="${PIPER_SAM2_USE_HEAVY_INITIALIZER:-false}" \
  -p arm_moving_threshold_rad_s:="${PIPER_ARM_MOVING_THRESHOLD_RAD_S:-0.08}" \
  -p arm_settled_threshold_rad_s:="${PIPER_ARM_SETTLED_THRESHOLD_RAD_S:-0.03}" \
  -p arm_motion_window_sec:="${PIPER_ARM_MOTION_WINDOW_SEC:-0.75}" \
  -p arm_moving_position_delta_rad:="${PIPER_ARM_MOVING_POSITION_DELTA_RAD:-0.012}" \
  -p arm_settled_position_delta_rad:="${PIPER_ARM_SETTLED_POSITION_DELTA_RAD:-0.009}" \
  -p camera_settle_time_sec:="${PIPER_CAMERA_SETTLE_TIME_SEC:-0.5}" \
  -p camera_timestamp_health_topic:="/piper/camera_timestamp_health" \
  -p max_reacquisition_attempts:="${PIPER_MAX_REACQUISITION_ATTEMPTS:-2}" \
  -p absent_retry_sec:="${PIPER_ABSENT_RETRY_SEC:-30.0}" \
  -p recovery_valid_frames:="${PIPER_RECOVERY_VALID_FRAMES:-5}" \
  -p low_confidence_refresh_threshold:="${PIPER_LOW_CONFIDENCE_REFRESH_THRESHOLD:-0.60}" \
  -p low_confidence_refresh_duration_sec:="${PIPER_LOW_CONFIDENCE_REFRESH_DURATION_SEC:-1.0}" \
  -p low_confidence_refresh_hysteresis:="${PIPER_LOW_CONFIDENCE_REFRESH_HYSTERESIS:-0.10}" \
  -p tracking_measurement_stale_sec:="${PIPER_TRACKING_MEASUREMENT_STALE_SEC:-0.75}"
