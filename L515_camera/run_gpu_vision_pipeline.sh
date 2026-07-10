#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export PIPER_HEAVY_DEVICE=cuda
export PIPER_SAM2_DEVICE=cuda
HEAVY_SPOOL="${PIPER_HEAVY_REFRESH_SPOOL:-/tmp/piper_heavy_refresh}"
SAM2_SPOOL="${PIPER_SAM2_LIVE_SPOOL:-/tmp/piper_sam2_live}"

declare -a PIDS=()

start_process() {
  local name=$1
  shift
  echo "Starting $name..."
  setsid "$@" &
  PIDS+=("$!")
}

shutdown() {
  trap - INT TERM EXIT
  echo "Stopping GPU vision pipeline..."
  for pid in "${PIDS[@]}"; do
    kill -INT -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  wait || true
}
trap shutdown INT TERM EXIT

clear_live_spool() {
  if [ "${PIPER_CLEAR_VISION_SPOOL:-1}" != "1" ]; then
    echo "Reusing existing vision spool directories."
    return
  fi
  case "$HEAVY_SPOOL" in
    /tmp/piper_heavy_refresh*) ;;
    *)
      echo "Refusing to clear unexpected heavy spool path: $HEAVY_SPOOL" >&2
      exit 1
      ;;
  esac
  case "$SAM2_SPOOL" in
    /tmp/piper_sam2_live*) ;;
    *)
      echo "Refusing to clear unexpected SAM2 spool path: $SAM2_SPOOL" >&2
      exit 1
      ;;
  esac
  echo "Clearing live vision spool state."
  for dir in \
    "$HEAVY_SPOOL/requests" \
    "$HEAVY_SPOOL/processing" \
    "$HEAVY_SPOOL/responses" \
    "$SAM2_SPOOL/frames" \
    "$SAM2_SPOOL/seeds" \
    "$SAM2_SPOOL/results"; do
    mkdir -p "$dir"
    find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  done
}

clear_live_spool

if [ "${PIPER_REUSE_EXISTING_CAMERA:-0}" = "1" ]; then
  echo "Reusing existing L515 camera; not starting a RealSense camera process."
else
  if [ "${PIPER_CAMERA_LOW_BANDWIDTH:-0}" = "1" ]; then
    echo "Using reduced-bandwidth L515 depth profile."
    start_process camera "$ROOT/L515_camera/start_l515_camera_low_bandwidth.sh"
  else
    start_process camera "$ROOT/L515_camera/start_l515_camera.sh"
  fi
  sleep "${PIPER_CAMERA_STARTUP_SEC:-7}"
fi
start_process heavy_bridge "$ROOT/L515_camera/run_heavy_refresh_bridge.sh"
start_process heavy_cuda_worker "$ROOT/L515_camera/run_heavy_model_worker.sh"
start_process sam2_cuda_worker "$ROOT/L515_camera/run_sam2_live_worker.sh"
start_process target_cloud "$ROOT/L515_camera/run_target_cloud.sh"
sleep 2
# Start last so its one-shot initialization request has active subscribers.
start_process sam2_bridge "$ROOT/L515_camera/run_sam2_live_bridge.sh"
sleep 2
start_process gpu_geometry "$ROOT/L515_camera/run_gpu_geometry.sh"

echo "GPU vision pipeline is running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID."
echo "Target: /piper/sam2_target_mask"
echo "Obstacles: /piper/sam2_obstacle_mask"
echo "Obstacle instances: /piper/obstacle_instances_3d"
echo "Cloud: /piper/target_cloud"
echo "Press Ctrl+C to stop all vision processes."

while true; do
  for index in "${!PIDS[@]}"; do
    pid=${PIDS[$index]}
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      echo "A vision process exited unexpectedly (pid=$pid status=$status)." >&2
      exit 1
    fi
  done
  sleep 1
done
