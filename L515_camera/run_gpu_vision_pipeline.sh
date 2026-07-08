#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export PIPER_HEAVY_DEVICE=cuda
export PIPER_SAM2_DEVICE=cuda

# Needed here so the launcher can discover an already-running ROS camera.
# Individual component launchers also source this file for standalone use.
source "$ROOT/L515_camera/source_l515_environment.sh"

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

topic_has_publisher() {
  local topic=$1
  local count
  count=$(ros2 topic info "$topic" 2>/dev/null | awk '/Publisher count:/ {print $3}')
  [ "${count:-0}" -gt 0 ] 2>/dev/null
}

topic_is_active() {
  local topic=$1
  local activity
  activity=$(timeout "${PIPER_CAMERA_PROBE_SEC:-3}s" \
    ros2 topic hz "$topic" 2>/dev/null || true)
  [ -n "$activity" ]
}

camera_is_available() {
  topic_is_active /camera/color/image_raw &&
    topic_has_publisher /camera/aligned_depth_to_color/image_raw &&
    topic_has_publisher /camera/color/camera_info
}

if [ "${PIPER_REUSE_EXISTING_CAMERA:-1}" = "1" ] && camera_is_available; then
  echo "Reusing existing ROS L515 RGB-D streams; no second camera process will be started."
  echo "The existing camera is externally managed and will not be stopped by this launcher."
  sleep "${PIPER_EXTERNAL_CAMERA_SETTLE_SEC:-1}"
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
