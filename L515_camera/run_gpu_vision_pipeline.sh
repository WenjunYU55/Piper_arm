#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export PIPER_HEAVY_DEVICE=cuda
export PIPER_SAM2_DEVICE=cuda

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

start_process camera "$ROOT/L515_camera/start_l515_camera.sh"
sleep "${PIPER_CAMERA_STARTUP_SEC:-7}"
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
