#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export L515_ROS_LOCALHOST_ONLY="${L515_ROS_LOCALHOST_ONLY:-$ROS_LOCALHOST_ONLY}"
# Keep the camera, watchdog, CUDA workers, geometry nodes, GUI, and driver on
# the same Foxy participant policy.  Mixing the default shared-memory
# participant with the GUI's UDP-only participant can leave discovery visible
# while parameter/service samples fail deserialization.
VISION_FASTDDS_PROFILE="${PIPER_VISION_FASTDDS_PROFILE:-$ROOT/fastdds_gui_udp_only.xml}"
if [ ! -f "$VISION_FASTDDS_PROFILE" ]; then
  echo "Missing vision Fast DDS profile: $VISION_FASTDDS_PROFILE" >&2
  exit 1
fi
export FASTRTPS_DEFAULT_PROFILES_FILE="$VISION_FASTDDS_PROFILE"
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
export PIPER_HEAVY_DEVICE=cuda
export PIPER_SAM2_DEVICE=cuda
HEAVY_SPOOL="${PIPER_HEAVY_REFRESH_SPOOL:-/tmp/piper_heavy_refresh}"
SAM2_SPOOL="${PIPER_SAM2_LIVE_SPOOL:-/tmp/piper_sam2_live}"
RECOVERY_ROOT="${PIPER_VISION_RECOVERY_ROOT:-/tmp/piper_vision_recovery}"
RECOVERY_REQUEST="${PIPER_VISION_RECOVERY_REQUEST:-$RECOVERY_ROOT/request.yaml}"
PROCESS_MANIFEST="${PIPER_VISION_PROCESS_MANIFEST:-$RECOVERY_ROOT/process_groups.txt}"
RECOVERY_COUNT=0
GENERATION_STARTED=0

declare -a PIDS=()
declare -a PID_NAMES=()

write_process_manifest() {
  local temporary="${PROCESS_MANIFEST}.tmp.$$"
  : > "$temporary"
  local index
  for index in "${!PIDS[@]}"; do
    printf '%s %s\n' "${PID_NAMES[$index]}" "${PIDS[$index]}" >> "$temporary"
  done
  mv -f "$temporary" "$PROCESS_MANIFEST"
}

start_process() {
  local name=$1
  shift
  echo "Starting $name..."
  setsid "$@" &
  PIDS+=("$!")
  PID_NAMES+=("$name")
  write_process_manifest
}

stop_last_process() {
  local last_index=$(( ${#PIDS[@]} - 1 ))
  local pid="${PIDS[$last_index]}"
  kill -INT -- "-$pid" 2>/dev/null || true
  sleep 1
  kill -TERM -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  unset 'PIDS[last_index]' 'PID_NAMES[last_index]'
  PIDS=("${PIDS[@]}")
  PID_NAMES=("${PID_NAMES[@]}")
  write_process_manifest
}

watchdog_graph_ready() {
  local nodes topic_info
  nodes="$(timeout 3 ros2 node list 2>/dev/null || true)"
  topic_info="$(timeout 3 ros2 topic info /piper/camera_timestamp_health 2>/dev/null || true)"
  printf '%s\n' "$nodes" | grep -Fxq '/camera_timestamp_watchdog' &&
    printf '%s\n' "$topic_info" | grep -Fq 'Publisher count: 1'
}

start_required_watchdog() {
  local attempt check
  for attempt in 1 2 3; do
    start_process camera_timestamp_watchdog \
      "$ROOT/L515_camera/run_camera_timestamp_watchdog.sh"
    for check in 1 2 3 4 5; do
      if watchdog_graph_ready; then
        echo "Camera timestamp watchdog is present and publishing health."
        return 0
      fi
      if ! kill -0 "${PIDS[${#PIDS[@]} - 1]}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    echo "Camera timestamp watchdog did not become ready (attempt $attempt/3)." >&2
    stop_last_process
    sleep 2
  done
  echo "Refusing to start perception without camera timestamp safety." >&2
  return 1
}

stop_processes() {
  for pid in "${PIDS[@]}"; do
    kill -INT -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  wait || true
  PIDS=()
  PID_NAMES=()
  rm -f "$PROCESS_MANIFEST"
}

shutdown() {
  trap - INT TERM EXIT
  echo "Stopping GPU vision pipeline..."
  stop_processes
}

handle_signal() {
  shutdown
  exit 0
}
trap handle_signal INT TERM
trap shutdown EXIT

validate_recovery_path() {
  case "$RECOVERY_ROOT" in
    /tmp/piper_vision_recovery|/tmp/piper_vision_recovery/*) ;;
    *)
      echo "Refusing unexpected vision recovery root: $RECOVERY_ROOT" >&2
      exit 1
      ;;
  esac
  case "$RECOVERY_REQUEST" in
    "$RECOVERY_ROOT"/*) ;;
    *)
      echo "Recovery request must be inside $RECOVERY_ROOT: $RECOVERY_REQUEST" >&2
      exit 1
      ;;
  esac
  case "$PROCESS_MANIFEST" in
    "$RECOVERY_ROOT"/*) ;;
    *)
      echo "Process manifest must be inside $RECOVERY_ROOT: $PROCESS_MANIFEST" >&2
      exit 1
      ;;
  esac
  mkdir -p "$RECOVERY_ROOT"
}

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

recovery_backoff() {
  case "$1" in
    1) echo "${PIPER_TIMESTAMP_RECOVERY_BACKOFF_1_SEC:-2}" ;;
    2) echo "${PIPER_TIMESTAMP_RECOVERY_BACKOFF_2_SEC:-5}" ;;
    3) echo "${PIPER_TIMESTAMP_RECOVERY_BACKOFF_3_SEC:-10}" ;;
    *) echo "${PIPER_TIMESTAMP_RECOVERY_MAX_BACKOFF_SEC:-30}" ;;
  esac
}

start_stack() {
  clear_live_spool
  rm -f "$RECOVERY_REQUEST"
  if [ "${PIPER_REUSE_EXISTING_CAMERA:-0}" = "1" ]; then
    echo "Reusing existing L515 camera; automatic camera restart is disabled."
    export PIPER_TIMESTAMP_AUTO_RECOVERY=0
  else
    export PIPER_TIMESTAMP_AUTO_RECOVERY="${PIPER_TIMESTAMP_AUTO_RECOVERY:-1}"
    if [ "${PIPER_CAMERA_LOW_BANDWIDTH:-0}" = "1" ]; then
      echo "Using reduced-bandwidth L515 depth profile."
      start_process camera "$ROOT/L515_camera/start_l515_camera_low_bandwidth.sh"
    else
      start_process camera "$ROOT/L515_camera/start_l515_camera.sh"
    fi
    sleep "${PIPER_CAMERA_STARTUP_SEC:-7}"
  fi
  start_required_watchdog
  start_process heavy_bridge "$ROOT/L515_camera/run_heavy_refresh_bridge.sh"
  start_process heavy_cuda_worker "$ROOT/L515_camera/run_heavy_model_worker.sh"
  start_process sam2_cuda_worker "$ROOT/L515_camera/run_sam2_live_worker.sh"
  start_process target_cloud "$ROOT/L515_camera/run_target_cloud.sh"
  sleep 2
  # Start last so its one-shot initialization request has active subscribers.
  start_process sam2_bridge "$ROOT/L515_camera/run_sam2_live_bridge.sh"
  sleep 2
  start_process gpu_geometry "$ROOT/L515_camera/run_gpu_geometry.sh"
  GENERATION_STARTED=$(date +%s)

  echo "GPU vision pipeline is running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID."
  echo "Timestamp health: /piper/camera_timestamp_health"
  echo "Target: /piper/sam2_target_mask"
  echo "Obstacles: /piper/sam2_obstacle_mask"
  echo "Obstacle instances: /piper/obstacle_instances_3d"
  echo "Cloud: /piper/target_cloud"
  echo "Press Ctrl+C to stop all vision processes."
}

validate_recovery_path
start_stack

while true; do
  if [ -f "$RECOVERY_REQUEST" ]; then
    if [ "${PIPER_REUSE_EXISTING_CAMERA:-0}" = "1" ]; then
      echo "Camera timestamp is unhealthy, but this pipeline does not own the camera." >&2
      echo "Automatic motion remains blocked; restart the external camera." >&2
      rm -f "$RECOVERY_REQUEST"
    else
      now=$(date +%s)
      if [ $((now - GENERATION_STARTED)) -ge \
          "${PIPER_TIMESTAMP_RECOVERY_RESET_SEC:-120}" ]; then
        RECOVERY_COUNT=0
      fi
      RECOVERY_COUNT=$((RECOVERY_COUNT + 1))
      delay=$(recovery_backoff "$RECOVERY_COUNT")
      echo "Camera timestamp fault confirmed while arm is stationary."
      sed -n '1,12p' "$RECOVERY_REQUEST" || true
      echo "Restarting the complete vision stack in ${delay}s (attempt $RECOVERY_COUNT)."
      stop_processes
      sleep "$delay"
      start_stack
      continue
    fi
  fi
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
