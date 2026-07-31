#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECOVERY_ROOT="${PIPER_VISION_RECOVERY_ROOT:-/tmp/piper_vision_recovery}"
PROCESS_MANIFEST="${PIPER_VISION_PROCESS_MANIFEST:-$RECOVERY_ROOT/process_groups.txt}"

case "$PROCESS_MANIFEST" in
  "$RECOVERY_ROOT"/*) ;;
  *)
    echo "Refusing process manifest outside $RECOVERY_ROOT: $PROCESS_MANIFEST" >&2
    exit 1
    ;;
esac

if [[ ! -f "$PROCESS_MANIFEST" ]]; then
  echo "No recorded GPU vision process groups are present."
  exit 0
fi

declare -a GROUPS_TO_STOP=()
while read -r name pid; do
  case "$name" in
    camera|camera_timestamp_watchdog|heavy_bridge|heavy_cuda_worker|sam2_cuda_worker|target_cloud|sam2_bridge|gpu_geometry) ;;
    *)
      echo "Ignoring unrecognized manifest entry: $name $pid" >&2
      continue
      ;;
  esac
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || continue
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
  [[ "$pgid" == "$pid" ]] || continue
  case "$args" in
    *"$ROOT/L515_camera/"*|*"$ROOT/AI_perception_tests/"*|*"$ROOT/piper_ros_foxy/install/"*|*"/opt/ros/foxy/bin/ros2"*)
      GROUPS_TO_STOP+=("$pid")
      ;;
    *)
      echo "Refusing stale or unrelated process group $pid ($name): $args" >&2
      ;;
  esac
done < "$PROCESS_MANIFEST"

if ((${#GROUPS_TO_STOP[@]} == 0)); then
  rm -f "$PROCESS_MANIFEST"
  echo "No live recorded GPU vision process groups remain."
  exit 0
fi

echo "Stopping ${#GROUPS_TO_STOP[@]} recorded GPU vision process groups..."
for pgid in "${GROUPS_TO_STOP[@]}"; do
  kill -INT -- "-$pgid" 2>/dev/null || true
done
sleep 2
for pgid in "${GROUPS_TO_STOP[@]}"; do
  kill -TERM -- "-$pgid" 2>/dev/null || true
done
sleep 1
for pgid in "${GROUPS_TO_STOP[@]}"; do
  if ps -eo pgid= | awk -v expected="$pgid" \
      '$1 == expected { found = 1 } END { exit !found }'; then
    echo "Process group $pgid did not stop; inspect it manually." >&2
    exit 1
  fi
done
rm -f "$PROCESS_MANIFEST"
echo "GPU vision pipeline processes stopped."
