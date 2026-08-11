#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/source_piper_foxy_environment.sh"
export PIPER_ARM_ROOT="$ROOT"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${PIPER_MISSION_FASTDDS_PROFILE:-$ROOT/fastdds_gui_udp_only.xml}"
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
export ROS_LOCALHOST_ONLY=0

# The action name and mission-owned ROS/process resources are intentionally
# singleton.  Two coordinators can otherwise accept the same action goal and
# independently start drivers, cameras, and command owners.  Keep this file
# descriptor open across exec so the lock has the exact launch lifetime.
MISSION_LOCK_FILE="${PIPER_MISSION_LOCK_FILE:-/tmp/piper_target_scan_mission.lock}"
exec 9>"$MISSION_LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: a PiPER target-scan coordinator is already running (lock: $MISSION_LOCK_FILE)." >&2
  exit 73
fi

REAL_MOTION="${PIPER_MISSION_ENABLE_REAL_MOTION:-0}"
GATEWAY_HEARTBEAT="${PIPER_MISSION_REQUIRE_GATEWAY_HEARTBEAT:-0}"
SPEEDS_QUALIFIED="${PIPER_MISSION_SPEEDS_QUALIFIED:-0}"
case "$REAL_MOTION" in 0) ROS_REAL_MOTION=false ;; 1) ROS_REAL_MOTION=true ;; *) exit 2 ;; esac
case "$GATEWAY_HEARTBEAT" in 0) ROS_GATEWAY_HEARTBEAT=false ;; 1) ROS_GATEWAY_HEARTBEAT=true ;; *) exit 2 ;; esac
case "$SPEEDS_QUALIFIED" in 0) ROS_SPEEDS_QUALIFIED=false ;; 1) ROS_SPEEDS_QUALIFIED=true ;; *) exit 2 ;; esac

exec ros2 launch piper_mobile_manipulation target_scan_mission.launch.py \
  project_root:="$ROOT" \
  manage_processes:=true \
  enable_real_arm_motion:="$ROS_REAL_MOTION" \
  motion_speed_profile_qualified:="$ROS_SPEEDS_QUALIFIED" \
  require_gateway_heartbeat:="$ROS_GATEWAY_HEARTBEAT" \
  max_pending_missions:="${PIPER_MISSION_MAX_PENDING:-8}" \
  mission_queue_coalesce_sec:="${PIPER_MISSION_QUEUE_COALESCE_SEC:-1.0}" \
  mission_spool_root:="${PIPER_MISSION_SPOOL_ROOT:-/tmp/piper_target_scan_missions}" \
  free_motion_speed_percent:="${PIPER_MISSION_FREE_MOTION_SPEED_PERCENT:-30.0}" \
  contact_speed_percent:="${PIPER_MISSION_CONTACT_SPEED_PERCENT:-10.0}"
