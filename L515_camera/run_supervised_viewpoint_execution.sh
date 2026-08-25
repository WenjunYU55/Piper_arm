#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VIEWPOINT_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# This stack consumes camera-derived ROS messages but does not link against the
# RealSense packages.  Keep its type support on the single Foxy/PiPER overlay;
# sourcing the RealSense build overlay here caused live JointState callbacks to
# stop in the Tesseract bridge while the endpoint still appeared in the graph.
# shellcheck disable=SC1091
source "$ROOT/source_piper_foxy_environment.sh"
export PIPER_ARM_ROOT="$ROOT"
export ROS_DOMAIN_ID="$VIEWPOINT_ROS_DOMAIN_ID"
# Foxy's ROS_LOCALHOST_ONLY=1 override re-enables Fast DDS shared memory and can
# replay corrupt graph-discovery state after a participant crash.  This profile
# supplies loopback-only UDP itself, so ROS_LOCALHOST_ONLY must remain 0.
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/fastdds_gui_udp_only.xml"
# Keep Foxy's native rmw_fastrtps endpoint QoS.  The XML remains responsible
# only for the loopback UDP participant transport; forcing XML endpoint QoS
# here makes variable-size PrepareAcquisition replies time out.
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
export ROS_LOCALHOST_ONLY=0

ENABLE_MOTION="${PIPER_ENABLE_REAL_VIEWPOINT_MOTION:-0}"
case "$ENABLE_MOTION" in
  0) ROS_ENABLE_MOTION=false ;;
  1) ROS_ENABLE_MOTION=true ;;
  *)
    echo "PIPER_ENABLE_REAL_VIEWPOINT_MOTION must be 0 or 1." >&2
    exit 2
    ;;
esac

SPEED_PERCENT="${PIPER_VIEWPOINT_SPEED_PERCENT:-5.0}"
MAX_VIEWS="${PIPER_VIEWPOINT_MAX_VIEWS:-13}"
MIN_VIEWS="${PIPER_VIEWPOINT_MIN_VIEWS:-13}"
AUTO_CAPTURE="${PIPER_VIEWPOINT_AUTO_CAPTURE:-1}"
MISSION_POLICY="${PIPER_VIEWPOINT_MISSION_POLICY:-0}"
CLOSED_LOOP_ONE_VIEW="${PIPER_VIEWPOINT_CLOSED_LOOP_ONE_VIEW:-0}"
FLOOR_PROFILE="${PIPER_FLOOR_PROFILE:-tabletop}"
case "$FLOOR_PROFILE" in
  tabletop|ground) ;;
  *)
    echo "PIPER_FLOOR_PROFILE must be exactly tabletop or ground." >&2
    exit 2
    ;;
esac
case "$AUTO_CAPTURE" in
  0) ROS_AUTO_CAPTURE=false ;;
  1) ROS_AUTO_CAPTURE=true ;;
  *)
    echo "PIPER_VIEWPOINT_AUTO_CAPTURE must be 0 or 1." >&2
    exit 2
    ;;
esac
case "$CLOSED_LOOP_ONE_VIEW" in
  0) ROS_CLOSED_LOOP_ONE_VIEW=false ;;
  1) ROS_CLOSED_LOOP_ONE_VIEW=true ;;
  *)
    echo "PIPER_VIEWPOINT_CLOSED_LOOP_ONE_VIEW must be 0 or 1." >&2
    exit 2
    ;;
esac
case "$MISSION_POLICY" in
  0) ROS_MISSION_POLICY=false ;;
  1) ROS_MISSION_POLICY=true ;;
  *)
    echo "PIPER_VIEWPOINT_MISSION_POLICY must be 0 or 1." >&2
    exit 2
    ;;
esac
if [ "$ENABLE_MOTION" = "1" ]; then
  echo "Starting supervised viewpoint executor with real motion opt-in."
  echo "It will not enable the arm and still requires exact plan approval."
else
  echo "Starting supervised viewpoint executor in proposal-only mode."
fi

exec ros2 launch piper_mobile_manipulation supervised_viewpoint_execution.launch.py \
  enable_real_arm_motion:="$ROS_ENABLE_MOTION" \
  speed_percent:="$SPEED_PERCENT" \
  max_execution_viewpoints:="$MAX_VIEWS" \
  min_execution_viewpoints:="$MIN_VIEWS" \
  auto_capture:="$ROS_AUTO_CAPTURE" \
  allow_mission_policy:="$ROS_MISSION_POLICY" \
  closed_loop_one_view:="$ROS_CLOSED_LOOP_ONE_VIEW" \
  floor_profile:="$FLOOR_PROFILE"
