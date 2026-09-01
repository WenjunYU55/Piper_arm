#!/usr/bin/env bash
#
# Source the one supported Foxy/PiPER overlay for GUI and scan processes.
# This file must be sourced, not executed.

_PIPER_ENV_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
_PIPER_ENV_ROOT="$(cd "$(dirname "$_PIPER_ENV_SCRIPT")" && pwd)"
_PIPER_ENV_WORKSPACE="${PIPER_WORKSPACE:-$_PIPER_ENV_ROOT/piper_ros_foxy}"

# Keep every process sourced through the canonical environment on the same
# bounded local Fast DDS graph as the driver and GUI.  Callers may override the
# domain or profile explicitly, but a fresh terminal must not silently fall
# back to domain 0 or Foxy's unstable shared-memory transport.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-${PIPER_ROS_DOMAIN_ID:-42}}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${PIPER_FASTRTPS_PROFILE:-$_PIPER_ENV_ROOT/fastdds_gui_udp_only.xml}}"
export RMW_FASTRTPS_USE_QOS_FROM_XML="${RMW_FASTRTPS_USE_QOS_FROM_XML:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

case $- in
  *u*) _PIPER_ENV_RESTORE_NOUNSET=1 ;;
  *) _PIPER_ENV_RESTORE_NOUNSET=0 ;;
esac

_piper_env_restore_nounset() {
  if [ "$_PIPER_ENV_RESTORE_NOUNSET" = "1" ]; then
    set -u
  fi
}

if [ ! -f /opt/ros/foxy/setup.bash ]; then
  echo "Missing ROS 2 Foxy setup: /opt/ros/foxy/setup.bash" >&2
  _piper_env_restore_nounset
  return 1
fi
if [ ! -f "$_PIPER_ENV_WORKSPACE/install/setup.bash" ]; then
  echo "Missing canonical PiPER setup: $_PIPER_ENV_WORKSPACE/install/setup.bash" >&2
  _piper_env_restore_nounset
  return 1
fi

# A shell that previously sourced the obsolete root colcon workspace can keep
# that package ahead of piper_ros_foxy even after sourcing the canonical setup.
# Clear inherited overlays before rebuilding the exact runtime environment.
set +u
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH

# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash
# shellcheck disable=SC1090
source "$_PIPER_ENV_WORKSPACE/install/setup.bash"

_PIPER_ENV_EXPECTED_MOBILE="$_PIPER_ENV_WORKSPACE/install/piper_mobile_manipulation"
_PIPER_ENV_EXPECTED_TESSERACT="$_PIPER_ENV_WORKSPACE/install/piper_tesseract_foxy"
_PIPER_ENV_ACTUAL_MOBILE="$(ros2 pkg prefix piper_mobile_manipulation 2>/dev/null || true)"
_PIPER_ENV_ACTUAL_TESSERACT="$(ros2 pkg prefix piper_tesseract_foxy 2>/dev/null || true)"

if [ "$_PIPER_ENV_ACTUAL_MOBILE" != "$_PIPER_ENV_EXPECTED_MOBILE" ]; then
  echo "Wrong piper_mobile_manipulation overlay: ${_PIPER_ENV_ACTUAL_MOBILE:-not found}" >&2
  echo "Expected: $_PIPER_ENV_EXPECTED_MOBILE" >&2
  _piper_env_restore_nounset
  return 1
fi
if [ "$_PIPER_ENV_ACTUAL_TESSERACT" != "$_PIPER_ENV_EXPECTED_TESSERACT" ]; then
  echo "Wrong piper_tesseract_foxy overlay: ${_PIPER_ENV_ACTUAL_TESSERACT:-not found}" >&2
  echo "Expected: $_PIPER_ENV_EXPECTED_TESSERACT" >&2
  _piper_env_restore_nounset
  return 1
fi

if ! python3 - <<'PY'
import piper_mobile_manipulation.scan_capture  # noqa: F401
from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.msg import (
    MotionPlan, OccluderAction, PlannerReadiness,
    TesseractPlan, TesseractReadiness)
from piper_mobile_manipulation.srv import (
    AuthorizeMission, PrepareAcquisition, RequestMotionPlan)
from piper_msgs.msg import PiperMotionLimits

required_fields = (
    'bootstrap_recovery_end_points',
    'bootstrap_recovery_joints',
    'bootstrap_recovery_delta_rad',
    'bootstrap_recovery_evidence_json',
    'motion_limits_sha256',
    'execution_speed_percent',
    'command_rate_hz',
    'timing_policy',
    'source_request_id',
)
message = MotionPlan()
missing = [name for name in required_fields if not hasattr(message, name)]
if missing:
    raise SystemExit(
        'Stale MotionPlan message schema; missing: ' + ', '.join(missing))
if not all(hasattr(message, name) for name in (
        'backend', 'backend_version', 'collision_model_qualified',
        'planning_duration_sec', 'trajectory_duration_sec')):
    raise SystemExit('Stale generic MotionPlan message schema')
readiness = PlannerReadiness()
if not all(hasattr(readiness, name) for name in (
        'generation_id', 'worker_ready', 'acquisition_ready',
        'multiview_ready', 'manipulation_ready', 'acquisition_blockers',
        'multiview_blockers', 'manipulation_blockers')):
    raise SystemExit('Stale PlannerReadiness message schema')
# These legacy Tesseract transports remain deliberate compatibility aliases.
if not hasattr(TesseractPlan(), 'source_request_id'):
    raise SystemExit('Stale TesseractPlan compatibility schema')
if not hasattr(TesseractReadiness(), 'manipulation_ready'):
    raise SystemExit('Stale TesseractReadiness compatibility schema')
mission_goal = RunTargetScan.Goal()
if not all(hasattr(mission_goal, name) for name in (
        'task_id', 'task_type', 'target_label', 'target_profile',
        'planner_backend', 'rough_target', 'target_confidence', 'deadline_sec')):
    raise SystemExit('Stale RunTargetScan action schema')
if not hasattr(RequestMotionPlan.Request(), 'plan_kind'):
    raise SystemExit('Stale RequestMotionPlan service schema')
if not hasattr(OccluderAction(), 'mission_sha256'):
    raise SystemExit('Stale OccluderAction message schema')
if not hasattr(AuthorizeMission.Request(), 'mission_sha256'):
    raise SystemExit('Stale AuthorizeMission service schema')
prepare_request = PrepareAcquisition.Request()
if not all(hasattr(prepare_request, name) for name in (
        'session_id', 'rough_target')):
    raise SystemExit('Stale PrepareAcquisition service schema')
limit_message = PiperMotionLimits()
limit_fields = (
    'joint_names',
    'max_velocity_rad_s',
    'max_acceleration_rad_s2',
    'valid',
    'limits_sha256',
    'source',
    'reason',
)
missing = [name for name in limit_fields if not hasattr(limit_message, name)]
if missing:
    raise SystemExit(
        'Stale PiperMotionLimits message schema; missing: ' + ', '.join(missing))
PY
then
  _piper_env_restore_nounset
  return 1
fi

_piper_env_restore_nounset

unset -f _piper_env_restore_nounset
unset _PIPER_ENV_SCRIPT
unset _PIPER_ENV_ROOT
unset _PIPER_ENV_WORKSPACE
unset _PIPER_ENV_EXPECTED_MOBILE
unset _PIPER_ENV_EXPECTED_TESSERACT
unset _PIPER_ENV_ACTUAL_MOBILE
unset _PIPER_ENV_ACTUAL_TESSERACT
unset _PIPER_ENV_RESTORE_NOUNSET
