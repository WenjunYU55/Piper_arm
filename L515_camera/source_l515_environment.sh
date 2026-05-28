#!/usr/bin/env bash

L515_ROS_SETUP=${L515_ROS_SETUP:-/opt/ros/foxy/setup.bash}
L515_REALSENSE_SETUP=${L515_REALSENSE_SETUP:-/home/prl/Piper_arm/L515_camera/realsense_ws/install/setup.bash}
L515_PIPER_SETUP=${L515_PIPER_SETUP:-/home/prl/Piper_arm/piper_ros_foxy/install/setup.bash}
case $- in
  *u*) _L515_RESTORE_NOUNSET=1 ;;
  *) _L515_RESTORE_NOUNSET=0 ;;
esac

_l515_source_setup() {
  local setup_file=$1
  local required=$2

  if [ ! -f "$setup_file" ]; then
    if [ "$required" = "1" ]; then
      echo "Missing required setup file: $setup_file"
      return 1
    fi
    return 0
  fi

  set +u
  # shellcheck disable=SC1090
  source "$setup_file"
  if [ "$_L515_RESTORE_NOUNSET" = "1" ]; then
    set -u
  fi
}

_l515_source_setup "$L515_ROS_SETUP" 1

export ROS_LOCALHOST_ONLY=${L515_ROS_LOCALHOST_ONLY:-1}

if [ "${L515_USE_CYCLONE:-0}" = "1" ] && [ -z "${RMW_IMPLEMENTATION:-}" ] && ros2 pkg prefix rmw_cyclonedds_cpp >/dev/null 2>&1; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi

_l515_source_setup "$L515_REALSENSE_SETUP" "${L515_REQUIRE_REALSENSE:-0}"
_l515_source_setup "$L515_PIPER_SETUP" "${L515_REQUIRE_PIPER:-0}"

unset -f _l515_source_setup
unset _L515_RESTORE_NOUNSET
