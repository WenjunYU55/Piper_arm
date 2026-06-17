#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

if [ ! -f /opt/ros/foxy/setup.bash ]; then
  echo "Missing ROS 2 Foxy setup: /opt/ros/foxy/setup.bash" >&2
  exit 1
fi

if [ ! -f /home/prl/Piper_arm/piper_ros_foxy/install/setup.bash ]; then
  echo "Missing PiPER workspace setup: /home/prl/Piper_arm/piper_ros_foxy/install/setup.bash" >&2
  echo "Build the workspace first if needed:" >&2
  echo "  cd /home/prl/Piper_arm/piper_ros_foxy" >&2
  echo "  colcon build --packages-select piper_mobile_manipulation" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash
# shellcheck disable=SC1091
source /home/prl/Piper_arm/piper_ros_foxy/install/setup.bash
set -u

python3 /home/prl/Piper_arm/L515_camera/capture_snapshot.py "$@"
