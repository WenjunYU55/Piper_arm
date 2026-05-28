#!/usr/bin/env bash
set -eo pipefail

cd /home/prl/Piper_arm/piper_ros_foxy

if [ ! -f install/setup.bash ]; then
  echo "Workspace is not built yet. Run:"
  echo "  cd /home/prl/Piper_arm/piper_ros_foxy"
  echo "  colcon build --packages-select piper_mobile_manipulation"
  exit 1
fi

L515_REQUIRE_REALSENSE=1
L515_REQUIRE_PIPER=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

echo "Starting L515 perception-only pipeline."
echo "This does not use target handoff and does not move the PiPER arm."
ros2 launch piper_mobile_manipulation perception_only.launch.py
