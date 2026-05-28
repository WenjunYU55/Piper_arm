#!/usr/bin/env bash
set -e

echo "Checking ROS 2 Foxy environment..."
L515_REQUIRE_REALSENSE=1
L515_REQUIRE_PIPER=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

echo
echo "librealsense packages installed through dpkg:"
if command -v dpkg-query >/dev/null 2>&1; then
  dpkg-query -W 'librealsense2*' 2>/dev/null || echo "No librealsense2 dpkg packages found"
else
  echo "dpkg-query not available"
fi

echo
echo "RealSense ROS package:"
if ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
  ros2 pkg prefix realsense2_camera
else
  echo "realsense2_camera not found"
fi

echo
echo "piper_mobile_manipulation package:"
if ros2 pkg prefix piper_mobile_manipulation >/dev/null 2>&1; then
  ros2 pkg prefix piper_mobile_manipulation
else
  echo "piper_mobile_manipulation not found. Build it with:"
  echo "  cd /home/prl/Piper_arm/piper_ros_foxy"
  echo "  colcon build --packages-select piper_mobile_manipulation"
fi

echo
echo "Connected RealSense devices, if librealsense tools are installed:"
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices | sed -n '1,80p'
  if rs-enumerate-devices 2>/dev/null | grep -q "L515"; then
    echo "L515 detected. Check firmware is 1.5.8.1 or later."
  fi
else
  echo "rs-enumerate-devices not found"
fi

echo
echo "Camera topics currently visible:"
ros2 topic list 2>/dev/null | grep camera || echo "No /camera topics visible. Start the RealSense launch first."

echo
echo "Expected topics for this package:"
echo "  /camera/color/image_raw"
echo "  /camera/color/camera_info"
echo "  /camera/aligned_depth_to_color/image_raw"
