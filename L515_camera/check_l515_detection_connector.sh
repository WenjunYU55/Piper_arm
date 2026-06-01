#!/usr/bin/env bash
set -euo pipefail

L515_REQUIRE_REALSENSE=1
L515_REQUIRE_PIPER=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

echo "Checking L515 target-detection connector."
echo
echo "Expected running terminals:"
echo "  1) ./start_l515_camera.sh"
echo "  2) ./run_l515_perception.sh"
echo

check_topic_type() {
  local topic=$1
  local expected=$2
  local actual

  actual=$(ros2 topic type "$topic" 2>/dev/null || true)
  if [ -z "$actual" ]; then
    echo "MISSING  $topic"
    echo "         expected type: $expected"
    return 1
  fi

  if [ "$actual" != "$expected" ]; then
    echo "WRONG    $topic"
    echo "         expected type: $expected"
    echo "         actual type:   $actual"
    return 1
  fi

  echo "OK       $topic [$actual]"
  return 0
}

echo_one_message() {
  local topic=$1

  if ros2 topic echo -h 2>&1 | grep -q -- '--once'; then
    timeout 5 ros2 topic echo --once "$topic" || echo "No $topic sample within 5 seconds."
  else
    timeout 5 bash -c 'ros2 topic echo "$1" 2>/dev/null | sed -n "1,/^---$/p; /^---$/q"' _ "$topic" \
      || echo "No $topic sample within 5 seconds."
  fi
}

echo "Camera-driver inputs:"
camera_ok=0
check_topic_type /camera/color/image_raw sensor_msgs/msg/Image || camera_ok=1
check_topic_type /camera/aligned_depth_to_color/image_raw sensor_msgs/msg/Image || camera_ok=1
check_topic_type /camera/color/camera_info sensor_msgs/msg/CameraInfo || camera_ok=1

echo
echo "Processed connector outputs:"
perception_ok=0
check_topic_type /piper/detection_2d piper_mobile_manipulation/msg/Detection2D || perception_ok=1
check_topic_type /piper/target_3d piper_mobile_manipulation/msg/Target3D || perception_ok=1
check_topic_type /piper/tracked_target piper_mobile_manipulation/msg/TrackedTarget || perception_ok=1
check_topic_type /piper/detection_debug_image sensor_msgs/msg/Image || perception_ok=1

echo
if [ "$camera_ok" -ne 0 ]; then
  echo "Camera topics are missing. Start ./start_l515_camera.sh first."
fi
if [ "$perception_ok" -ne 0 ]; then
  echo "Perception topics are missing. Start ./run_l515_perception.sh after the camera."
fi

if [ "$camera_ok" -ne 0 ] || [ "$perception_ok" -ne 0 ]; then
  exit 1
fi

echo "One-message samples from the future base/arm connector topics:"
echo
echo "--- /piper/detection_2d ---"
echo_one_message /piper/detection_2d
echo
echo "--- /piper/target_3d ---"
echo_one_message /piper/target_3d
echo
echo "--- /piper/tracked_target ---"
echo_one_message /piper/tracked_target
echo
echo "Connector check complete."
