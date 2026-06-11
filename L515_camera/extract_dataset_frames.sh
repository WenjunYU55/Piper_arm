#!/usr/bin/env bash
set -eo pipefail

export ROS_DOMAIN_ID=42
L515_REQUIRE_REALSENSE=1
L515_REQUIRE_PIPER=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

IMAGE_TOPIC=${1:-/camera/color/image_raw}
OUTPUT_DIR=${2:-/home/prl/Piper_arm/datasets/plant_tracking/images/raw}
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Extracting frames from ${IMAGE_TOPIC} into ${OUTPUT_DIR}"
echo "Run ros2 bag play in another terminal if the topic is coming from a bag."
ros2 run image_view extract_images --ros-args -r image:="$IMAGE_TOPIC"
