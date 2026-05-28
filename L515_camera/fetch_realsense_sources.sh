#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/prl/Piper_arm/L515_camera/realsense_ws
SDK_TAG=${SDK_TAG:-v2.50.0}
ROS_TAG=${ROS_TAG:-4.0.4}

mkdir -p "${WORKSPACE}/src"

echo "Fetching RealSense source into:"
echo "  ${WORKSPACE}/src"
echo
echo "Selected versions:"
echo "  librealsense:  ${SDK_TAG}"
echo "  realsense-ros: ${ROS_TAG}"
echo

if [ -d "${WORKSPACE}/src/librealsense/.git" ]; then
  echo "librealsense already exists. Updating checkout to ${SDK_TAG}."
  git -C "${WORKSPACE}/src/librealsense" fetch --tags --depth 1 origin "${SDK_TAG}"
  git -C "${WORKSPACE}/src/librealsense" checkout "${SDK_TAG}"
else
  git clone --branch "${SDK_TAG}" --depth 1 \
    https://github.com/realsenseai/librealsense.git \
    "${WORKSPACE}/src/librealsense"
fi

if [ -d "${WORKSPACE}/src/realsense-ros/.git" ]; then
  echo "realsense-ros already exists. Updating checkout to ${ROS_TAG}."
  git -C "${WORKSPACE}/src/realsense-ros" fetch --tags --depth 1 origin "${ROS_TAG}"
  git -C "${WORKSPACE}/src/realsense-ros" checkout "${ROS_TAG}"
else
  git clone --branch "${ROS_TAG}" --depth 1 \
    https://github.com/realsenseai/realsense-ros.git \
    "${WORKSPACE}/src/realsense-ros"
fi

echo
echo "Done. Sources are in ${WORKSPACE}/src"
echo "Next step is dependency install/build; do not build until we confirm system packages and kernel/librealsense state."
