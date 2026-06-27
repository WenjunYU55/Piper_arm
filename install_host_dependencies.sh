#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${EUID}" -eq 0 ]; then
  echo "Run this script as your normal user; it invokes sudo when needed." >&2
  exit 1
fi

if [ ! -f /opt/ros/foxy/setup.bash ]; then
  echo "ROS 2 Foxy is required at /opt/ros/foxy." >&2
  echo "Install ROS first, then rerun this script." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  can-utils \
  cmake \
  curl \
  ethtool \
  git \
  iproute2 \
  python3-can \
  python3-colcon-common-extensions \
  python3-numpy \
  python3-opencv \
  python3-pip \
  python3-rosdep \
  python3-scipy \
  python3-setuptools \
  python3-tk \
  python3-yaml \
  ros-foxy-cv-bridge \
  ros-foxy-diagnostic-updater \
  ros-foxy-image-tools \
  ros-foxy-joint-state-publisher-gui \
  ros-foxy-message-filters \
  ros-foxy-robot-state-publisher \
  ros-foxy-ros2-control \
  ros-foxy-ros2-controllers \
  ros-foxy-rqt-image-view \
  ros-foxy-rviz2 \
  ros-foxy-tf2-geometry-msgs \
  ros-foxy-tf2-ros \
  ros-foxy-xacro

# Ubuntu 20.04 provides python-can 3.3.2, but piper_sdk requires >=3.3.4.
# python-can 4.6+ dropped Python 3.8, so keep the tested Python 3.8-compatible release.
python3 -m pip install --user "python-can==4.5.0" "piper_sdk==0.6.1"

# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
# joint_state_publisher_gui is installed explicitly above. Foxy's archived
# rosdep index no longer resolves that package key reliably after Foxy EOL.
rosdep install --from-paths "$ROOT/piper_ros_foxy/src" --ignore-src -r -y \
  --rosdistro foxy \
  --skip-keys joint_state_publisher_gui

echo "Host dependencies installed. Build instructions are in README.md."
