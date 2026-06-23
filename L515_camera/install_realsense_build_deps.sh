#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  libeigen3-dev \
  libusb-1.0-0-dev \
  libssl-dev \
  libudev-dev \
  ros-foxy-diagnostic-updater \
  ros-foxy-xacro \
  ros-foxy-cv-bridge \
  ros-foxy-image-transport \
  ros-foxy-builtin-interfaces \
  ros-foxy-geometry-msgs \
  ros-foxy-nav-msgs \
  ros-foxy-rclcpp \
  ros-foxy-rclcpp-components \
  ros-foxy-ros-environment \
  ros-foxy-rosidl-default-generators \
  ros-foxy-sensor-msgs \
  ros-foxy-std-msgs \
  ros-foxy-tf2 \
  ros-foxy-rqt-image-view \
  ros-foxy-rmw-cyclonedds-cpp \
  ros-foxy-tf2-geometry-msgs \
  ros-foxy-tf2-ros \
  python3-opencv \
  python3-colcon-common-extensions \
  python3-numpy \
  python3-yaml \
  pkg-config \
  usbutils \
  udev \
  dkms
