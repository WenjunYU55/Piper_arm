#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  libusb-1.0-0-dev \
  libssl-dev \
  libudev-dev \
  ros-foxy-diagnostic-updater \
  ros-foxy-xacro \
  ros-foxy-cv-bridge \
  ros-foxy-image-transport \
  ros-foxy-rqt-image-view \
  ros-foxy-rmw-cyclonedds-cpp \
  ros-foxy-tf2-geometry-msgs \
  ros-foxy-tf2-ros \
  python3-opencv \
  python3-colcon-common-extensions \
  pkg-config \
  usbutils \
  udev \
  dkms
