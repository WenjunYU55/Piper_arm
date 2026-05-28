#!/usr/bin/env bash
set -euo pipefail

echo "Checking USB bus..."
if command -v lsusb >/dev/null 2>&1; then
  lsusb
  echo
  echo "USB topology:"
  lsusb -t || true
else
  echo "lsusb not found. Install it with: sudo apt-get install -y usbutils"
fi

echo
echo "USB autosuspend setting:"
if [ -r /sys/module/usbcore/parameters/autosuspend ]; then
  cat /sys/module/usbcore/parameters/autosuspend
else
  echo "Cannot read /sys/module/usbcore/parameters/autosuspend"
fi

echo
echo "Sourcing ROS 2 Foxy and local RealSense workspace..."
L515_REQUIRE_REALSENSE=1
# shellcheck disable=SC1091
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh

echo
echo "RealSense firmware updater device list:"
if command -v rs-fw-update >/dev/null 2>&1; then
  rs-fw-update -l || true
else
  echo "rs-fw-update not found"
fi

echo
echo "RealSense SDK short device list:"
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices -s || true
else
  echo "rs-enumerate-devices not found"
fi

echo
echo "Recent kernel USB messages:"
if command -v dmesg >/dev/null 2>&1; then
  dmesg 2>/dev/null | tail -n 80 || echo "dmesg is restricted on this system"
else
  echo "dmesg not found"
fi
