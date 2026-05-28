#!/usr/bin/env bash
set -euo pipefail

LIBREALSENSE_SRC=/home/prl/Piper_arm/L515_camera/realsense_ws/src/librealsense

if [ ! -d "$LIBREALSENSE_SRC" ]; then
  echo "Missing librealsense source at $LIBREALSENSE_SRC"
  echo "Run ./fetch_realsense_sources.sh first."
  exit 1
fi

echo "Installing RealSense udev rules..."
cd "$LIBREALSENSE_SRC"
sudo ./scripts/setup_udev_rules.sh

echo
echo "Disabling USB autosuspend for this boot..."
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend >/dev/null

echo
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo
echo "Done. Unplug the L515, wait 10 seconds, then plug it back in."
echo "After that, run:"
echo "  cd /home/prl/Piper_arm/L515_camera"
echo "  ./diagnose_l515_usb.sh"
