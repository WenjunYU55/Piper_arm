#!/bin/bash

set -e

echo "=== PiPER startup check ==="

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PIPER_WORKSPACE="${PIPER_WORKSPACE:-$SCRIPT_DIR/piper_ros_foxy}"

CAN_PORT="${PIPER_CAN_PORT:-can0}"
CAN_BITRATE="${PIPER_CAN_BITRATE:-1000000}"
CAN_USB_ADDRESS="${PIPER_CAN_USB_ADDRESS:-}"
PIPER_AUTO_ENABLE="${PIPER_AUTO_ENABLE:-false}"
PIPER_GRIPPER_EXIST="${PIPER_GRIPPER_EXIST:-true}"
PIPER_JOINT_CTRL_TOPIC="${PIPER_JOINT_CTRL_TOPIC:-/joint_ctrl_single}"
PIPER_ROS_DOMAIN_ID="${PIPER_ROS_DOMAIN_ID:-42}"
PIPER_ENABLE_TIMEOUT="${PIPER_ENABLE_TIMEOUT:-15.0}"
PIPER_JOINT_BOUNDS_PATH="${PIPER_JOINT_BOUNDS_PATH:-$SCRIPT_DIR/piper_joint_bounds.json}"
PIPER_FASTRTPS_PROFILE="${PIPER_FASTRTPS_PROFILE:-$SCRIPT_DIR/fastdds_gui_udp_only.xml}"
export ROS_DOMAIN_ID="$PIPER_ROS_DOMAIN_ID"

# Foxy/Fast DDS shared-memory participants on this workstation can retain
# stale port state across GUI/driver restarts.  Run the driver on the same
# loopback UDP transport as the GUI-owned scan processes so feedback, enable
# services and one-target MoveJ commands cannot split across transports.
if [ -f "$PIPER_FASTRTPS_PROFILE" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PIPER_FASTRTPS_PROFILE"
    export RMW_FASTRTPS_USE_QOS_FROM_XML=0
else
    echo "ERROR: Fast DDS transport profile not found."
    echo "Expected: $PIPER_FASTRTPS_PROFILE"
    exit 1
fi

# 1. Source ROS 2 Foxy
if [ -f /opt/ros/foxy/setup.bash ]; then
    source /opt/ros/foxy/setup.bash
else
    echo "ERROR: ROS 2 Foxy setup file not found."
    echo "Expected: /opt/ros/foxy/setup.bash"
    exit 1
fi

# 2. Source PiPER workspace
if [ -f "$PIPER_WORKSPACE/install/setup.bash" ]; then
    source "$PIPER_WORKSPACE/install/setup.bash"
else
    echo "ERROR: PiPER workspace setup file not found."
    echo "Expected: $PIPER_WORKSPACE/install/setup.bash"
    echo ""
    echo "Try:"
    echo "  cd $PIPER_WORKSPACE"
    echo "  colcon build --symlink-install"
    exit 1
fi

# 3. Check ROS distro
if [ "$ROS_DISTRO" != "foxy" ]; then
    echo "ERROR: ROS_DISTRO is not foxy."
    echo "Current ROS_DISTRO: $ROS_DISTRO"
    exit 1
fi

echo "ROS 2 Foxy sourced."

# 4. Check PiPER package exists
if ! ros2 pkg list | grep -q "^piper$"; then
    echo "ERROR: piper package not found."
    echo ""
    echo "Try:"
    echo "  cd $PIPER_WORKSPACE"
    echo "  source /opt/ros/foxy/setup.bash"
    echo "  source install/setup.bash"
    echo "  colcon build --symlink-install"
    exit 1
fi

echo "PiPER package found."

# 5. Check Python runtime dependencies
python3 - << 'EOF'
missing = []

modules = {
    "can": "python-can",
    "scipy": "scipy",
    "piper_sdk": "piper_sdk"
}

for import_name, package_name in modules.items():
    try:
        __import__(import_name)
    except ImportError:
        missing.append(package_name)

if missing:
    print("ERROR: Missing Python packages:", ", ".join(missing))
    print("")
    print("Install with:")
    print("  pip3 install python-can scipy piper_sdk")
    raise SystemExit(1)

print("Python runtime dependencies found.")
EOF

# 6. Go to workspace
if [ -d "$PIPER_WORKSPACE" ]; then
    cd "$PIPER_WORKSPACE"
else
    echo "ERROR: Workspace directory not found."
    echo "Expected: $PIPER_WORKSPACE"
    exit 1
fi

# 7. Check CAN interface exists
if ! ip link show "$CAN_PORT" > /dev/null 2>&1; then
    echo "ERROR: CAN interface $CAN_PORT not found."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check the USB-CAN adapter is plugged in."
    echo "  2. Check the arm is powered."
    echo "  3. Run: lsusb"
    echo "  4. Run: ip link"
    echo "  5. Run: dmesg | grep -i can"
    echo ""
    echo "If your CAN interface is not called $CAN_PORT, run with:"
    echo "  PIPER_CAN_PORT=<interface> ./start_piper.sh"
    exit 1
fi

echo "CAN interface $CAN_PORT found."

can_is_ready() {
    ip link show "$CAN_PORT" | head -n 1 | grep -q 'state UP' \
        && ip -details link show "$CAN_PORT" \
            | grep -q "bitrate $CAN_BITRATE"
}

# 8. Reuse boot-provisioned CAN. Autonomous/headless startup must never wait
# for an interactive sudo password. A direct terminal run retains the legacy
# one-time activation fallback for development hosts.
if can_is_ready; then
    echo "CAN interface is already UP at $CAN_BITRATE bit/s."
else
    if [ ! -t 0 ]; then
        echo "ERROR: CAN interface $CAN_PORT is not provisioned for headless startup."
        echo "Run once from an operator terminal:"
        echo "  cd $SCRIPT_DIR"
        echo "  PIPER_CAN_PORT=$CAN_PORT ./scripts/setup/install_piper_can_service.sh"
        exit 1
    fi
    echo "CAN is not ready; starting the interactive development-host setup."
    if [ -n "$CAN_USB_ADDRESS" ]; then
        bash can_activate.sh "$CAN_PORT" "$CAN_BITRATE" "$CAN_USB_ADDRESS"
    else
        bash can_activate.sh "$CAN_PORT" "$CAN_BITRATE"
    fi
fi

# 9. Confirm CAN is UP at the exact PiPER bitrate.
if ! can_is_ready; then
    echo "ERROR: CAN interface did not become UP at $CAN_BITRATE bit/s."
    echo ""
    echo "Provision it once with:"
    echo "  PIPER_CAN_PORT=$CAN_PORT ./scripts/setup/install_piper_can_service.sh"
    exit 1
fi

echo "CAN interface is UP."

# 10. Launch PiPER driver
echo "Launching PiPER driver."
echo "ROS_DOMAIN_ID is $ROS_DOMAIN_ID."
echo "Using UDP-only Fast DDS transport: $FASTRTPS_DEFAULT_PROFILES_FILE"
echo "Arm will NOT auto-enable."
echo "Use scripts/robot/enable_piper.sh or scripts/robot/disable_piper.sh only for explicit commissioning."
echo "WARNING: once enabled, reset/gui/joint commands can move the real arm."

ros2 launch piper start_single_piper.launch.py \
  can_port:="$CAN_PORT" \
  auto_enable:="$PIPER_AUTO_ENABLE" \
  gripper_exist:="$PIPER_GRIPPER_EXIST" \
  enable_timeout:="$PIPER_ENABLE_TIMEOUT" \
  joint_bounds_path:="$PIPER_JOINT_BOUNDS_PATH" \
  joint_ctrl_topic:="$PIPER_JOINT_CTRL_TOPIC"
