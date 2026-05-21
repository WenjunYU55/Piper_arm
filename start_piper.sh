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
PIPER_RVIZ_CTRL_FLAG="${PIPER_RVIZ_CTRL_FLAG:-false}"
PIPER_JOINT_CTRL_TOPIC="${PIPER_JOINT_CTRL_TOPIC:-/joint_ctrl_single}"
PIPER_ROS_DOMAIN_ID="${PIPER_ROS_DOMAIN_ID:-42}"
PIPER_ENABLE_TIMEOUT="${PIPER_ENABLE_TIMEOUT:-15.0}"
export ROS_DOMAIN_ID="$PIPER_ROS_DOMAIN_ID"

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

# 8. Reset CAN before activation
echo "Resetting CAN interface..."
sudo ip link set "$CAN_PORT" down || true

# 9. Activate CAN
echo "Activating CAN interface..."
if [ -n "$CAN_USB_ADDRESS" ]; then
    bash can_activate.sh "$CAN_PORT" "$CAN_BITRATE" "$CAN_USB_ADDRESS"
else
    bash can_activate.sh "$CAN_PORT" "$CAN_BITRATE"
fi

# 10. Confirm CAN is UP
if ! ip link show "$CAN_PORT" | grep -q "UP"; then
    echo "ERROR: CAN interface did not come UP."
    echo ""
    echo "Try manually:"
    echo "  sudo ip link set $CAN_PORT down"
    echo "  sudo ip link set $CAN_PORT type can bitrate $CAN_BITRATE"
    echo "  sudo ip link set $CAN_PORT up"
    echo "  ip -details link show $CAN_PORT"
    exit 1
fi

echo "CAN interface is UP."

# 11. Launch PiPER driver
echo "Launching PiPER driver."
echo "ROS_DOMAIN_ID is $ROS_DOMAIN_ID."
echo "Arm will NOT auto-enable."
echo "Use a second terminal for enable_piper, disable_piper, and topic commands."

ros2 launch piper start_single_piper.launch.py \
  can_port:="$CAN_PORT" \
  auto_enable:="$PIPER_AUTO_ENABLE" \
  gripper_exist:="$PIPER_GRIPPER_EXIST" \
  rviz_ctrl_flag:="$PIPER_RVIZ_CTRL_FLAG" \
  enable_timeout:="$PIPER_ENABLE_TIMEOUT" \
  joint_ctrl_topic:="$PIPER_JOINT_CTRL_TOPIC"
