#!/bin/bash

set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PIPER_WORKSPACE="${PIPER_WORKSPACE:-$SCRIPT_DIR/piper_ros_foxy}"
PIPER_BRIDGE_HOST="${PIPER_BRIDGE_HOST:-0.0.0.0}"
PIPER_BRIDGE_PORT="${PIPER_BRIDGE_PORT:-8080}"
PIPER_ROS_DOMAIN_ID="${PIPER_ROS_DOMAIN_ID:-42}"
export ROS_DOMAIN_ID="$PIPER_ROS_DOMAIN_ID"

if [ -f /opt/ros/foxy/setup.bash ]; then
    source /opt/ros/foxy/setup.bash
else
    echo "ERROR: ROS 2 Foxy setup file not found."
    echo "Expected: /opt/ros/foxy/setup.bash"
    exit 1
fi

if [ -f "$PIPER_WORKSPACE/install/setup.bash" ]; then
    source "$PIPER_WORKSPACE/install/setup.bash"
else
    echo "ERROR: PiPER workspace setup file not found."
    echo "Expected: $PIPER_WORKSPACE/install/setup.bash"
    echo ""
    echo "Try:"
    echo "  cd $PIPER_WORKSPACE"
    echo "  source /opt/ros/foxy/setup.bash"
    echo "  colcon build --packages-select piper_remote"
    exit 1
fi

if ! ros2 pkg list | grep -q "^piper_remote$"; then
    echo "ERROR: piper_remote package not found."
    echo ""
    echo "Try:"
    echo "  cd $PIPER_WORKSPACE"
    echo "  source /opt/ros/foxy/setup.bash"
    echo "  colcon build --packages-select piper_remote"
    exit 1
fi

echo "Starting PiPER remote bridge."
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "Host: $PIPER_BRIDGE_HOST"
echo "Port: $PIPER_BRIDGE_PORT"
echo ""
echo "From another device on the same network, open:"
echo "  http://<robot-ip>:$PIPER_BRIDGE_PORT/health"
echo ""

ros2 launch piper_remote remote_bridge.launch.py \
  host:="$PIPER_BRIDGE_HOST" \
  port:="$PIPER_BRIDGE_PORT"
