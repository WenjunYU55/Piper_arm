#!/bin/bash

set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PIPER_WORKSPACE="${PIPER_WORKSPACE:-$SCRIPT_DIR/piper_ros_foxy}"
PIPER_ROS_DOMAIN_ID="${PIPER_ROS_DOMAIN_ID:-42}"
export ROS_DOMAIN_ID="$PIPER_ROS_DOMAIN_ID"

source /opt/ros/foxy/setup.bash
source "$PIPER_WORKSPACE/install/setup.bash"

for _ in $(seq 1 20); do
    if ros2 service list | grep -q '^/enable_srv$'; then
        break
    fi
    echo "Waiting for /enable_srv..."
    sleep 0.5
done

ros2 service call /enable_srv piper_msgs/srv/Enable "{enable_request: true}"
