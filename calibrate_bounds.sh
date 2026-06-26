#!/bin/bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PIPER_WORKSPACE="${PIPER_WORKSPACE:-$SCRIPT_DIR/piper_ros_foxy}"
PIPER_ROS_DOMAIN_ID="${PIPER_ROS_DOMAIN_ID:-42}"
export ROS_DOMAIN_ID="$PIPER_ROS_DOMAIN_ID"

source /opt/ros/foxy/setup.bash
source "$PIPER_WORKSPACE/install/setup.bash"

echo "Bounds calibration records feedback only; you manually move the arm during prompts."
exec "$SCRIPT_DIR/piper_calibrate_bounds.py" "$@"
