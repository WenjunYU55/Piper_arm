#!/usr/bin/env bash
set -e
ROOT="${PIPER_ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/L515_camera/source_l515_environment.sh"
echo "Starting supervised cube workflow (dry-run only; it cannot move the arm)."
exec ros2 launch piper_mobile_manipulation supervised_cube_workflow.launch.py
