#!/bin/bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

echo "PiPER J6 diagnostics are read-only unless --calibrate is supplied."
echo "Stop start_piper.sh and the GUI before using direct CAN diagnostics."
exec python3 "$SCRIPT_DIR/piper_joint6_zero.py" "$@"
