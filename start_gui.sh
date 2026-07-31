#!/bin/bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PIPER_ROS_DOMAIN_ID="${PIPER_ROS_DOMAIN_ID:-42}"
export ROS_DOMAIN_ID="$PIPER_ROS_DOMAIN_ID"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/source_piper_foxy_environment.sh"

# Foxy ships Fast DDS 2.1.4.  Keep the GUI and its owned ROS scan processes off
# the shared-memory transport: stale SHM port state previously caused an
# unbounded allocation in internal graph deserialization.  UDPv4 remains
# fully interoperable with the existing local driver and perception nodes.
export FASTRTPS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/fastdds_gui_udp_only.xml"
# Keep Foxy's native rmw_fastrtps endpoint QoS.  Enabling XML endpoint QoS on
# this install selects fixed-size service histories and drops the larger
# PrepareAcquisition reply even though the service remains discoverable.
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
# Foxy's ROS_LOCALHOST_ONLY=1 code path replaces the XML transport list and
# silently re-enables shared memory.  The XML profile itself restricts UDP to
# 127.0.0.1, so setting this to 0 does not expose the GUI off-host.
export ROS_LOCALHOST_ONLY=0

echo "WARNING: the PiPER GUI can enable/disable the arm and publish real joint commands."
echo "Using UDP-only Fast DDS transport for the GUI-owned process tree."
exec "$SCRIPT_DIR/piper_gui_native.py" "$@"
