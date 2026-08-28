#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAN_PORT="${PIPER_CAN_PORT:-can0}"
UNIT_SOURCE="$ROOT/deployment/piper-can@.service"
RULE_TEMPLATE="$ROOT/deployment/80-piper-can.rules.in"

if [[ ! "$CAN_PORT" =~ ^can[0-9]+$ ]]; then
    echo "ERROR: PIPER_CAN_PORT must match can<number>; received: $CAN_PORT" >&2
    exit 2
fi
if [ ! -f "$UNIT_SOURCE" ] || [ ! -f "$RULE_TEMPLATE" ]; then
    echo "ERROR: PiPER CAN deployment files are missing." >&2
    exit 1
fi
if pgrep -f '[p]iper_single_ctrl' >/dev/null; then
    echo "ERROR: Stop the PiPER driver before installing or restarting CAN." >&2
    exit 1
fi
if ! ip link show "$CAN_PORT" >/dev/null 2>&1; then
    echo "ERROR: SocketCAN interface $CAN_PORT is not present." >&2
    echo "Connect the USB-CAN adapter and retry." >&2
    exit 1
fi

temporary_dir="$(mktemp -d)"
trap 'rm -r "$temporary_dir"' EXIT
sed "s/@CAN_INTERFACE@/$CAN_PORT/g" \
    "$RULE_TEMPLATE" >"$temporary_dir/80-piper-can.rules"

echo "Installing boot-time PiPER CAN configuration for $CAN_PORT."
echo "This is the one provisioning step that requires sudo."
sudo install -m 0644 "$UNIT_SOURCE" /etc/systemd/system/piper-can@.service
sudo install -m 0644 "$temporary_dir/80-piper-can.rules" \
    /etc/udev/rules.d/80-piper-can.rules
sudo systemctl daemon-reload
sudo udevadm control --reload-rules
sudo systemctl enable "piper-can@$CAN_PORT.service"
sudo systemctl restart "piper-can@$CAN_PORT.service"

if ! ip link show "$CAN_PORT" | head -n 1 | grep -q 'state UP'; then
    echo "ERROR: $CAN_PORT did not become UP." >&2
    sudo systemctl status --no-pager "piper-can@$CAN_PORT.service" || true
    exit 1
fi
if ! ip -details link show "$CAN_PORT" | grep -q 'bitrate 1000000'; then
    echo "ERROR: $CAN_PORT is not configured at 1000000 bit/s." >&2
    exit 1
fi

echo "PiPER CAN boot service installed and active."
ip -details link show "$CAN_PORT"
