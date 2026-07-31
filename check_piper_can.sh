#!/usr/bin/env bash
set -euo pipefail

CAN_PORT="${PIPER_CAN_PORT:-can0}"
CAPTURE_SECONDS="${PIPER_CAN_PREFLIGHT_SECONDS:-3}"

if ! command -v ip >/dev/null 2>&1 || ! command -v candump >/dev/null 2>&1; then
  echo "ERROR: iproute2 and can-utils are required." >&2
  exit 2
fi
if ! ip link show "$CAN_PORT" >/dev/null 2>&1; then
  echo "ERROR: CAN interface $CAN_PORT does not exist." >&2
  exit 2
fi

details="$(ip -details -statistics link show "$CAN_PORT")"
printf '%s\n' "$details"
if ! grep -q 'can state ERROR-ACTIVE' <<<"$details"; then
  echo "ERROR: $CAN_PORT is not ERROR-ACTIVE. Do not start or enable the arm driver." >&2
  exit 1
fi

capture="$(mktemp /tmp/piper_can_preflight.XXXXXX)"
trap 'rm -f "$capture"' EXIT
timeout "$CAPTURE_SECONDS" candump -L "$CAN_PORT" >"$capture" 2>&1 || status=$?
status="${status:-0}"
if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
  sed -n '1,20p' "$capture" >&2
  echo "ERROR: passive CAN capture failed." >&2
  exit 1
fi
if [[ ! -s "$capture" ]]; then
  echo "ERROR: no valid CAN frames arrived in ${CAPTURE_SECONDS}s." >&2
  echo "Check arm power/E-stop, CAN-H/CAN-L polarity, common ground, and termination." >&2
  exit 1
fi

echo "PASS: $CAN_PORT is ERROR-ACTIVE and valid CAN frames are arriving."
sed -n '1,10p' "$capture"
