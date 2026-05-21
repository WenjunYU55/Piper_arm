#!/usr/bin/env python3

import os
import signal
import time


BRIDGE_MARKERS = (
    "/home/prl/Piper_arm/start_bridge",
    "/home/prl/Piper_arm/start_bridge.sh",
    "piper_remote remote_bridge.launch.py",
    "piper_remote/remote_bridge",
)


def read_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            data = handle.read().replace(b"\x00", b" ").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""
    return data.decode("utf-8", errors="replace")


def bridge_pids():
    current_pid = os.getpid()
    matches = []

    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue

        pid = int(name)
        if pid == current_pid:
            continue

        cmdline = read_cmdline(pid)
        if not cmdline:
            continue

        if any(marker in cmdline for marker in BRIDGE_MARKERS):
            matches.append((pid, cmdline))

    return matches


def wait_for_exit(pids, timeout):
    deadline = time.time() + timeout
    remaining = set(pids)

    while remaining and time.time() < deadline:
        for pid in list(remaining):
            if not os.path.exists(f"/proc/{pid}"):
                remaining.remove(pid)
        if remaining:
            time.sleep(0.1)

    return remaining


def signal_processes(matches, sig):
    for pid, _ in matches:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Permission denied stopping PID {pid}. Try with sudo.")


def main():
    matches = bridge_pids()
    if not matches:
        print("PiPER bridge is not running.")
        return 0

    print("Stopping PiPER bridge processes:")
    for pid, cmdline in matches:
        print(f"  PID {pid}: {cmdline}")

    signal_processes(matches, signal.SIGTERM)
    remaining = wait_for_exit([pid for pid, _ in matches], timeout=3.0)

    if remaining:
        print("Bridge did not stop after SIGTERM; sending SIGKILL.")
        signal_processes([(pid, "") for pid in remaining], signal.SIGKILL)
        remaining = wait_for_exit(remaining, timeout=2.0)

    if remaining:
        print("Some bridge processes could not be stopped:")
        for pid in sorted(remaining):
            print(f"  PID {pid}")
        return 1

    print("PiPER bridge stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
