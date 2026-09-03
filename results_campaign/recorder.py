"""Polling sidecar that preserves ephemeral mission evidence.

This process is intentionally file-only: no rclpy, publishers, services,
actions, planners, cameras, or robot command interfaces are imported.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import time

from .campaign import CampaignStore
from .collector import collect_task


def poll(store: CampaignStore, interval_sec: float = 1.0, once: bool = False) -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    stopped = {'value': False}

    def stop(_signum, _frame):
        stopped['value'] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopped['value']:
        for attempt in store.task_attempts():
            try:
                collect_task(store.project_root, attempt)
            except (OSError, TypeError, ValueError):
                # A source may be half-written. The next poll retries it.
                continue
        if once:
            return
        time.sleep(max(0.2, float(interval_sec)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--interval-sec', type=float, default=5.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    store = CampaignStore(args.project_root, args.campaign)
    store.create_or_load()
    poll(store, interval_sec=args.interval_sec, once=args.once)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
