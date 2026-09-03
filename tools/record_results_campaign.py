#!/usr/bin/env python3
"""Run the passive, file-only PiPER campaign evidence recorder."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_campaign.recorder import main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(main())
