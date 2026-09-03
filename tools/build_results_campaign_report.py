#!/usr/bin/env python3
"""Build CSV, XLSX, PNG and PDF results from a recorded campaign."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_campaign.report import main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(main())
