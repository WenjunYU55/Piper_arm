"""Command-free regression checks for the 2026-08-20 failed aim record."""

from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[4]
TOOLS = REPOSITORY / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from replay_nbv_coverage import replay  # noqa: E402


def test_scan_20260820_142919_is_rejected_by_repaired_final_aim_gate():
    scan_dir = (
        REPOSITORY / 'datasets' / 'active_scan' /
        'scan_20260820_142919')
    if not scan_dir.is_dir():
        pytest.skip('recorded NBV regression scan is not available')

    result = replay(scan_dir)
    aim = result['generations'][0]['capture_aim']

    # The arm accurately achieved the old requested orientation, but that
    # request aimed at stale/biased target geometry.  The repaired executor
    # must refuse to commit the resulting observation and request a new NBV.
    assert aim['selected_to_achieved_aim_error_deg'] < 0.1
    assert aim['final_target_aim_error_deg'] == pytest.approx(
        13.4237306077, abs=1e-6)
    assert not aim['passes_repaired_final_aim_gate']
    assert aim['repaired_final_aim_limit_deg'] == 5.0
