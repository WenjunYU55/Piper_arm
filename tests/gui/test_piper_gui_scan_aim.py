"""GUI camera-on-ray tolerance regressions."""

from pathlib import Path

import pytest

from piper_gui.scan_aim import read_scan_aim, validate_scan_aim, write_scan_aim


def _config(tmp_path, value=5.0):
    path = Path(tmp_path) / 'scan_execution_params.yaml'
    path.write_text(
        '/**:\n  ros__parameters:\n'
        '    scan_target_max_boresight_deg: 20.0\n'
        '    final_capture_aim_tolerance_deg: %.1f  # hard-capped\n'
        '    debug: true\n' % value,
        encoding='utf-8')
    return path


def test_scan_aim_round_trip_changes_only_authoritative_value(tmp_path):
    path = _config(tmp_path)
    before = path.read_text(encoding='utf-8')
    saved = write_scan_aim(path, 3.5)
    assert saved.final_capture_aim_tolerance_deg == pytest.approx(3.5)
    assert read_scan_aim(path) == saved
    assert path.read_text(encoding='utf-8') == before.replace('5.0', '3.5', 1)


@pytest.mark.parametrize('value', [0.9, 90.1, 'bad', float('nan')])
def test_scan_aim_rejects_values_outside_later_view_range(value):
    with pytest.raises(ValueError, match='camera aim tolerance'):
        validate_scan_aim(value)


@pytest.mark.parametrize('value', [1.0, 5.0, 45.0, 90.0])
def test_scan_aim_accepts_full_later_view_range(value):
    assert validate_scan_aim(value).final_capture_aim_tolerance_deg == value


def test_scan_aim_refuses_missing_or_duplicate_authority(tmp_path):
    path = Path(tmp_path) / 'scan_execution_params.yaml'
    path.write_text('/**:\n  ros__parameters:\n    debug: true\n')
    with pytest.raises(ValueError, match='exactly one'):
        read_scan_aim(path)
