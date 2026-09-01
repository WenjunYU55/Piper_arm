"""Tests for command-free mission efficiency aggregation."""

import pytest

from motion_planning.mission_efficiency import (
    mission_efficiency_row, summarize_mission_rows)


def result(backend='tesseract', elapsed=60.0, captures=3,
           outcome='SUCCEEDED', safe=True):
    return {
        'task_id': 'task-12345678',
        'outcome': outcome,
        'safe_shutdown': safe,
        'capture_count': captures,
        'action_summary': {
            'planner_backend': backend,
            'phase_timing': {
                'total_elapsed_sec': elapsed,
                'phase_totals_sec': {
                    'STARTING': 5.0,
                    'VIEW_PLANNING': 15.0,
                    'CAPTURING': 40.0,
                },
            },
        },
    }


def test_normalizes_efficiency_metrics():
    row = mission_efficiency_row(result(), source='one.json')

    assert row['timing_available'] is True
    assert row['seconds_per_capture'] == pytest.approx(20.0)
    assert row['captures_per_minute'] == pytest.approx(3.0)
    assert row['phase_totals_sec']['VIEW_PLANNING'] == pytest.approx(15.0)


def test_legacy_result_is_retained_but_not_timed():
    row = mission_efficiency_row({
        'task_id': 'old', 'outcome': 'FAILED', 'capture_count': 0,
        'action_summary': {},
    })

    assert row['timing_available'] is False
    assert row['total_elapsed_sec'] is None
    assert row['seconds_per_capture'] is None


def test_summarizes_backends_without_mixing_legacy_rows():
    rows = [
        mission_efficiency_row(result('tesseract')),
        mission_efficiency_row(result('curobo', 30.0, 3)),
        mission_efficiency_row({
            'outcome': 'FAILED', 'capture_count': 0,
            'action_summary': {'planner_backend': 'curobo'},
        }),
    ]

    summary = summarize_mission_rows(rows)

    assert summary['curobo']['mission_count'] == 2
    assert summary['curobo']['timed_mission_count'] == 1
    assert summary['curobo']['total_elapsed_sec']['mean'] == pytest.approx(
        30.0)
    assert summary['tesseract']['success_rate'] == pytest.approx(1.0)
