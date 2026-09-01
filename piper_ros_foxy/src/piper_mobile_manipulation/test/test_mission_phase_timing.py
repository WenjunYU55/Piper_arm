"""Pure tests for additive mission phase timing diagnostics."""

import pytest

from piper_mobile_manipulation.mission_core import MissionPhase, MissionSession


def session():
    return MissionSession({
        'task_id': 'phase-timing-test',
        'mission_sha256': 'a' * 64,
        'deadline_sec': 1200.0,
    }, started_monotonic=10.0, phase_started_monotonic=10.0)


def test_repeated_phases_are_retained_and_totalled():
    value = session()
    value.transition(MissionPhase.STARTING, 'start', now=11.0)
    value.transition(MissionPhase.PREFLIGHT, 'ready', now=13.0)
    value.transition(MissionPhase.STARTING, 'retry', now=14.0)

    report = value.phase_timing_summary(now=17.0)

    assert [item['phase'] for item in report['intervals']] == [
        'GOAL_LATCHED', 'STARTING', 'PREFLIGHT', 'STARTING']
    assert report['phase_totals_sec']['STARTING'] == pytest.approx(5.0)
    assert report['phase_totals_sec']['PREFLIGHT'] == pytest.approx(1.0)
    assert report['total_elapsed_sec'] == pytest.approx(7.0)


def test_timing_snapshot_does_not_mutate_live_session():
    value = session()
    value.transition(MissionPhase.STARTING, 'start', now=11.0)

    first = value.phase_timing_summary(now=12.0)
    second = value.phase_timing_summary(now=13.0)

    assert len(value.phase_timing_intervals) == 1
    assert first['phase_totals_sec']['STARTING'] == pytest.approx(1.0)
    assert second['phase_totals_sec']['STARTING'] == pytest.approx(2.0)


def test_terminal_transition_closes_last_nonterminal_interval():
    value = session()
    value.transition(MissionPhase.STOPPING, 'cleanup', now=12.0)
    value.transition(MissionPhase.SUCCEEDED, 'done', now=15.0)

    report = value.phase_timing_summary(now=15.0)

    assert report['phase_totals_sec']['STOPPING'] == pytest.approx(3.0)
    assert report['phase_totals_sec']['SUCCEEDED'] == pytest.approx(0.0)
