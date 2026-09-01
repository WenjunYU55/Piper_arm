"""Pure aggregation of completed mission timing diagnostics.

The functions consume result payloads only.  They do not import ROS and cannot
start a process, camera, planner, CAN interface, or robot controller.
"""

import math

from motion_planning.benchmarking import numeric_summary


def mission_efficiency_row(payload, source=''):
    """Normalize one action-result payload into a comparison row."""
    summary = payload.get('action_summary', {})
    timing = summary.get('phase_timing', {})
    phase_totals = timing.get('phase_totals_sec', {})
    elapsed = timing.get('total_elapsed_sec')
    captures = int(payload.get('capture_count', 0))
    has_timing = (
        isinstance(elapsed, (int, float))
        and math.isfinite(float(elapsed))
        and float(elapsed) >= 0.0
        and isinstance(phase_totals, dict)
    )
    elapsed_value = float(elapsed) if has_timing else None
    seconds_per_capture = None
    captures_per_minute = None
    if has_timing and captures > 0 and elapsed_value > 0.0:
        seconds_per_capture = elapsed_value / captures
        captures_per_minute = 60.0 * captures / elapsed_value
    return {
        'source': str(source),
        'task_id': str(payload.get('task_id', '')),
        'planner_backend': str(summary.get('planner_backend', 'unknown')),
        'outcome': str(payload.get('outcome', '')),
        'safe_shutdown': bool(payload.get('safe_shutdown', False)),
        'capture_count': captures,
        'timing_available': has_timing,
        'total_elapsed_sec': elapsed_value,
        'seconds_per_capture': seconds_per_capture,
        'captures_per_minute': captures_per_minute,
        'phase_totals_sec': {
            str(phase): float(duration)
            for phase, duration in phase_totals.items()
            if isinstance(duration, (int, float))
            and math.isfinite(float(duration))
            and float(duration) >= 0.0
        } if has_timing else {},
    }


def summarize_mission_rows(rows):
    """Aggregate comparable missions by planner backend."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['planner_backend'], []).append(row)
    result = {}
    for backend, values in sorted(grouped.items()):
        measured = [row for row in values if row['timing_available']]
        completed = [row for row in measured if row['outcome'] == 'SUCCEEDED']
        phases = sorted({
            phase for row in measured
            for phase in row['phase_totals_sec']
        })
        result[backend] = {
            'mission_count': len(values),
            'timed_mission_count': len(measured),
            'success_count': len(completed),
            'success_rate': (
                float(len(completed)) / len(measured) if measured else None),
            'safe_shutdown_rate': (
                sum(bool(row['safe_shutdown']) for row in measured)
                / float(len(measured)) if measured else None),
            'total_elapsed_sec': numeric_summary(
                row['total_elapsed_sec'] for row in measured),
            'capture_count': numeric_summary(
                row['capture_count'] for row in measured),
            'seconds_per_capture': numeric_summary(
                row['seconds_per_capture'] for row in measured),
            'captures_per_minute': numeric_summary(
                row['captures_per_minute'] for row in measured),
            'phase_totals_sec': {
                phase: numeric_summary(
                    row['phase_totals_sec'].get(phase) for row in measured)
                for phase in phases
            },
        }
    return result
