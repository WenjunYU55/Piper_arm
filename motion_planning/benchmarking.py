"""Pure helpers for command-free motion-planner comparisons.

The benchmark owns volatile request identity and timing fields.  Scene,
calibration, limits, start state, candidate order, and planning policy remain
frozen by the source fixture so the two planner backends receive the same
physical problem.
"""

import copy
import hashlib
import json
import math
import statistics
import time


BACKEND_PLANNERS = {
    'tesseract': ('RRTConnect', 'OMPL_ISP'),
    'curobo': ('MotionGen', 'CUROBO_V1'),
}


def canonical_bytes(value):
    """Return deterministic finite JSON bytes."""
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode('utf-8')


def sha256_value(value):
    """Hash one JSON-compatible value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def attach_digest(value, field):
    """Return a copy with a canonical digest bound at ``field``."""
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = sha256_value(result)
    return result


def scenario_payload(request):
    """Remove only volatile/backend-selector fields from a request."""
    value = copy.deepcopy(request)
    for field in ('request_id', 'request_sha256', 'created_at_ns',
                  'expires_at_ns'):
        value.pop(field, None)
    value.pop('planner_backend', None)
    planning = value.get('planning', {})
    planning.pop('planner', None)
    planning.pop('pipeline', None)
    return value


def scenario_sha256(request):
    """Identify the backend-neutral frozen planning problem."""
    return sha256_value(scenario_payload(request))


def materialize_request(
        template, backend, run_index, now_ns=None, ttl_sec=600.0):
    """Create one fresh valid worker request from a frozen template."""
    if backend not in BACKEND_PLANNERS:
        raise ValueError('unsupported planner backend: %s' % backend)
    if isinstance(run_index, bool) or int(run_index) < 0:
        raise ValueError('run_index must be a non-negative integer')
    ttl = float(ttl_sec)
    if not math.isfinite(ttl) or ttl <= 0.0:
        raise ValueError('ttl_sec must be positive and finite')
    created = time.time_ns() if now_ns is None else int(now_ns)
    result = copy.deepcopy(template)
    result['planner_backend'] = backend
    planner, pipeline = BACKEND_PLANNERS[backend]
    result.setdefault('planning', {})['planner'] = planner
    result['planning']['pipeline'] = pipeline
    result['created_at_ns'] = created
    result['expires_at_ns'] = created + int(ttl * 1_000_000_000)
    identity = {
        'scenario_sha256': scenario_sha256(result),
        'backend': backend,
        'run_index': int(run_index),
        'created_at_ns': created,
    }
    result['request_id'] = sha256_value(identity)[:32]
    return attach_digest(result, 'request_sha256')


def trajectory_metrics(response):
    """Return backend-neutral path metrics from one worker response."""
    segments = response.get('segments', [])
    duration = 0.0
    point_count = 0
    path_length = 0.0
    maximum_step = 0.0
    for segment in segments:
        points = segment.get('points', [])
        point_count += len(points)
        if points:
            duration += float(points[-1]['time_from_start_s'])
        for first, second in zip(points, points[1:]):
            left = [float(value) for value in first['positions_rad']]
            right = [float(value) for value in second['positions_rad']]
            deltas = [abs(a - b) for a, b in zip(left, right)]
            path_length += sum(deltas)
            maximum_step = max(maximum_step, max(deltas, default=0.0))
    diagnostics = response.get('planning_diagnostics', {})
    return {
        'trajectory_duration_sec': duration,
        'trajectory_point_count': point_count,
        'joint_space_path_length_rad': path_length,
        'maximum_joint_step_rad': maximum_step,
        'selected_viewpoint_count': len(
            response.get('selected_viewpoints', [])),
        'candidate_viewpoints_considered': int(
            diagnostics.get('candidate_viewpoints_considered', 0)),
        'candidate_viewpoints_rejected': int(
            diagnostics.get('candidate_viewpoints_rejected', 0)),
        'feasible_viewpoints': int(diagnostics.get(
            'feasible_viewpoints', len(response.get(
                'selected_viewpoints', [])))),
        'backend_reported_planning_sec': float(
            diagnostics.get('planning_duration_sec', -1.0)),
    }


def percentile(values, percentage):
    """Return a linearly interpolated percentile without dependencies."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentage) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values):
    """Summarize finite measurements for benchmark reports."""
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
        and float(value) >= 0.0
    ]
    if not finite:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None,
                'minimum': None, 'maximum': None}
    return {
        'n': len(finite),
        'mean': statistics.fmean(finite),
        'median': statistics.median(finite),
        'p95': percentile(finite, 95.0),
        'minimum': min(finite),
        'maximum': max(finite),
    }


def summarize_trials(trials):
    """Group per-request trials by backend for a concise comparison."""
    grouped = {}
    for trial in trials:
        backend = str(trial['backend'])
        grouped.setdefault(backend, []).append(trial)
    result = {}
    for backend, rows in sorted(grouped.items()):
        measured = [row for row in rows if not row.get('warmup', False)]
        positive = [
            row for row in measured
            if row.get('expected_role') == 'recorded_achieved_geometry']
        negative = [
            row for row in measured
            if row.get('expected_role') == 'negative_control']
        policy = [
            row for row in measured
            if row.get('expected_role') == 'policy_control']
        successes = [row for row in positive if row.get('status') == 'success']
        rejected_negative = [
            row for row in negative if row.get('status') != 'success']
        exact = [
            row for row in successes
            if row.get('exact_collision_validation') == 'passed'
        ]
        result[backend] = {
            'trial_count': len(measured),
            'positive_trial_count': len(positive),
            'negative_control_count': len(negative),
            'policy_control_count': len(policy),
            'success_count': len(successes),
            'success_rate': (
                float(len(successes)) / len(positive) if positive else None),
            'negative_control_rejection_count': len(rejected_negative),
            'negative_control_rejection_rate': (
                float(len(rejected_negative)) / len(negative)
                if negative else None),
            'exact_validated_success_count': len(exact),
            'exact_validated_success_rate': (
                float(len(exact)) / len(successes) if successes else 0.0),
            'request_wall_sec': numeric_summary(
                row.get('request_wall_sec') for row in positive),
            'backend_reported_planning_sec': numeric_summary(
                row.get('backend_reported_planning_sec')
                for row in successes),
            'trajectory_duration_sec': numeric_summary(
                row.get('trajectory_duration_sec') for row in successes),
            'joint_space_path_length_rad': numeric_summary(
                row.get('joint_space_path_length_rad') for row in successes),
            'trajectory_point_count': numeric_summary(
                row.get('trajectory_point_count') for row in successes),
        }
    return result
