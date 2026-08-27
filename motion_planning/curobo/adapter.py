"""Pure request and trajectory adapter for the cuRobo worker."""

import hashlib
import json
import math


JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
SUPPORTED_PLAN_KINDS = (
    'MULTIVIEW_SCAN', 'ROUGH_ACQUISITION', 'RETURN_HOME')
UNSUPPORTED_PLAN_KINDS = (
    'OCCLUSION_PROBE', 'OCCLUDER_PICK_PLACE', 'OCCLUDER_PUSH')
COMMAND_RATE_HZ = 20.0
MAXIMUM_JOINT_STEP_RAD = 0.05
MOVEJ_NOMINAL_VELOCITY_RAD_S = (5.0, 5.0, 5.0, 5.0, 5.0, 3.0)
FIXED_MOUNT_SEAM_M = 0.010


class CuroboContractError(ValueError):
    """Reject malformed or unsupported worker data before GPU use."""


def canonical_bytes(value):
    """Return deterministic JSON bytes used by the spool hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')


def sha256_value(value):
    """Hash one JSON-compatible value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def attach_digest(payload, field):
    """Return a shallow copy with a deterministic digest field."""
    result = dict(payload)
    result.pop(field, None)
    result[field] = sha256_value(result)
    return result


def verify_digest(payload, field):
    """Fail if a payload is not bound to its declared digest."""
    if not isinstance(payload, dict):
        raise CuroboContractError('payload must be an object')
    declared = str(payload.get(field, ''))
    content = dict(payload)
    content.pop(field, None)
    if len(declared) != 64 or sha256_value(content) != declared:
        raise CuroboContractError('%s mismatch' % field)


def finite_vector(value, length, label):
    """Return one bounded finite numeric vector."""
    if not isinstance(value, (list, tuple)) or len(value) != int(length):
        raise CuroboContractError(
            '%s must contain %d values' % (label, length))
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise CuroboContractError('%s contains non-finite values' % label)
    return result


def validate_request(request):
    """Validate cuRobo-owned fields before loading CUDA or a model."""
    if not isinstance(request, dict):
        raise CuroboContractError('request must be an object')
    verify_digest(request, 'request_sha256')
    if request.get('planner_backend') != 'curobo':
        raise CuroboContractError('request planner_backend is not curobo')
    plan_kind = str(request.get('plan_kind', ''))
    if plan_kind in UNSUPPORTED_PLAN_KINDS:
        raise CuroboContractError(
            'plan kind %s is explicitly unsupported by cuRobo' % plan_kind)
    if plan_kind not in SUPPORTED_PLAN_KINDS:
        raise CuroboContractError('unsupported plan kind: %s' % plan_kind)
    planning = request.get('planning', {})
    if (
            planning.get('planner') != 'MotionGen'
            or planning.get('pipeline') != 'CUROBO_V1'):
        raise CuroboContractError(
            'cuRobo planner configuration is missing or inconsistent')
    start = request.get('start_state', {})
    if tuple(start.get('joint_names', ())) != JOINT_NAMES:
        raise CuroboContractError(
            'start joint order must be joint1 through joint6')
    start_positions = finite_vector(
        start.get('positions_rad'), 6, 'start positions')
    limits = request.get('limits', {}).get('position_rad')
    if not isinstance(limits, list) or len(limits) != 6:
        raise CuroboContractError(
            'request position limits must contain six ranges')
    outside_limits = []
    for index, (position, bounds) in enumerate(zip(start_positions, limits)):
        low, high = finite_vector(bounds, 2, 'joint position limit')
        if low >= high:
            raise CuroboContractError('joint position limit is empty')
        if position < low or position > high:
            outside_limits.append(index + 1)
    if outside_limits:
        raise CuroboContractError(
            'unsupported cuRobo bootstrap from outside position limits: %s'
            % ','.join('joint%d' % index for index in outside_limits))
    scene = request.get('scene', {})
    finite_vector(scene.get('target_center_m'), 3, 'target center')
    candidates = scene.get('candidate_views')
    if not isinstance(candidates, list):
        raise CuroboContractError('candidate_views must be a list')
    if plan_kind != 'RETURN_HOME' and not candidates:
        raise CuroboContractError('planning request has no candidate views')
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise CuroboContractError('candidate %d is not an object' % index)
        finite_vector(
            candidate.get('camera_position_m'), 3,
            'candidate %d camera position' % index)
        direction = finite_vector(
            candidate.get('look_direction'), 3,
            'candidate %d look direction' % index)
        if sum(value * value for value in direction) <= 1e-12:
            raise CuroboContractError(
                'candidate %d look direction is zero' % index)
    obstacles = scene.get('obstacles', [])
    if not isinstance(obstacles, list):
        raise CuroboContractError('scene obstacles must be a list')
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict) or obstacle.get('type') != 'box':
            raise CuroboContractError(
                'obstacle %d is not a supported box' % index)
        low = finite_vector(
            obstacle.get('minimum_m'), 3, 'obstacle minimum')
        high = finite_vector(
            obstacle.get('maximum_m'), 3, 'obstacle maximum')
        if any(left >= right for left, right in zip(low, high)):
            raise CuroboContractError('obstacle %d has empty bounds' % index)
    return request


def obstacle_cuboids(request, floor_z_m=None):
    """Convert the authoritative box scene to cuRobo cuboid dictionaries."""
    result = []
    for index, obstacle in enumerate(request['scene'].get('obstacles', [])):
        low = finite_vector(obstacle['minimum_m'], 3, 'obstacle minimum')
        high = finite_vector(obstacle['maximum_m'], 3, 'obstacle maximum')
        result.append({
            'name': str(obstacle.get('id', 'obstacle_%d' % index)),
            'pose': [
                (low[0] + high[0]) * 0.5,
                (low[1] + high[1]) * 0.5,
                (low[2] + high[2]) * 0.5,
                1.0, 0.0, 0.0, 0.0,
            ],
            'dims': [high[i] - low[i] for i in range(3)],
        })
    if floor_z_m is not None:
        floor = float(floor_z_m)
        if not math.isfinite(floor):
            raise CuroboContractError('floor_z_m is not finite')
        result.append({
            'name': 'configured_support_floor',
            # Match the rigid mounting contact exception in the canonical
            # Tesseract SRDF without disabling floor collision for moving
            # links: the world box stops just below the fixed mount plane.
            'pose': [
                0.0, 0.0, floor - FIXED_MOUNT_SEAM_M - 0.05,
                1.0, 0.0, 0.0, 0.0],
            'dims': [4.0, 4.0, 0.10],
        })
    return tuple(result)


def normalize_trajectory(positions, native_dt_sec, speed_percent):
    """Preserve path geometry while producing the common 20 Hz schedule."""
    rows = [finite_vector(row, 6, 'trajectory position') for row in positions]
    if len(rows) < 2:
        raise CuroboContractError(
            'native trajectory must contain at least two positions')
    native_dt = float(native_dt_sec)
    speed = float(speed_percent)
    if not math.isfinite(native_dt) or native_dt <= 0.0:
        raise CuroboContractError('native trajectory dt is invalid')
    if not math.isfinite(speed) or speed < 1.0 or speed > 100.0:
        raise CuroboContractError('execution speed must be within 1..100')
    result = [rows[0]]
    times = [0.0]
    period = 1.0 / COMMAND_RATE_HZ
    scaled_velocity = tuple(
        value * speed / 100.0 for value in MOVEJ_NOMINAL_VELOCITY_RAD_S)
    for left, right in zip(rows, rows[1:]):
        deltas = tuple(abs(b - a) for a, b in zip(left, right))
        subdivisions = max(
            1,
            int(math.ceil(max(deltas) / MAXIMUM_JOINT_STEP_RAD)),
        )
        for step in range(1, subdivisions + 1):
            fraction = float(step) / float(subdivisions)
            point = tuple(
                a + (b - a) * fraction for a, b in zip(left, right))
            increment = tuple(
                abs(b - a) for a, b in zip(result[-1], point))
            interval = max(
                period,
                native_dt / subdivisions,
                max(
                    delta / limit
                    for delta, limit in zip(increment, scaled_velocity)),
            )
            result.append(point)
            times.append(times[-1] + interval)
    zero = [0.0] * 6
    return [{
        'positions_rad': list(point),
        'velocities_rad_s': list(zero),
        'accelerations_rad_s2': list(zero),
        'time_from_start_s': round(when, 9),
    } for point, when in zip(result, times)]


def trajectory_segment(points, is_return_home=False):
    """Build generic collision-qualified segment metadata."""
    return {
        'points': list(points),
        'minimum_clearance_m': -1.0,
        'limiting_link_pair': 'unreported_by_curobo_v0.7.8',
        'is_return_home': bool(is_return_home),
        'bootstrap_recovery_used': False,
        'bootstrap_recovery_end_point': -1,
        'bootstrap_recovery_joint': 0,
        'bootstrap_recovery_delta_rad': 0.0,
        'startup_home_static': False,
        'configured_home_direct_joint_move': False,
        'configured_home_goal_positions_rad': [],
        'collision_validation_bypassed': False,
        'home_stage': '',
        'validation': 'curobo_v0.7.8_motiongen_collision_qualified',
        'trajectory_blending': 'curobo_native_interpolated_path',
        'pass_through_blending_applied': False,
        'sdk_execution_mode': 'TIMED_STREAM',
        'sdk_command_anchor_count': len(points),
    }


def worker_rejection_code(error):
    """Map predictable failures to stable fail-closed reason codes."""
    text = str(error).lower()
    if 'unsupported' in text:
        return 'PLANNER_UNSUPPORTED'
    if 'cuda' in text or 'curobo' in text or 'model' in text:
        return 'PLANNER_UNAVAILABLE'
    if 'collision' in text or 'valid query' in text:
        return 'PLANNING_COLLISION_REJECTED'
    return 'PLANNING_FAILED'
