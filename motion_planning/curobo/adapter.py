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
MAXIMUM_SCHEDULED_POINTS = 60000
FIXED_MOUNT_SEAM_M = 0.010
# Equal to the established per-joint planner inset and the existing 0.005-rad
# feedback-only boundary allowance.  Raw limits are never expanded: a start
# outside them is rejected before any re-entry is constructed.
START_STATE_MAXIMUM_CLIP_EXCURSION_RAD = 0.005
START_STATE_REENTRY_INTERIOR_OFFSET_RAD = 0.0001
START_STATE_REENTRY_MAXIMUM_STEP_RAD = 0.00025


class CuroboContractError(ValueError):
    """Reject malformed or unsupported worker data before GPU use."""


class CuroboPlanningError(CuroboContractError):
    """Base class for typed, fail-closed planner-runtime failures."""

    rejection_code = 'PLANNING_FAILED'

    def __init__(self, message, planning_diagnostics=None):
        super().__init__(message)
        self.planning_diagnostics = dict(planning_diagnostics or {})


class CuroboCandidateExhausted(CuroboPlanningError):
    """All candidates in one bounded shortlist were attempted and failed."""

    rejection_code = 'PLANNER_EXHAUSTED'


class CuroboPlanningBudgetExceeded(CuroboPlanningError):
    """One request consumed its immutable worker planning budget."""

    rejection_code = 'PLANNER_TIMEOUT'


class CuroboCollisionRejected(CuroboPlanningError):
    """cuRobo rejected a start, goal, or path on collision evidence."""

    rejection_code = 'PLANNING_COLLISION_REJECTED'


class CuroboOutputInvalid(CuroboPlanningError):
    """A native result could not satisfy the common execution contract."""

    rejection_code = 'PLANNING_FAILED'


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


def target_ray_standoff_samples(candidate, maximum_samples=9):
    """Return deterministic, capability-bounded standoffs for one target ray.

    The representative point remains first for continuity with the existing
    bridge contract.  Remaining samples fill the largest uncovered supported
    interval, so cuRobo searches the ray without probing unsupported gaps.
    """
    if candidate.get('candidate_geometry') != 'target_ray':
        raise CuroboContractError('candidate is not target-ray geometry')
    limit = int(maximum_samples)
    if limit < 1:
        raise CuroboContractError('target-ray sample limit must be positive')
    minimum = float(candidate.get('ray_min_standoff_m'))
    maximum = float(candidate.get('ray_max_standoff_m'))
    if (
            not math.isfinite(minimum) or not math.isfinite(maximum)
            or minimum <= 0.0 or maximum < minimum):
        raise CuroboContractError('target-ray standoff bounds are invalid')
    raw_intervals = candidate.get('ray_capability_intervals_m')
    intervals = []
    if raw_intervals is not None:
        if not isinstance(raw_intervals, (list, tuple)) or not raw_intervals:
            raise CuroboContractError(
                'target-ray capability intervals are invalid')
        for raw in raw_intervals:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise CuroboContractError(
                    'target-ray capability interval is invalid')
            lower, upper = float(raw[0]), float(raw[1])
            if (
                    not math.isfinite(lower) or not math.isfinite(upper)
                    or lower < minimum - 1e-9
                    or upper > maximum + 1e-9 or upper < lower):
                raise CuroboContractError(
                    'target-ray capability interval is invalid')
            intervals.append((max(minimum, lower), min(maximum, upper)))
    if not intervals:
        intervals = [(minimum, maximum)]

    def supported(value):
        return any(
            lower - 1e-9 <= value <= upper + 1e-9
            for lower, upper in intervals)

    samples = []

    def add(value):
        value = float(value)
        if (
                math.isfinite(value) and supported(value)
                and all(abs(value - existing) > 1e-9 for existing in samples)):
            samples.append(value)

    for key in (
            'ray_standoff_m', 'ray_scoring_standoff_m',
            'ray_preferred_max_standoff_m'):
        if candidate.get(key) is not None:
            add(candidate[key])
    for lower, upper in intervals:
        add(lower)
        add(upper)
        add((lower + upper) * 0.5)

    samples = samples[:limit]
    while len(samples) < limit:
        best = None
        for interval_index, (lower, upper) in enumerate(intervals):
            inside = sorted(
                value for value in samples
                if lower - 1e-9 <= value <= upper + 1e-9)
            boundaries = [lower] + inside + [upper]
            for left, right in zip(boundaries, boundaries[1:]):
                gap = right - left
                candidate_value = (left + right) * 0.5
                key = (gap, -interval_index, -candidate_value)
                if gap > 1e-9 and (best is None or key > best[0]):
                    best = (key, candidate_value)
        if best is None:
            break
        add(best[1])
    return tuple(samples)


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
        if candidate.get('candidate_geometry') == 'target_ray':
            if plan_kind != 'MULTIVIEW_SCAN':
                raise CuroboContractError(
                    'target-ray candidate is scan-only')
            ray = finite_vector(
                candidate.get('ray_direction'), 3,
                'candidate %d ray direction' % index)
            ray_norm = math.sqrt(sum(value * value for value in ray))
            try:
                ray_id = int(candidate.get('ray_id', -1))
                standoff = float(candidate.get('ray_standoff_m'))
            except (TypeError, ValueError):
                raise CuroboContractError(
                    'candidate %d target-ray fields are malformed' % index)
            if ray_id < 0 or abs(ray_norm - 1.0) > 1e-6:
                raise CuroboContractError(
                    'candidate %d target-ray fields are invalid' % index)
            samples = target_ray_standoff_samples(candidate)
            if not any(abs(standoff - value) <= 1e-9 for value in samples):
                raise CuroboContractError(
                    'candidate %d representative standoff is unsupported'
                    % index)
            target = finite_vector(
                scene.get('target_center_m'), 3, 'target center')
            expected = [
                origin + axis * standoff
                for origin, axis in zip(target, ray)]
            actual = finite_vector(
                candidate.get('camera_position_m'), 3,
                'candidate %d camera position' % index)
            if any(abs(left - right) > 1e-6 for left, right in zip(
                    actual, expected)):
                raise CuroboContractError(
                    'candidate %d target-ray representative is inconsistent'
                    % index)
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


def validate_trajectory_position_limits(positions, position_limits):
    """Reject planner-native points outside the authoritative raw limits."""
    limits = [
        finite_vector(bounds, 2, 'trajectory joint position limit')
        for bounds in position_limits
    ]
    if len(limits) != len(JOINT_NAMES):
        raise CuroboContractError(
            'trajectory position limits must contain six ranges')
    for joint_index, (low, high) in enumerate(limits):
        if low >= high:
            raise CuroboContractError(
                'trajectory joint%d position limit is empty'
                % (joint_index + 1))
    for point_index, row in enumerate(positions):
        point = finite_vector(row, 6, 'trajectory position')
        for joint_index, (value, bounds) in enumerate(zip(point, limits)):
            low, high = bounds
            if value < low or value > high:
                excess = max(low - value, value - high)
                raise CuroboContractError(
                    'cuRobo trajectory point %d joint%d=%.12g is outside '
                    '[%.12g, %.12g] by %.12g rad'
                    % (
                        point_index, joint_index + 1, value,
                        low, high, excess))


def position_limit_reentry_path(
        start_positions, position_limits, position_limit_clip,
        maximum_clip_excursion_rad=START_STATE_MAXIMUM_CLIP_EXCURSION_RAD,
        interior_offset_rad=START_STATE_REENTRY_INTERIOR_OFFSET_RAD,
        maximum_step_rad=START_STATE_REENTRY_MAXIMUM_STEP_RAD):
    """Return a bounded path from raw-valid feedback into planner limits.

    cuRobo deliberately clips selected PiPER joint limits so planned motion
    remains away from the physical boundary.  Controller feedback can settle
    a few encoder counts outside that *planner* inset while remaining inside
    the authoritative raw limit.  Such a state must not be passed to
    MotionGen, and it must not become a general limit bypass.  This helper
    therefore accepts only a small excursion beyond the clipped interval and
    moves monotonically to a point just inside it.  Collision qualification is
    owned by the worker before this path can enter a MotionPlan.

    An empty tuple means the measured state already satisfies the clipped
    planner interval.
    """
    start = finite_vector(
        start_positions, len(JOINT_NAMES), 'start positions')
    if not isinstance(position_limits, (list, tuple)) or \
            len(position_limits) != len(JOINT_NAMES):
        raise CuroboContractError(
            'position-limit re-entry requires six raw limit ranges')
    clips = finite_vector(
        position_limit_clip, len(JOINT_NAMES), 'position-limit clip')
    maximum_excursion = float(maximum_clip_excursion_rad)
    interior_offset = float(interior_offset_rad)
    maximum_step = float(maximum_step_rad)
    if (
            not math.isfinite(maximum_excursion) or maximum_excursion < 0.0
            or not math.isfinite(interior_offset) or interior_offset <= 0.0
            or not math.isfinite(maximum_step) or maximum_step <= 0.0):
        raise CuroboContractError(
            'position-limit re-entry policy is invalid')

    target = list(start)
    changed = []
    for joint_index, (position, raw_bounds, clip) in enumerate(zip(
            start, position_limits, clips)):
        low, high = finite_vector(
            raw_bounds, 2, 'joint position limit')
        if low >= high or clip < 0.0:
            raise CuroboContractError(
                'position-limit re-entry joint%d policy is invalid'
                % (joint_index + 1))
        clipped_low = low + clip
        clipped_high = high - clip
        if clipped_low + interior_offset >= clipped_high - interior_offset:
            raise CuroboContractError(
                'position-limit re-entry joint%d interval is empty'
                % (joint_index + 1))
        if position < low or position > high:
            raise CuroboContractError(
                'position-limit re-entry joint%d is outside the raw limit'
                % (joint_index + 1))
        if position < clipped_low:
            excursion = clipped_low - position
            if excursion > maximum_excursion + 1e-12:
                raise CuroboContractError(
                    'position-limit re-entry joint%d exceeds the bounded '
                    'feedback excursion' % (joint_index + 1))
            target[joint_index] = clipped_low + interior_offset
            changed.append(joint_index)
        elif position > clipped_high:
            excursion = position - clipped_high
            if excursion > maximum_excursion + 1e-12:
                raise CuroboContractError(
                    'position-limit re-entry joint%d exceeds the bounded '
                    'feedback excursion' % (joint_index + 1))
            target[joint_index] = clipped_high - interior_offset
            changed.append(joint_index)

    if not changed:
        return ()
    maximum_delta = max(
        abs(target[index] - start[index]) for index in changed)
    sample_count = max(
        1, int(math.ceil(maximum_delta / maximum_step - 1e-12)))
    return tuple([
        float(left + (right - left) * float(sample) / float(sample_count))
        for left, right in zip(start, target)
    ] for sample in range(sample_count + 1))


def normalize_trajectory(
        positions, native_dt_sec, speed_percent, position_limits=None):
    """Preserve native vertices in a fixed-rate PiPER MoveJ schedule.

    cuRobo supplies collision-checked path geometry at ``native_dt_sec``.
    PiPER's JointCtrl boundary accepts positions rather than trajectory time,
    velocity, or acceleration fields, so each native segment is sampled at the
    common 20 Hz transport rate.  Whole ticks are allocated independently per
    segment from the native duration, the speed-scaled MoveJ velocity model,
    and the hard adjacent-command step ceiling.  This retains every native
    vertex and never shortcuts a collision-checked corner.
    """
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
    if position_limits is not None:
        validate_trajectory_position_limits(rows, position_limits)
    period = 1.0 / COMMAND_RATE_HZ
    scaled_velocity = tuple(
        value * speed / 100.0 for value in MOVEJ_NOMINAL_VELOCITY_RAD_S)
    result = [rows[0]]
    for left, right in zip(rows, rows[1:]):
        deltas = tuple(abs(b - a) for a, b in zip(left, right))
        ticks_for_native_time = int(math.ceil(
            native_dt / period - 1e-12))
        ticks_for_velocity = int(math.ceil(max(
            delta / (limit * period)
            for delta, limit in zip(deltas, scaled_velocity)) - 1e-12))
        ticks_for_step = int(math.ceil(
            max(deltas) / MAXIMUM_JOINT_STEP_RAD - 1e-12))
        ticks = max(
            1,
            ticks_for_native_time,
            ticks_for_velocity,
            ticks_for_step,
        )
        if len(result) + ticks > MAXIMUM_SCHEDULED_POINTS:
            raise CuroboContractError(
                'scheduled cuRobo path exceeds point limit')
        for tick in range(1, ticks + 1):
            fraction = float(tick) / float(ticks)
            point = tuple(
                a + (b - a) * fraction for a, b in zip(left, right))
            result.append(point)
    zero = [0.0] * 6
    scheduled = [{
        'positions_rad': list(point),
        'velocities_rad_s': list(zero),
        'accelerations_rad_s2': list(zero),
        'time_from_start_s': round(index * period, 9),
    } for index, point in enumerate(result)]
    if position_limits is not None:
        validate_trajectory_position_limits(
            [point['positions_rad'] for point in scheduled], position_limits)
    return scheduled


def _prepend_start_transition(points, transition_positions, speed_percent):
    """Prepend a validated transition without altering native vertices."""
    planned = [dict(point) for point in points]
    recovery = normalize_trajectory(
        transition_positions, 1.0 / COMMAND_RATE_HZ, speed_percent)
    if any(
            abs(float(left) - float(right)) > 1e-6
            for left, right in zip(
                recovery[-1]['positions_rad'], planned[0]['positions_rad'])):
        raise CuroboContractError(
            'bootstrap recovery endpoint does not match planned path start')
    offset = float(recovery[-1]['time_from_start_s'])
    for point in planned[1:]:
        point['time_from_start_s'] = round(
            offset + float(point['time_from_start_s']), 9)
    return recovery + planned[1:], len(recovery) - 1


def prepend_bootstrap_recovery(points, recovery_positions, speed_percent):
    """Prepend a bounded folded-start escape to the native path."""
    return _prepend_start_transition(
        points, recovery_positions, speed_percent)


def prepend_position_limit_reentry(points, reentry_positions, speed_percent):
    """Prepend a collision-qualified internal-limit re-entry path."""
    return _prepend_start_transition(
        points, reentry_positions, speed_percent)


def trajectory_segment(points, is_return_home=False, bootstrap_recovery=None):
    """Build generic collision-qualified segment metadata."""
    segment = {
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
    if bootstrap_recovery is not None:
        joints = [int(value) for value in bootstrap_recovery['joint_numbers']]
        deltas = [float(value) for value in bootstrap_recovery['delta_rad']]
        segment.update({
            'bootstrap_recovery_used': True,
            'bootstrap_recovery_end_point': int(
                bootstrap_recovery['end_point']),
            'bootstrap_recovery_joint': joints[0] if len(joints) == 1 else 0,
            'bootstrap_recovery_delta_rad': (
                deltas[0] if len(deltas) == 1 else 0.0),
            'bootstrap_recovery_joints': joints,
            'bootstrap_recovery_deltas_rad': deltas,
            'bootstrap_recovery_minimum_clearance_m': -1.0,
            'bootstrap_recovery_limiting_link_pair':
                'curobo_start_state_self_collision_proxy',
            'bootstrap_recovery_samples': int(
                bootstrap_recovery['end_point']) + 1,
            'bootstrap_start_contacts': [],
            'validation':
                'curobo_v0.7.8_bounded_bootstrap_then_motiongen',
        })
    return segment


def worker_rejection_code(error):
    """Map predictable failures to stable fail-closed reason codes."""
    explicit = str(getattr(error, 'rejection_code', ''))
    if explicit:
        return explicit
    text = str(error).lower()
    if 'unsupported' in text:
        return 'PLANNER_UNSUPPORTED'
    if 'normalized output failed' in text or 'trajectory point' in text:
        return 'PLANNING_FAILED'
    if 'collision' in text or 'valid query' in text:
        return 'PLANNING_COLLISION_REJECTED'
    if 'cuda' in text or 'backend unavailable' in text or 'model' in text:
        return 'PLANNER_UNAVAILABLE'
    return 'PLANNING_FAILED'
