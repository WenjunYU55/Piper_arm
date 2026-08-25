"""Shared schema-v5 constants and validation primitives."""

import math
import re

from piper_tesseract_foxy.contract_core import ContractError
from piper_tesseract_foxy.contract_hashing import sha256_value


# The aim provenance fields are additive.  Retain schema 5 so a bridge and
# worker can be rolled independently without turning the repair into a flag
# day for the private transport contract.
SCHEMA_VERSION = 5
MAX_FINAL_AIM_OFFSET_DEG = 5.0
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
TIMING_POLICY = 'tesseract_stream_v3'
COMMAND_RATE_HZ = 20.0
MOVEJ_NOMINAL_VELOCITY_RAD_S = (5.0, 5.0, 5.0, 5.0, 5.0, 3.0)
MAX_PROTOCOL_VELOCITY_RAD_S = 3.0
MAX_PROTOCOL_ACCELERATION_RAD_S2 = 5.0
MAX_BOOTSTRAP_START_LIMIT_TOLERANCE_RAD = 0.04
MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD = 0.3
PLAN_KINDS = ('MULTIVIEW_SCAN', 'ROUGH_ACQUISITION', 'RETURN_HOME')
PROVENANCE_SOURCES = ('tracked_target', 'rough_coordinate', 'configured_home')
SCENE_OBSERVATION_MODES = ('perception_snapshot', 'bootstrap_static')
SAFE_ID = re.compile(r'^[a-f0-9]{16,64}$')
SOURCE_REQUEST_ID = re.compile(r'^[A-Za-z0-9_.:-]{8,128}$')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_CANDIDATE_VIEWS = 100
MAX_OBSTACLES = 256
MAX_CAPTURE_VIEWPOINTS = 13
MAX_SEGMENTS = MAX_CAPTURE_VIEWPOINTS + 1
MAX_POINTS_PER_SEGMENT = 60000
QUEUE_NAMES = ('requests', 'processing', 'responses', 'failed')
HEALTH_FILENAME = 'worker_health.json'
MAX_HEALTH_BYTES = 16 * 1024


def finite_vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ContractError('%s must contain %d values' % (label, length))
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ContractError('%s contains a boolean' % label)
        number = float(item)
        if not math.isfinite(number):
            raise ContractError('%s contains a non-finite value' % label)
        result.append(number)
    return result


def angular_separation_deg(first, second):
    """Return a finite angular separation for two validated directions."""
    left = finite_vector(first, 3, 'first direction')
    right = finite_vector(second, 3, 'second direction')
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if min(left_norm, right_norm) <= 1e-12:
        raise ContractError('aim direction must be non-zero')
    cosine = sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def target_ray_position_matches(candidate, selected, target_center):
    """Validate that a selected endpoint lies on its requested ray interval."""
    target = finite_vector(target_center, 3, 'target-ray center')
    direction = finite_vector(
        candidate.get('ray_direction'), 3, 'target-ray direction')
    norm = math.sqrt(sum(value * value for value in direction))
    if norm <= 1e-12:
        raise ContractError('target-ray direction must be non-zero')
    direction = [value / norm for value in direction]
    position = finite_vector(
        selected.get('camera_position_m'), 3,
        'selected target-ray camera position')
    relative = [value - origin for value, origin in zip(position, target)]
    standoff = sum(value * axis for value, axis in zip(relative, direction))
    residual = math.sqrt(sum(
        (value - standoff * axis) ** 2
        for value, axis in zip(relative, direction)))
    minimum = float(candidate.get('ray_min_standoff_m'))
    maximum = float(candidate.get('ray_max_standoff_m'))
    reported = float(selected.get('ray_standoff_m', standoff))
    return bool(
        math.isfinite(standoff)
        and math.isfinite(reported)
        and minimum - 1e-9 <= standoff <= maximum + 1e-9
        and abs(reported - standoff) <= 1e-6
        and residual <= 1e-6)


def require_sha256(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[a-f0-9]{64}', value) is None:
        raise ContractError('%s must be a lowercase SHA-256 digest' % label)
    return value


def motion_limits_digest(velocities, accelerations):
    """Return the canonical controller-limit digest also published by the driver."""
    return sha256_value({
        'joint_names': list(JOINT_NAMES),
        'max_velocity_rad_s': [
            round(float(value), 9) for value in velocities],
        'max_acceleration_rad_s2': [
            round(float(value), 9) for value in accelerations],
        'source': 'piper_sdk_controller_feedback',
    })


def validate_motion_limits(limits):
    """Validate controller-derived position, velocity, and acceleration limits."""
    velocities = finite_vector(
        limits.get('max_velocity_rad_s'), 6,
        'limits.max_velocity_rad_s')
    accelerations = finite_vector(
        limits.get('max_acceleration_rad_s2'), 6,
        'limits.max_acceleration_rad_s2')
    if any(
            value <= 0.0 or value > MAX_PROTOCOL_VELOCITY_RAD_S
            for value in velocities):
        raise ContractError('controller velocity limits are invalid')
    if any(
            value <= 0.0 or value > MAX_PROTOCOL_ACCELERATION_RAD_S2
            for value in accelerations):
        raise ContractError('controller acceleration limits are invalid')
    expected = require_sha256(
        limits.get('motion_limits_sha256'),
        'limits.motion_limits_sha256')
    if motion_limits_digest(velocities, accelerations) != expected:
        raise ContractError('limits.motion_limits_sha256 does not match values')
    if limits.get('source') != 'piper_sdk_controller_feedback':
        raise ContractError('limits.source is unsupported')
    return velocities, accelerations


def validate_plan_identity(payload):
    plan_kind = payload.get('plan_kind')
    if plan_kind not in PLAN_KINDS:
        raise ContractError('plan_kind is unsupported')
    provenance = payload.get('target_provenance')
    if not isinstance(provenance, dict):
        raise ContractError('target_provenance must be an object')
    source = provenance.get('source')
    expected_source = {
        'MULTIVIEW_SCAN': 'tracked_target',
        'ROUGH_ACQUISITION': 'rough_coordinate',
        'RETURN_HOME': 'configured_home',
    }[plan_kind]
    if source != expected_source or source not in PROVENANCE_SOURCES:
        raise ContractError(
            'target_provenance.source does not match plan_kind')
    source_request_id = provenance.get('source_request_id', '')
    if plan_kind == 'ROUGH_ACQUISITION':
        if (
                not isinstance(source_request_id, str)
                or SOURCE_REQUEST_ID.fullmatch(source_request_id) is None):
            raise ContractError(
                'rough-coordinate source_request_id is missing or invalid')
    elif source_request_id not in ('', None):
        raise ContractError(
            'non-acquisition provenance must not carry source_request_id')
    frame_id = provenance.get('frame_id')
    if frame_id != 'base_link':
        raise ContractError('target_provenance.frame_id must be base_link')
    stamp = provenance.get('stamp')
    if not isinstance(stamp, dict):
        raise ContractError('target_provenance.stamp must be an object')
    sec = stamp.get('sec')
    nanosec = stamp.get('nanosec')
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        raise ContractError('target_provenance.stamp contains a boolean')
    try:
        sec = int(sec)
        nanosec = int(nanosec)
    except (TypeError, ValueError):
        raise ContractError('target_provenance.stamp is invalid')
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ContractError('target_provenance.stamp is invalid')
    return plan_kind, provenance


def trajectory_digest(segments, binding):
    return sha256_value({
        'joint_names': JOINT_NAMES,
        'segments': segments,
        'binding': binding,
    })
