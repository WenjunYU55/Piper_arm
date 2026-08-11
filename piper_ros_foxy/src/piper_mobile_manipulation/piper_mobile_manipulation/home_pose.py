"""Validated, atomic persistence for the operator-selected scan home pose."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


HOME_POSE_SCHEMA_VERSION = 3
LEGACY_HOME_POSE_SCHEMA_VERSION = 1
UNDIRECTED_STAGED_HOME_POSE_SCHEMA_VERSION = 2
STARTUP_WRIST_DIRECTION = 'increasing'
STORAGE_WRIST_DIRECTION = 'decreasing'
WRIST_DIRECTION_READY_TOLERANCE_RAD = 0.030


def validate_home_positions(values):
    if (
            isinstance(values, (str, bytes, bytearray, dict))
            or not hasattr(values, '__len__')
            or not hasattr(values, '__iter__')
            or len(values) != 6):
        raise ValueError('home pose must contain exactly six joint positions')
    positions = [float(value) for value in values]
    if not all(math.isfinite(value) for value in positions):
        raise ValueError('home pose contains a non-finite joint position')
    return positions


def payload_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validate_joint6(value, label='joint6 angle'):
    angle = float(value)
    if not math.isfinite(angle):
        raise ValueError('%s is non-finite' % label)
    return angle


def validate_home_profile_limits(profile, joint_limits, tolerance_rad=0.0):
    """Reject a persisted staged profile outside the supplied model limits.

    The caller supplies the production limit authority so this persistence
    helper does not grow a second copy of the robot model.  Disabled-position
    observations are provenance only and are deliberately not motion targets.
    """
    if not isinstance(profile, dict):
        raise ValueError('home profile is missing')
    limits = list(joint_limits)
    if len(limits) != 6:
        raise ValueError('home joint limits must contain six pairs')
    tolerance = float(tolerance_rad)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('home joint-limit tolerance is invalid')
    parsed_limits = []
    for index, pair in enumerate(limits):
        if not isinstance(pair, (list, tuple)) and not hasattr(pair, '__len__'):
            raise ValueError('home joint%d limit is not a pair' % (index + 1))
        if len(pair) != 2:
            raise ValueError('home joint%d limit is not a pair' % (index + 1))
        low, high = float(pair[0]), float(pair[1])
        if not all(math.isfinite(value) for value in (low, high)) or low >= high:
            raise ValueError('home joint%d limit is invalid' % (index + 1))
        parsed_limits.append((low, high))

    positions = validate_home_positions(profile.get('positions_rad'))
    for index, value in enumerate(positions):
        low, high = parsed_limits[index]
        if value < low - tolerance or value > high + tolerance:
            raise ValueError(
                'rough home joint%d is outside planning limits' % (index + 1))
    for label, value in (
            ('mission-ready joint6', profile.get('mission_ready_joint6_rad')),
            ('storage joint6', profile.get('storage_joint6_rad'))):
        angle = validate_joint6(value, label)
        low, high = parsed_limits[5]
        if angle < low - tolerance or angle > high + tolerance:
            raise ValueError('%s is outside planning limits' % label)
    return profile


def validate_staged_wrist_direction(profile, current_positions=None):
    """Prove the configured and measured J6 branches select safe directions."""
    if not isinstance(profile, dict):
        raise ValueError('home profile is missing')
    if not bool(profile.get('staged_home_configured', False)):
        raise ValueError('staged home profile is not configured')
    startup_direction = str(profile.get('startup_wrist_direction', ''))
    storage_direction = str(profile.get('storage_wrist_direction', ''))
    if startup_direction != STARTUP_WRIST_DIRECTION:
        raise ValueError(
            'startup J6 direction must be increasing toward ready')
    if storage_direction != STORAGE_WRIST_DIRECTION:
        raise ValueError(
            'storage J6 direction must be decreasing toward storage')
    ready = validate_joint6(
        profile.get('mission_ready_joint6_rad'),
        'mission-ready joint6 angle')
    storage = validate_joint6(
        profile.get('storage_joint6_rad'), 'storage joint6 angle')
    if storage >= ready - WRIST_DIRECTION_READY_TOLERANCE_RAD:
        raise ValueError(
            'storage J6 must be on the negative branch below the ready angle '
            'for increasing startup')
    if current_positions is not None:
        current = validate_home_positions(current_positions)[5]
        if (
                abs(current - ready) >
                WRIST_DIRECTION_READY_TOLERANCE_RAD
                and current >= ready):
            raise ValueError(
                'measured J6 is on the positive storage branch; increasing '
                'startup requires the configured negative-J6 storage branch '
                'before enable')
    return profile


def staged_home_targets(profile, current_positions):
    """Return startup-wrist, rough-home and storage-wrist joint targets.

    The two wrist moves deliberately preserve joints 1-5.  The rough-home
    target remains a normal six-joint collision-planned pose, so J6 is back at
    the recorded mission-ready angle before the final storage rotation.
    """
    if not isinstance(profile, dict):
        raise ValueError('home profile is missing')
    current = validate_home_positions(current_positions)
    rough = validate_home_positions(profile.get('positions_rad'))
    ready = validate_joint6(
        profile.get('mission_ready_joint6_rad'),
        'mission-ready joint6 angle')
    storage = validate_joint6(
        profile.get('storage_joint6_rad'), 'storage joint6 angle')
    validate_staged_wrist_direction(profile, current)
    if abs(rough[5] - ready) > 1e-9:
        raise ValueError(
            'rough home joint6 must equal the mission-ready joint6 angle')
    startup_wrist = list(current)
    startup_wrist[5] = ready
    storage_wrist = list(rough)
    storage_wrist[5] = storage
    return {
        'startup_wrist_positions_rad': startup_wrist,
        'rough_home_positions_rad': rough,
        'storage_wrist_positions_rad': storage_wrist,
    }


def save_home_pose(
        path, positions, observed_positions=None,
        storage_joint6_rad=None, mission_ready_joint6_rad=None,
        staged_home_configured=None):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ready = validate_home_positions(positions)[5]
    if mission_ready_joint6_rad is not None:
        ready = validate_joint6(
            mission_ready_joint6_rad, 'mission-ready joint6 angle')
    positions = validate_home_positions(positions)
    positions[5] = ready
    storage = (
        ready if storage_joint6_rad is None else
        validate_joint6(storage_joint6_rad, 'storage joint6 angle'))
    configured = (
        bool(staged_home_configured)
        if staged_home_configured is not None
        else storage_joint6_rad is not None)
    payload = {
        'schema_version': HOME_POSE_SCHEMA_VERSION,
        'joint_names': [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        'positions_rad': positions,
        'mission_ready_joint6_rad': ready,
        'storage_joint6_rad': storage,
        'staged_home_configured': configured,
        'saved_at_sec': time.time(),
    }
    if configured:
        payload['startup_wrist_direction'] = STARTUP_WRIST_DIRECTION
        payload['storage_wrist_direction'] = STORAGE_WRIST_DIRECTION
        validate_staged_wrist_direction(payload)
    if observed_positions is not None:
        payload['observed_disabled_positions_rad'] = \
            validate_home_positions(observed_positions)
    payload['home_pose_sha256'] = payload_sha256(payload)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.piper_home_pose.', suffix='.tmp',
        dir=str(destination.parent))
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def load_home_pose(path):
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return None
    with source.open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError('home pose file is not an object')
    expected = str(payload.get('home_pose_sha256', ''))
    unsigned = dict(payload)
    unsigned.pop('home_pose_sha256', None)
    if expected != payload_sha256(unsigned):
        raise ValueError('home pose SHA-256 mismatch')
    schema = int(payload.get('schema_version', 0))
    if schema not in (
            LEGACY_HOME_POSE_SCHEMA_VERSION,
            UNDIRECTED_STAGED_HOME_POSE_SCHEMA_VERSION,
            HOME_POSE_SCHEMA_VERSION):
        raise ValueError('home pose schema version is unsupported')
    positions = validate_home_positions(payload.get('positions_rad'))
    result = dict(payload)
    result['positions_rad'] = positions
    if schema == LEGACY_HOME_POSE_SCHEMA_VERSION:
        # A legacy home remains readable for diagnostics and manual operation,
        # but cannot silently authorize the new staged powered sequence.
        result['mission_ready_joint6_rad'] = positions[5]
        result['storage_joint6_rad'] = positions[5]
        result['staged_home_configured'] = False
    else:
        result['mission_ready_joint6_rad'] = validate_joint6(
            payload.get('mission_ready_joint6_rad'),
            'mission-ready joint6 angle')
        result['storage_joint6_rad'] = validate_joint6(
            payload.get('storage_joint6_rad'), 'storage joint6 angle')
        if not isinstance(payload.get('staged_home_configured'), bool):
            raise ValueError('staged_home_configured must be boolean')
        result['staged_home_configured'] = bool(
            payload['staged_home_configured'])
        if abs(positions[5] - result['mission_ready_joint6_rad']) > 1e-9:
            raise ValueError(
                'rough home joint6 does not match mission-ready joint6')
        if schema == UNDIRECTED_STAGED_HOME_POSE_SCHEMA_VERSION:
            # Schema 2 did not bind the physical wrist direction. It remains
            # readable for diagnostics but cannot authorize powered startup.
            result['staged_home_configured'] = False
        elif result['staged_home_configured']:
            result['startup_wrist_direction'] = str(
                payload.get('startup_wrist_direction', ''))
            result['storage_wrist_direction'] = str(
                payload.get('storage_wrist_direction', ''))
            validate_staged_wrist_direction(result)
    if 'observed_disabled_positions_rad' in payload:
        result['observed_disabled_positions_rad'] = validate_home_positions(
            payload.get('observed_disabled_positions_rad'))
    return result
