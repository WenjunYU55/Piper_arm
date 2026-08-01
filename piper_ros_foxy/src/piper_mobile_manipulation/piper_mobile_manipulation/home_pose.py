"""Validated, atomic persistence for the operator-selected scan home pose."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


HOME_POSE_SCHEMA_VERSION = 1


def validate_home_positions(values):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
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


def save_home_pose(path, positions, observed_positions=None):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        'schema_version': HOME_POSE_SCHEMA_VERSION,
        'joint_names': [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        'positions_rad': validate_home_positions(positions),
        'saved_at_sec': time.time(),
    }
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
    if int(payload.get('schema_version', 0)) != HOME_POSE_SCHEMA_VERSION:
        raise ValueError('home pose schema version is unsupported')
    positions = validate_home_positions(payload.get('positions_rad'))
    result = dict(payload)
    result['positions_rad'] = positions
    if 'observed_disabled_positions_rad' in payload:
        result['observed_disabled_positions_rad'] = validate_home_positions(
            payload.get('observed_disabled_positions_rad'))
    return result
