import json

import pytest

from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    save_home_pose,
    validate_home_positions,
)


def test_home_pose_round_trip_is_hashed_and_permission_bounded(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    saved = save_home_pose(path, positions)
    assert load_home_pose(path)['positions_rad'] == positions
    assert len(saved['home_pose_sha256']) == 64
    assert path.stat().st_mode & 0o777 == 0o600


def test_home_pose_rejects_wrong_shape_nonfinite_and_tampering(tmp_path):
    with pytest.raises(ValueError, match='exactly six'):
        validate_home_positions([0.0] * 5)
    with pytest.raises(ValueError, match='non-finite'):
        validate_home_positions([0.0] * 5 + [float('nan')])
    path = tmp_path / 'piper_home_pose.json'
    save_home_pose(path, [0.0] * 6)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['positions_rad'][0] = 1.0
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        load_home_pose(path)


def test_absent_home_pose_uses_caller_default(tmp_path):
    assert load_home_pose(tmp_path / 'missing.json') is None


def test_home_pose_preserves_optional_disabled_observation(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    powered = [0.1, 0.0, 0.0, -0.04, 0.44, 0.0]
    observed = [0.1, -0.029, 0.006, -0.04, 0.44, 0.0]
    save_home_pose(path, powered, observed_positions=observed)
    loaded = load_home_pose(path)
    assert loaded['positions_rad'] == powered
    assert loaded['observed_disabled_positions_rad'] == observed
