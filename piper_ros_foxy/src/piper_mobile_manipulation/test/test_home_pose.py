import json
from array import array

import pytest

from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    save_home_pose,
    staged_home_targets,
    validate_home_profile_limits,
    validate_home_positions,
    validate_staged_wrist_direction,
)


def test_home_pose_round_trip_is_hashed_and_permission_bounded(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    saved = save_home_pose(path, positions)
    assert load_home_pose(path)['positions_rad'] == positions
    assert load_home_pose(path)['staged_home_configured'] is False
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


def test_home_pose_accepts_ros_joint_state_array_sequence():
    positions = array('d', [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert validate_home_positions(positions) == list(positions)


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


def test_staged_home_profile_preserves_other_joints_for_wrist_moves(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    rough = [0.1, 0.2, -0.3, 0.4, 0.5, 1.2]
    save_home_pose(
        path, rough, storage_joint6_rad=-0.5,
        mission_ready_joint6_rad=1.2)
    profile = load_home_pose(path)
    assert profile['staged_home_configured'] is True
    targets = staged_home_targets(
        profile, [-0.4, 0.3, -0.2, 0.1, 0.6, -0.5])
    assert targets['startup_wrist_positions_rad'] == [
        -0.4, 0.3, -0.2, 0.1, 0.6, 1.2]
    assert targets['rough_home_positions_rad'] == rough
    assert targets['storage_wrist_positions_rad'] == [
        0.1, 0.2, -0.3, 0.4, 0.5, -0.5]


def test_staged_home_rejects_wrong_signed_startup_branch(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    save_home_pose(
        path, [0.0] * 6, storage_joint6_rad=-3.13,
        mission_ready_joint6_rad=0.0)
    profile = load_home_pose(path)
    assert validate_staged_wrist_direction(
        profile, [0.0, 0.0, 0.0, 0.0, 0.4, -3.10]) is profile
    with pytest.raises(ValueError, match='positive storage branch'):
        validate_staged_wrist_direction(
            profile, [0.0, 0.0, 0.0, 0.0, 0.4, 3.10])


def test_staged_home_rejects_positive_storage_profile(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    with pytest.raises(ValueError, match='negative branch'):
        save_home_pose(
            path, [0.0] * 6, storage_joint6_rad=3.13,
            mission_ready_joint6_rad=0.0)


def test_staged_home_profile_rejects_ready_mismatch(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    saved = save_home_pose(
        path, [0.0] * 6, storage_joint6_rad=-2.0,
        mission_ready_joint6_rad=1.0)
    saved['positions_rad'][5] = 0.0
    unsigned = dict(saved)
    unsigned.pop('home_pose_sha256')
    from piper_mobile_manipulation.home_pose import payload_sha256
    saved['home_pose_sha256'] = payload_sha256(unsigned)
    path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        load_home_pose(path)


def test_staged_home_profile_is_checked_against_supplied_model_limits(tmp_path):
    path = tmp_path / 'piper_home_pose.json'
    limits = [[-1.0, 1.0] for _ in range(6)]
    save_home_pose(
        path, [0.0] * 6, storage_joint6_rad=-0.999,
        mission_ready_joint6_rad=0.0)
    profile = load_home_pose(path)
    assert validate_home_profile_limits(profile, limits) is profile

    profile['storage_joint6_rad'] = -1.001
    with pytest.raises(ValueError, match='storage joint6'):
        validate_home_profile_limits(profile, limits)

    profile['storage_joint6_rad'] = 0.0
    profile['positions_rad'][2] = 1.001
    with pytest.raises(ValueError, match='rough home joint3'):
        validate_home_profile_limits(profile, limits)
