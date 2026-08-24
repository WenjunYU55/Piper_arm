from pathlib import Path

import pytest
import yaml

from run_fixed_hand_eye_calibration import (
    FixedPoseRunner,
    JOINT_NAMES,
    calibration_home_targets,
    load_pose_file,
)


POSE_FILE = (
    Path(__file__).resolve().parent
    / 'calibration/hand_eye/fixed_calibration_poses.yaml'
)


def test_fixed_sequence_has_capture_split_and_neutral_transit():
    data = load_pose_file(POSE_FILE)
    assert sum(pose.get('group') == 'fitting' for pose in data['poses']) == 10
    assert sum(pose.get('group') == 'validation' for pose in data['poses']) == 2
    assert sum(not pose.get('capture', True) for pose in data['poses']) == 1
    assert JOINT_NAMES == [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def test_pose_loader_rejects_out_of_range_j6(tmp_path):
    data = yaml.safe_load(POSE_FILE.read_text())
    data['poses'][0]['positions_rad'][5] = 3.2
    candidate = tmp_path / 'poses.yaml'
    candidate.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match=r'J6 \+/-pi'):
        load_pose_file(candidate)


def test_home_targets_require_and_order_all_three_stages():
    profile = {
        'pre_home_configured': True,
        'staged_home_configured': True,
        'pre_home_positions_rad': [0.0, 0.4, -0.5, 0.0, 0.6, 0.0],
        'positions_rad': [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        'storage_joint6_rad': -3.12,
    }
    assert calibration_home_targets(profile) == (
        [0.0, 0.4, -0.5, 0.0, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.4, -3.12],
    )


def test_home_then_disable_never_disables_before_all_stages_settle():
    class FakeRunner:
        def __init__(self):
            self.events = []

        def motion_authority(self):
            return True, 'ready'

        def publish_target(self, target):
            self.events.append(('publish', list(target)))

        def wait_for_target(self, target, timeout_sec):
            self.events.append(('settled', list(target), timeout_sec))
            return True, 0.001

        def request_disable(self):
            self.events.append(('disable',))
            return True, 'disabled'

        def hold_measured(self):
            self.events.append(('hold',))

    profile = {
        'pre_home_configured': True,
        'staged_home_configured': True,
        'pre_home_positions_rad': [0.0, 0.4, -0.5, 0.0, 0.6, 0.0],
        'positions_rad': [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        'storage_joint6_rad': -3.12,
    }
    runner = FakeRunner()
    ok, _reason = FixedPoseRunner.home_then_disable(runner, profile, 90.0)
    assert ok is True
    published = [event[1] for event in runner.events if event[0] == 'publish']
    assert published == [
        profile['pre_home_positions_rad'],
        profile['positions_rad'],
        [0.0, 0.0, 0.0, 0.0, 0.4, -3.12],
        [0.0, 0.0, 0.0, 0.0, 0.4, -3.12],
    ]
    assert runner.events[-1] == ('disable',)


def test_failed_pre_home_holds_and_never_disables():
    class FakeRunner:
        def __init__(self):
            self.events = []

        def motion_authority(self):
            return True, 'ready'

        def publish_target(self, target):
            self.events.append(('publish', list(target)))

        def wait_for_target(self, target, timeout_sec):
            return False, 1.0

        def request_disable(self):
            self.events.append(('disable',))
            return True, 'disabled'

        def hold_measured(self):
            self.events.append(('hold',))

    profile = {
        'pre_home_configured': True,
        'staged_home_configured': True,
        'pre_home_positions_rad': [0.0, 0.4, -0.5, 0.0, 0.6, 0.0],
        'positions_rad': [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        'storage_joint6_rad': -3.12,
    }
    runner = FakeRunner()
    ok, reason = FixedPoseRunner.home_then_disable(runner, profile, 90.0)
    assert ok is False
    assert 'PRE_HOME did not reach' in reason
    assert ('hold',) in runner.events
    assert ('disable',) not in runner.events
