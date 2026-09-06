"""Endpoint filtering uses loaded rigid geometry, without IK or robot commands."""

from types import SimpleNamespace

import numpy as np
import pytest

from motion_planning.curobo.worker import CuroboBackend
from motion_planning.curobo.adapter import CuroboCandidateExhausted


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def backend():
    value = object.__new__(CuroboBackend)
    value.tensor_args = SimpleNamespace(to_device=lambda x: x)
    value.floor_z_m = 0.0
    value.collision_manifest = {'external_floor_clearance': {
        'enabled': True, 'floor_z_m': 0.0, 'clearance_m': 0.005,
        'origin_link6_m': [0, 0, 0], 'size_m': [0.02, 0.02, 0.02],
    }}
    # Camera is rigidly 0.1 m above link6; both frames have identity rotation.
    value.motion_gen = SimpleNamespace(kinematics=SimpleNamespace(
        get_state=lambda q: SimpleNamespace(
            ee_position=Array([[0, 0, 0.3]]),
            ee_quaternion=Array([[1, 0, 0, 0]])),
        get_link_poses=lambda q, names: SimpleNamespace(
            position=Array([[0, 0, 0.2]]),
            quaternion=Array([[1, 0, 0, 0]]))))
    return value


def goal(z):
    return {'camera_position_m': [0, 0, z],
            'look_direction': [0, 0, 1], 'roll_rad': 0.0}


def test_holder_floor_precheck_uses_camera_to_link_offset():
    value = backend()
    goals = [goal(0.05), goal(0.3)]
    request = {'plan_kind': 'ROUGH_ACQUISITION', 'scene': {}}
    assert value._endpoint_precheck([0]*6, goals, request, {}) == [goals[1]]


def test_endpoint_obstacle_rejected_before_native_planning():
    value = backend()
    request = {'plan_kind': 'ROUGH_ACQUISITION', 'scene': {'obstacles': [{
        'type': 'box', 'id': 'block', 'minimum_m': [-0.02, -0.02, 0.18],
        'maximum_m': [0.02, 0.02, 0.22]}]}}
    assert value._endpoint_precheck([0]*6, [goal(0.3)], request, {}) == []
    # No MotionGen method is present on the stub: an accidental call fails.


@pytest.mark.parametrize('target,accepted', [([0, 0, 0.4], 1),
                                             ([0.4, 0, 0.7], 0)])
def test_endpoint_distance_and_initial_aim_rejected(target, accepted):
    value = backend()
    request = {'plan_kind': 'MULTIVIEW_SCAN',
               'scan_session': {'accepted_views': accepted},
               'scene': {'target_center_m': target}}
    assert value._endpoint_precheck([0]*6, [goal(0.3)], request, {}) == []


def test_precheck_keeps_valid_endpoint_and_full_path_checks_still_run():
    value = backend()
    request = {'plan_kind': 'MULTIVIEW_SCAN',
               'scan_session': {'accepted_views': 1},
               'scene': {'target_center_m': [0, 0, 0.7]}}
    goals = [goal(0.3)]
    assert value._endpoint_precheck([0]*6, goals, request, {}) == goals
    value._attached_tool_external_rejection = lambda *a: 'midpath floor collision'
    assert value._candidate_path_rejection([], request, {}) == 'midpath floor collision'


def test_rigid_projection_matches_known_rotated_camera_pose():
    value = backend()
    observed = []
    value._external_attached_validation = lambda *a, **kw: observed.append(
        kw['link_transforms'])
    g = {'camera_position_m': [0.4, 0, 0.3],
         'look_direction': [1, 0, 0], 'roll_rad': 0.0}
    value._endpoint_precheck([0]*6, [g], {'plan_kind': 'ROUGH_ACQUISITION'}, {})
    # Camera's local -Z offset becomes world -X under the requested rotation.
    assert observed[0][0][0] == pytest.approx([0.3, 0, 0.3])


def test_nonfinite_loaded_fk_fails_closed():
    value = backend()
    value.motion_gen.kinematics.get_link_poses = lambda *a: SimpleNamespace(
        position=Array([[float('nan'), 0, 0]]),
        quaternion=Array([[1, 0, 0, 0]]))
    with pytest.raises(ValueError, match='transforms are invalid'):
        value._endpoint_precheck([0]*6, [goal(0.3)],
                                 {'plan_kind': 'ROUGH_ACQUISITION'}, {})


def test_all_invalid_endpoints_never_invoke_motiongen():
    value = backend()
    candidate = dict(goal(0.05), id=7)
    request = {'plan_kind': 'ROUGH_ACQUISITION',
               'planning': {'effective_speed_percent': 5,
                            'roll_samples_rad': [0.0]}, 'scene': {}}
    with pytest.raises(ValueError, match='no collision-safe'):
        value._plan_pose([0]*6, candidate, request)


def test_failed_ray_is_reported_even_when_next_ray_succeeds():
    value = backend()
    value._update_world = lambda r: None
    value._position_limit_reentry = lambda *a: None
    value._bootstrap_recovery = lambda *a: None

    def plan(start, candidate, request):
        if candidate['ray_id'] == 1:
            raise CuroboCandidateExhausted('all endpoint choices exhausted')
        return candidate, [
            {'positions_rad': [0]*6, 'time_from_start_s': 0.0},
            {'positions_rad': [0.01]*6, 'time_from_start_s': 0.05}], 0.1

    value._plan_pose = plan
    request = {'plan_kind': 'MULTIVIEW_SCAN',
               'start_state': {'positions_rad': [0]*6},
               'planning': {'min_viewpoints': 1, 'max_viewpoints': 1,
                            'include_return_home': False},
               'scene': {'candidate_views': [{'id': 1, 'ray_id': 1},
                                             {'id': 2, 'ray_id': 2}]}}
    selected, _, _ = value.plan(request)
    assert selected[0]['ray_id'] == 2
    assert value.last_planning_diagnostics['exhausted_ray_ids'] == [1]
