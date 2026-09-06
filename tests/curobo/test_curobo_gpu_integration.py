"""Opt-in real cuRobo/CUDA planner tests; these never command the robot."""

import copy
import json
import math
import os
from pathlib import Path

import pytest


RUN_GPU_TESTS = os.environ.get('PIPER_RUN_CUROBO_GPU_TESTS') == '1'
pytestmark = pytest.mark.skipif(
    not RUN_GPU_TESTS,
    reason='set PIPER_RUN_CUROBO_GPU_TESTS=1 in the pinned cuRobo environment',
)

if RUN_GPU_TESTS:
    from motion_planning.curobo import PINNED_VERSION
    from motion_planning.curobo.adapter import (
        attach_digest,
        CuroboContractError,
        validate_request,
    )
    from motion_planning.curobo.worker import CuroboBackend


FIXTURE_VARIABLES = {
    'MULTIVIEW_SCAN': 'PIPER_CUROBO_GPU_MULTIVIEW_REQUEST',
    'ROUGH_ACQUISITION': 'PIPER_CUROBO_GPU_ACQUISITION_REQUEST',
    'RETURN_HOME': 'PIPER_CUROBO_GPU_RETURN_HOME_REQUEST',
}


def _request(kind):
    variable = FIXTURE_VARIABLES[kind]
    path = Path(os.environ.get(variable, '')).expanduser().resolve()
    if not path.is_file():
        pytest.fail('%s must point to a frozen generic planner request' % variable)
    request = json.loads(path.read_text(encoding='utf-8'))
    return validate_request(request)


def test_rigid_endpoint_projection_matches_loaded_link_fk(backend, monkeypatch):
    import numpy as np
    from motion_planning.curobo.worker import (
        camera_quaternion, quaternion_rotation_matrices_wxyz)
    start = _request('MULTIVIEW_SCAN')['start_state']['positions_rad']
    observed = []
    monkeypatch.setattr(backend, '_external_attached_validation',
                        lambda *a, **kw: observed.append(kw['link_transforms']))
    for delta in (0.0, 0.1, -0.1):
        joints = list(start)
        joints[0] += delta
        tensor = backend.tensor_args.to_device([joints])
        state = backend.motion_gen.kinematics.get_state(tensor)
        position = state.ee_position.detach().cpu().numpy().reshape(-1, 3)[0]
        rotation = quaternion_rotation_matrices_wxyz(
            state.ee_quaternion.detach().cpu().numpy().reshape(-1, 4))[0]
        forward = rotation[:, 2]
        zero = quaternion_rotation_matrices_wxyz(np.asarray([
            camera_quaternion(forward, 0.0)]))[0]
        roll = math.atan2(np.dot(rotation[:, 0], zero[:, 1]),
                          np.dot(rotation[:, 0], zero[:, 0]))
        pose = backend.motion_gen.kinematics.get_link_poses(tensor, ['link6'])
        expected_position = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
        expected_rotation = quaternion_rotation_matrices_wxyz(
            pose.quaternion.detach().cpu().numpy().reshape(-1, 4))[0]
        backend._endpoint_precheck(start, [{
            'camera_position_m': position, 'look_direction': forward,
            'roll_rad': roll}], {'plan_kind': 'ROUGH_ACQUISITION'}, {})
        assert observed[-1][0][0] == pytest.approx(expected_position, abs=2e-6)
        assert observed[-1][1][0] == pytest.approx(expected_rotation, abs=2e-6)


def _assert_fixed_rate_schedule(segment):
    points = segment['points']
    assert len(points) >= 2
    assert all(
        right['time_from_start_s'] - left['time_from_start_s']
        == pytest.approx(0.05, abs=1e-9)
        for left, right in zip(points, points[1:]))


@pytest.mark.parametrize('rate', [20., 100., 200.])
def test_real_bounded_fallback_batch_preserves_aim_and_collision_checks(backend, monkeypatch, rate):
    """Inject only the nominal miss; fallback MotionGen and all checks are real."""
    import numpy as np
    from types import SimpleNamespace
    from motion_planning.curobo.worker import quaternion_rotation_matrices_wxyz
    start = [0.0, 0.8, -0.7, 0.0, 0.7, 0.0]
    backend._update_world({'scene': {'obstacles': []}})
    state = backend.motion_gen.kinematics.get_state(backend.tensor_args.to_device([start]))
    position = state.ee_position.detach().cpu().numpy().reshape(3)
    rotation = quaternion_rotation_matrices_wxyz(
        state.ee_quaternion.detach().cpu().numpy().reshape(1, 4))[0]
    forward = rotation[:, 2]
    # The known orientation is one explicitly requested alternative, not a
    # worker-invented result. Reconstruct its roll in the worker convention.
    from motion_planning.curobo.worker import camera_quaternion
    zero = quaternion_rotation_matrices_wxyz(
        np.asarray([camera_quaternion(forward, 0.0)]))[0]
    roll = float(np.arctan2(np.dot(rotation[:, 0], zero[:, 1]),
                            np.dot(rotation[:, 0], zero[:, 0])))
    nominal = forward * math.cos(math.radians(2)) + rotation[:, 0] * math.sin(math.radians(2))
    target = position + nominal * 0.4
    candidate = {'id': 1, 'ray_id': 1, 'candidate_geometry': 'target_ray',
                 'camera_position_m': position.tolist(), 'ray_direction': (-nominal).tolist(),
                 'look_direction': nominal.tolist(), 'ray_min_standoff_m': 0.4,
                 'ray_max_standoff_m': 0.4, 'ray_standoff_m': 0.4,
                 'maximum_final_aim_offset_deg': 5.0,
                 'fallback_look_directions': [forward.tolist(),
                     (forward * math.cos(0.01) + rotation[:, 1] * math.sin(0.01)).tolist()]}
    request = {'plan_kind': 'MULTIVIEW_SCAN', 'scan_session': {'accepted_views': 1},
               'scene': {'target_center_m': target.tolist(), 'obstacles': []},
               'planning': {'effective_speed_percent': 5.0, 'roll_samples_rad': [roll],
                            'command_rate_hz': rate},
               'limits': {'position_rad': [[-3.2, 3.2]] * 6}}
    native = backend.motion_gen.plan_goalset
    calls = []
    def nominal_miss_then_native(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(success=SimpleNamespace(item=lambda: False),
                                   status='IK_FAIL', total_time=0.0)
        return native(*args, **kwargs)
    monkeypatch.setattr(backend.motion_gen, 'plan_goalset', nominal_miss_then_native)
    selected, points, _duration = backend._plan_pose(start, candidate, request)
    assert len(calls) == 2
    assert selected['look_direction'] in candidate['fallback_look_directions']
    assert selected['aim_offset_deg'] < 4.0
    assert selected['curobo_attempt_diagnostics'][-1]['path_qualification'] == 'accepted'
    assert not backend._candidate_path_rejection(points, request, selected)


@pytest.fixture(scope='module')
def backend():
    config = Path(os.environ.get(
        'PIPER_CUROBO_ROBOT_CONFIG', '')).expanduser().resolve()
    if not config.is_file():
        pytest.fail(
            'PIPER_CUROBO_ROBOT_CONFIG must identify the generated PiPER model')
    value = CuroboBackend(
        config, float(os.environ.get('PIPER_CUROBO_FLOOR_Z_M', '0.005')))
    assert value.version.startswith(PINNED_VERSION)
    assert value.environment['cuda_available'] is True
    assert value.environment['gpu_name']
    return value


@pytest.mark.parametrize('kind', tuple(FIXTURE_VARIABLES))
def test_real_planner_supports_each_declared_plan_kind(backend, kind):
    selected, segments, duration = backend.plan(_request(kind))
    assert duration >= 0.0
    assert segments
    assert all(len(segment['points']) >= 2 for segment in segments)
    if kind == 'RETURN_HOME':
        assert selected == []
        assert segments[-1]['is_return_home'] is True
    else:
        assert selected
        for segment in segments:
            _assert_fixed_rate_schedule(segment)
    if kind == 'MULTIVIEW_SCAN':
        assert any(
            attempt.get('path_qualification') == 'accepted'
            for viewpoint in selected
            for attempt in viewpoint['curobo_attempt_diagnostics'])


@pytest.mark.parametrize('rate', [100., 200.])
def test_real_high_rate_acquisition_schedule(backend, rate):
    """Command-free real MotionGen output honors the requested transport rate."""
    request = copy.deepcopy(_request('ROUGH_ACQUISITION'))
    request['planning']['command_rate_hz'] = rate
    request = attach_digest(request, 'request_sha256')
    selected, segments, _ = backend.plan(request)
    assert selected and segments
    for segment in segments:
        points = segment['points']
        assert len(points) >= 2
        for left, right in zip(points, points[1:]):
            dt = right['time_from_start_s'] - left['time_from_start_s']
            assert dt == pytest.approx(1. / rate, abs=1e-9)
            assert max(abs(a - b) for a, b in zip(
                left['positions_rad'], right['positions_rad'])) <= .05 + 1e-9


def test_real_planner_rejects_a_deliberately_blocked_scene(backend):
    request = copy.deepcopy(_request('MULTIVIEW_SCAN'))
    request.pop('request_sha256', None)
    request['scene']['obstacles'].append({
        'id': 'gpu_test_blocking_volume',
        'type': 'box',
        'minimum_m': [-2.0, -2.0, -2.0],
        'maximum_m': [2.0, 2.0, 2.0],
    })
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(CuroboContractError):
        backend.plan(validate_request(request))


def test_persistent_worker_transitions_from_acquisition_to_target_ray(backend):
    """Protect the fixed-shape goal set across the real mission request order."""
    acquisition_selected, _segments, _duration = backend.plan(
        _request('ROUGH_ACQUISITION'))
    scan_selected, _segments, _duration = backend.plan(
        _request('MULTIVIEW_SCAN'))
    assert acquisition_selected
    assert scan_selected
    assert acquisition_selected[0]['curobo_attempt_diagnostics'][0][
        'native_padded_goalset_size'] == 54
    assert scan_selected[0]['curobo_attempt_diagnostics'][0][
        'native_padded_goalset_size'] == 54


def test_rough_acquisition_accepts_the_configured_folded_start(backend):
    """The first acquisition mirrors Tesseract's bounded startup exception."""
    request = copy.deepcopy(_request('ROUGH_ACQUISITION'))
    request.pop('request_sha256', None)
    request['start_state']['positions_rad'] = [
        -0.010187296, 0.0, -0.01692068,
        0.068485144, 0.441280868, 0.012594568,
    ]
    request = validate_request(attach_digest(request, 'request_sha256'))
    selected, segments, _duration = backend.plan(request)
    assert selected
    first = segments[0]
    assert first['bootstrap_recovery_used'] is True
    assert first['bootstrap_recovery_joint'] == 3
    assert abs(first['bootstrap_recovery_delta_rad']) <= 0.15
    endpoint = first['bootstrap_recovery_end_point']
    assert 1 <= endpoint < len(first['points']) - 1
    assert first['points'][0]['positions_rad'] == pytest.approx(
        request['start_state']['positions_rad'])


def test_exact_bunker_world_preserves_known_joint_path(backend):
    """Exercise the generated model without requiring mission fixtures."""
    provenance = backend.model_provenance
    assert provenance['schema_version'] == 3
    assert provenance['position_limit_clip_rad'] == pytest.approx(
        [0.005, 0.0, 0.0, 0.005, 0.005, 0.005])
    assert provenance['hardware_qualified'] is True
    assert provenance['hardware_qualification']['scope'] == (
        'supervised_5_percent_target_scan')
    assert provenance['hardware_qualification']['floor_profile'] == 'tabletop'
    assert provenance['hardware_qualification'][
        'free_motion_speed_percent'] == pytest.approx(5.0)
    assert provenance['hardware_qualification'][
        'contact_speed_percent'] == pytest.approx(5.0)
    assert provenance['hardware_qualification'][
        'real_motion_requires_explicit_opt_in'] is True
    assert provenance['conservative_geometry'] is False
    assert {
        item['name'] for item in provenance['fixed_world_meshes']
    } == {
        'bunker_chassis_collision',
        'bunker_sensor_station_collision',
        'piper_base_collision',
    }
    assert set(provenance['moving_link_surface_coverage']) == set(
        backend.collision_link_names)

    backend._update_world({'scene': {'obstacles': []}})
    neutral = [0.0, 0.8, -0.7, 0.0, 0.7, 0.0]
    qualified_scan = [
        0.3189509166, 0.7800870124, -1.6258884709,
        -0.6660237320, -0.2154052887, 0.0403545644,
    ]
    request = {
        'planning': {'effective_speed_percent': 5.0},
        'limits': {'position_rad': [[-3.2, 3.2] for _ in range(6)]},
    }
    outbound, _outbound_duration = backend._plan_joint_goal(
        neutral, qualified_scan, request)
    inbound, _inbound_duration = backend._plan_joint_goal(
        qualified_scan, neutral, request)
    assert len(outbound) >= 2
    assert len(inbound) >= 2
    assert outbound[-1]['positions_rad'] == pytest.approx(qualified_scan)
    assert inbound[-1]['positions_rad'] == pytest.approx(neutral)


def test_world_reuse_preserves_collision_and_obstacle_removal(backend):
    clear = {'scene': {'obstacles': []}}
    joints = [0.0, 0.8, -0.7, 0.0, 0.7, 0.0]
    backend.last_planning_diagnostics = {}
    backend._update_world(clear)
    backend._update_world(clear)
    assert backend.last_planning_diagnostics['world_geometry_reused']
    assert backend._start_state_status(joints)[0]
    blocked = {'scene': {'obstacles': [{
        'id': 'cache_block', 'type': 'box',
        'minimum_m': [-2, -2, -2], 'maximum_m': [2, 2, 2]}]}}
    try:
        backend._update_world(blocked)
        assert not backend.last_planning_diagnostics['world_geometry_reused']
        backend._update_world(blocked)
        assert backend.last_planning_diagnostics['world_geometry_reused']
        assert not backend._start_state_status(joints)[0]
    finally:
        backend._update_world(clear)
    assert not backend.last_planning_diagnostics['world_geometry_reused']
    assert backend._start_state_status(joints)[0]


def test_exact_world_still_rejects_blocking_dynamic_geometry(backend):
    backend._update_world({'scene': {'obstacles': [{
        'id': 'gpu_test_blocking_volume',
        'type': 'box',
        'minimum_m': [-2.0, -2.0, -2.0],
        'maximum_m': [2.0, 2.0, 2.0],
    }]}})
    neutral = [0.0, 0.8, -0.7, 0.0, 0.7, 0.0]
    qualified_scan = [
        0.3189509166, 0.7800870124, -1.6258884709,
        -0.6660237320, -0.2154052887, 0.0403545644,
    ]
    with pytest.raises(CuroboContractError):
        backend._plan_joint_goal(
            neutral, qualified_scan,
            {
                'planning': {'effective_speed_percent': 5.0},
                'limits': {
                    'position_rad': [[-3.2, 3.2] for _ in range(6)]},
            })


def test_same_native_gpu_trajectory_retains_profile_at_experimental_rates(
        backend, monkeypatch, tmp_path):
    """One real solve, three software-only schedules; no robot connection."""
    import numpy as np
    native_results = []
    native_plan = backend._timed_native_plan

    def record_native(*args, **kwargs):
        result = native_plan(*args, **kwargs)
        native_results.append(result)
        return result

    monkeypatch.setattr(backend, '_timed_native_plan', record_native)
    backend._update_world({'scene': {'obstacles': []}})
    backend.last_planning_diagnostics = {}
    limits = [[-3.2, 3.2]] * 6
    baseline, _ = backend._plan_joint_goal(
        [0., .8, -.7, 0., .7, 0.],
        [.3189509166, .7800870124, -1.6258884709,
         -.6660237320, -.2154052887, .0403545644],
        {'planning': {'effective_speed_percent': 5.},
         'limits': {'position_rad': limits}})
    assert len(native_results) == 1
    result = native_results[0]
    source = result.get_interpolated_plan().position.detach().cpu().numpy().reshape(-1, 6)
    reports = []
    for rate in (100., 200.):
        points = backend._path(result, 5., limits, rate)
        q = np.array([p['positions_rad'] for p in points])
        stride = (len(points) - 1) // (len(source) - 1)
        assert q[::stride] == pytest.approx(source, abs=1e-12)
        assert points[-1]['time_from_start_s'] >= baseline[-1]['time_from_start_s']
        assert np.diff([p['time_from_start_s'] for p in points]) == pytest.approx(1 / rate)
        velocity = np.diff(q, axis=0) * rate
        assert np.all(np.max(np.abs(velocity), axis=0) <= np.array([.25] * 5 + [.15]) + 1e-8)
        timing = backend.last_planning_diagnostics['trajectory_timing']
        assert timing['policy'] == 'uniform_native_time_scale'
        assert len(timing['native_peak_velocity']) == 6
        reports.append(dict(timing))
    (tmp_path / 'same_native_trajectory.json').write_text(json.dumps({
        'evidence': 'COMMAND_FREE_GPU_SAME_NATIVE_PATH',
        'baseline_duration_sec': baseline[-1]['time_from_start_s'],
        'experimental': reports,
    }, indent=2))
    print('same-native timing report:', tmp_path / 'same_native_trajectory.json')


def test_curated_model_rejects_known_folded_self_collision(backend):
    backend._update_world({'scene': {'obstacles': []}})
    valid, status = backend.motion_gen.check_start_state(backend._joint_state([
        0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0,
    ]))
    assert valid is False
    assert str(status).endswith('INVALID_START_STATE_SELF_COLLISION')


def test_real_model_enforces_joint5_internal_limit_clip(backend):
    backend._update_world({'scene': {'obstacles': []}})
    inside = [0.0, 0.8, -0.7, 0.0, 1.214, 0.0]
    beyond_clip = [0.0, 0.8, -0.7, 0.0, 1.217, 0.0]
    inside_valid, inside_status = backend._start_state_status(inside)
    outside_valid, outside_status = backend._start_state_status(beyond_clip)
    assert inside_valid is True
    assert inside_status == 'None'
    assert outside_valid is False
    assert outside_status.endswith('INVALID_START_STATE_JOINT_LIMITS')


def test_real_model_reenters_recorded_joint6_feedback_boundary(backend):
    """Reproduce the 2026-09-02 post-capture clipped-limit failure."""
    request = copy.deepcopy(_request('MULTIVIEW_SCAN'))
    request.pop('request_sha256', None)
    request['start_state']['positions_rad'] = [
        0.047482568,
        0.7713736800000001,
        -0.659313424,
        0.0,
        0.71712284,
        3.136623084,
    ]
    request = validate_request(attach_digest(request, 'request_sha256'))
    backend._update_world(request)

    recovery = backend._position_limit_reentry(
        request['start_state']['positions_rad'], request)

    assert recovery is not None
    assert recovery['joint_numbers'] == [6]
    assert recovery['positions'][0] == pytest.approx(
        request['start_state']['positions_rad'])
    assert recovery['positions'][-1][5] < math.pi - 0.005
    valid, status = backend._start_state_status(recovery['positions'][-1])
    assert valid is True
    assert status == 'None'
