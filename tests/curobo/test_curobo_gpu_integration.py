"""Opt-in real cuRobo/CUDA planner tests; these never command the robot."""

import copy
import json
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


def _assert_fixed_rate_schedule(segment):
    points = segment['points']
    assert len(points) >= 2
    assert all(
        right['time_from_start_s'] - left['time_from_start_s']
        == pytest.approx(0.05, abs=1e-9)
        for left, right in zip(points, points[1:]))


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
