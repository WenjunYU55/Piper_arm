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


def test_exact_bunker_world_preserves_known_joint_path(backend):
    """Exercise the generated model without requiring mission fixtures."""
    provenance = backend.model_provenance
    assert provenance['schema_version'] == 2
    assert provenance['hardware_qualified'] is False
    assert provenance['conservative_geometry'] is False
    assert {
        item['name'] for item in provenance['fixed_world_meshes']
    } == {
        'bunker_chassis_collision',
        'bunker_sensor_station_collision',
    }
    assert set(provenance['moving_link_surface_coverage']) == set(
        backend.collision_link_names)

    backend._update_world({'scene': {'obstacles': []}})
    neutral = [0.0, 0.8, -0.7, 0.0, 0.7, 0.0]
    qualified_scan = [
        0.3189509166, 0.7800870124, -1.6258884709,
        -0.6660237320, -0.2154052887, 0.0403545644,
    ]
    request = {'planning': {'effective_speed_percent': 5.0}}
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
            {'planning': {'effective_speed_percent': 5.0}})


def test_curated_model_rejects_known_link2_link5_collision(backend):
    backend._update_world({'scene': {'obstacles': []}})
    valid, status = backend.motion_gen.check_start_state(backend._joint_state([
        0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0,
    ]))
    assert valid is False
    assert str(status).endswith('INVALID_START_STATE_SELF_COLLISION')
