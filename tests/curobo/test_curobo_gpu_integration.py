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
