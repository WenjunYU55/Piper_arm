"""Pure cuRobo adapter tests; no CUDA or cuRobo import is permitted here."""

import json
import math
from types import SimpleNamespace

import pytest

from motion_planning.curobo.adapter import (
    attach_digest,
    CuroboContractError,
    normalize_trajectory,
    obstacle_cuboids,
    prepend_bootstrap_recovery,
    trajectory_segment,
    validate_request,
)
from motion_planning.curobo.worker import CuroboBackend, Worker


def request_fixture():
    request = {
        'schema_version': 5,
        'planner_backend': 'curobo',
        'plan_kind': 'MULTIVIEW_SCAN',
        'start_state': {
            'joint_names': [
                'joint1', 'joint2', 'joint3',
                'joint4', 'joint5', 'joint6'],
            'positions_rad': [0.0] * 6,
        },
        'planning': {
            'planner': 'MotionGen',
            'pipeline': 'CUROBO_V1',
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
        },
        'scene': {
            'target_center_m': [0.4, 0.0, 0.12],
            'candidate_views': [{
                'id': 7,
                'camera_position_m': [0.1, 0.0, 0.12],
                'look_direction': [1.0, 0.0, 0.0],
            }],
            'obstacles': [{
                'id': 'box', 'type': 'box',
                'minimum_m': [0.2, -0.1, 0.0],
                'maximum_m': [0.3, 0.1, 0.2],
            }],
        },
    }
    return attach_digest(request, 'request_sha256')


def test_request_and_world_conversion_preserve_authoritative_geometry():
    request = validate_request(request_fixture())
    cuboids = obstacle_cuboids(request, floor_z_m=0.005)
    assert cuboids[0]['pose'][:3] == pytest.approx([0.25, 0.0, 0.1])
    assert cuboids[0]['dims'] == pytest.approx([0.1, 0.2, 0.2])
    assert cuboids[-1]['name'] == 'configured_support_floor'
    assert cuboids[-1]['pose'][2] == pytest.approx(-0.055)
    assert cuboids[-1]['dims'] == pytest.approx([4.0, 4.0, 0.10])


@pytest.mark.parametrize('mutation', [
    lambda value: value.update(planner_backend='tesseract'),
    lambda value: value.update(plan_kind='OCCLUSION_PROBE'),
    lambda value: value['start_state'].update(joint_names=['joint2'] * 6),
    lambda value: value['start_state'].update(
        positions_rad=[0.0, 0.0, math.nan, 0.0, 0.0, 0.0]),
    lambda value: value['scene'].update(candidate_views=[]),
])
def test_invalid_or_unsupported_requests_fail_closed(mutation):
    request = request_fixture()
    request.pop('request_sha256')
    mutation(request)
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(CuroboContractError):
        validate_request(request)


def test_request_digest_correlation_is_mandatory():
    request = request_fixture()
    request['scene']['target_center_m'][0] = 0.5
    with pytest.raises(CuroboContractError, match='request_sha256 mismatch'):
        validate_request(request)


def test_out_of_limit_bootstrap_is_explicitly_unsupported():
    request = request_fixture()
    request.pop('request_sha256')
    request['start_state']['positions_rad'][2] = 1.01
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(CuroboContractError, match='unsupported cuRobo bootstrap'):
        validate_request(request)


def test_native_path_is_slowed_and_subdivided_without_changing_endpoints():
    points = normalize_trajectory(
        [[0.0] * 6, [0.12, 0.0, 0.0, 0.0, 0.0, 0.0]],
        native_dt_sec=0.01,
        speed_percent=5.0,
    )
    assert points[0]['positions_rad'] == [0.0] * 6
    assert points[-1]['positions_rad'][0] == pytest.approx(0.12)
    assert len(points) == 4
    assert all(
        right['time_from_start_s'] > left['time_from_start_s']
        for left, right in zip(points, points[1:]))
    assert all(
        max(abs(b - a) for a, b in zip(
            left['positions_rad'], right['positions_rad'])) <= 0.05
        for left, right in zip(points, points[1:]))


def test_bootstrap_recovery_is_prepended_and_declared_generically():
    planned = normalize_trajectory(
        [[0.0, 0.0, -0.06, 0.0, 0.4, 0.0],
         [0.0, 0.1, -0.20, 0.0, 0.5, 0.0]],
        native_dt_sec=0.05,
        speed_percent=5.0,
    )
    recovery = [
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, -0.03, 0.0, 0.4, 0.0],
        [0.0, 0.0, -0.06, 0.0, 0.4, 0.0],
    ]
    points, endpoint = prepend_bootstrap_recovery(
        planned, recovery, speed_percent=5.0)
    segment = trajectory_segment(points, bootstrap_recovery={
        'end_point': endpoint,
        'joint_numbers': [3],
        'delta_rad': [-0.06],
    })
    assert points[0]['positions_rad'] == recovery[0]
    assert points[endpoint]['positions_rad'] == recovery[-1]
    assert points[-1]['positions_rad'] == planned[-1]['positions_rad']
    assert all(
        right['time_from_start_s'] > left['time_from_start_s']
        for left, right in zip(points, points[1:]))
    assert segment['bootstrap_recovery_used'] is True
    assert segment['bootstrap_recovery_joint'] == 3
    assert segment['bootstrap_recovery_delta_rad'] == pytest.approx(-0.06)


def test_curobo_bootstrap_exception_is_rough_acquisition_only():
    backend = object.__new__(CuroboBackend)
    backend._start_state_status = lambda position: (
        (True, 'None') if position[2] <= -0.06 else
        (False, 'MotionGenStatus.INVALID_START_STATE_SELF_COLLISION'))
    request = request_fixture()
    request['plan_kind'] = 'ROUGH_ACQUISITION'
    request['scene']['observation_mode'] = 'bootstrap_static'
    start = [0.0] * 6
    recovery = CuroboBackend._bootstrap_recovery(backend, start, request)
    assert recovery['joint_numbers'] == [3]
    assert recovery['delta_rad'] == pytest.approx([-0.06])
    assert recovery['positions'][0] == start
    assert recovery['positions'][-1][2] == pytest.approx(-0.06)

    request['plan_kind'] = 'MULTIVIEW_SCAN'
    request['scene']['observation_mode'] = 'perception_snapshot'
    with pytest.raises(CuroboContractError, match='outside bootstrap scope'):
        CuroboBackend._bootstrap_recovery(backend, start, request)


@pytest.mark.parametrize('bad', [
    [], [[0.0] * 6], [[0.0] * 6, [math.inf] * 6],
])
def test_malformed_native_trajectory_fails_closed(bad):
    with pytest.raises(CuroboContractError):
        normalize_trajectory(bad, 0.05, 5.0)


def test_collision_qualification_requires_model_evidence_and_operator_opt_in(
        monkeypatch):
    worker = SimpleNamespace(backend=SimpleNamespace(
        model_provenance={'hardware_qualified': False}))
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    assert Worker.collision_model_qualified(worker) is False

    worker.backend.model_provenance['hardware_qualified'] = True
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '0')
    assert Worker.collision_model_qualified(worker) is False

    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    assert Worker.collision_model_qualified(worker) is True


def test_worker_health_stays_bounded_when_model_provenance_is_large():
    provenance = {
        'hardware_qualified': False,
        'collision_spheres': ['sphere'] * 4000,
    }
    worker = SimpleNamespace(
        generation_id='a' * 32,
        backend_error='',
        backend=SimpleNamespace(
            version='0.7.8',
            robot_config_sha256='b' * 64,
            model_provenance=provenance,
            environment={'cuda_available': True},
        ),
        model_hashes={
            'srdf_sha256': 'c' * 64,
            'collision_manifest_sha256': 'd' * 64,
        },
        collision_model_qualified=lambda: False,
    )

    health = Worker.health(worker)
    diagnostics = Worker.diagnostics(worker)
    encoded_health = json.dumps(
        health, sort_keys=True, separators=(',', ':')).encode('utf-8')

    assert len(encoded_health) <= 16 * 1024
    assert 'model_provenance' not in health
    assert diagnostics['model_provenance'] == provenance
    assert health['robot_config_sha256'] == diagnostics['robot_config_sha256']
    assert health['collision_manifest_sha256'] == (
        diagnostics['collision_manifest_sha256'])
