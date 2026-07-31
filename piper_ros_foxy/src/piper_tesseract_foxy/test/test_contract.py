import copy
import time

import pytest

from piper_tesseract_foxy.contract import (
    attach_digest,
    ContractError,
    JOINT_NAMES,
    motion_limits_digest,
    SCHEMA_VERSION,
    Spool,
    trajectory_digest,
    validate_request,
    validate_response,
)
from piper_tesseract_foxy.worker import Worker


def request_fixture(plan_kind='MULTIVIEW_SCAN'):
    now = time.time_ns()
    provenance = {
        'source': (
            'tracked_target' if plan_kind == 'MULTIVIEW_SCAN'
            else 'rough_coordinate'),
        'frame_id': 'base_link',
        'stamp': {'sec': 123, 'nanosec': 456},
    }
    if plan_kind == 'ROUGH_ACQUISITION':
        provenance['source_request_id'] = 'acq-0123456789abcdef'
    value = {
        'schema_version': SCHEMA_VERSION,
        'plan_kind': plan_kind,
        'target_provenance': provenance,
        'request_id': 'a' * 32,
        'created_at_ns': now,
        'expires_at_ns': now + 10_000_000_000,
        'start_state': {
            'joint_names': list(JOINT_NAMES),
            'positions_rad': [0.0] * 6,
        },
        'scene': {
            'target_center_m': [0.5, 0.0, 0.2],
            'target_provenance': provenance,
            'observation_mode': (
                'perception_snapshot'
                if plan_kind == 'MULTIVIEW_SCAN'
                else 'bootstrap_static'),
            'candidate_views': [{
                'id': 1,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
            }],
            'obstacles': [],
        },
        'model': {
            'mode': 0,
            'xacro_sha256': '1' * 64,
            'srdf_sha256': '2' * 64,
            'collision_manifest_sha256': '3' * 64,
        },
        'calibration': {
            'hand_eye_sha256': '4' * 64,
            'T_link6_camera': [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
            'max_velocity_rad_s': [1.0] * 6,
            'max_acceleration_rad_s2': [2.0] * 6,
            'motion_limits_sha256': motion_limits_digest(
                [1.0] * 6, [2.0] * 6),
            'source': 'piper_sdk_controller_feedback',
        },
        'planning': {
            'planner': 'RRTConnect',
            'pipeline': 'OMPL_ISP',
            'deterministic_seed': 42,
            'roll_samples_rad': [0.0],
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'max_execution_joint_step_rad': 0.10,
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
            'joint_specific_costs': {},
            'return_home_positions_rad': (
                [0.0, -0.032, -0.026, -0.039, 0.346, 0.107]
                if plan_kind == 'MULTIVIEW_SCAN' else []),
        },
    }
    return attach_digest(value, 'request_sha256')


def response_fixture(request):
    points = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 1.0,
            'positions_rad': [0.01] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    segments = [{'points': points}]
    if request['plan_kind'] == 'MULTIVIEW_SCAN':
        home_points = [
            copy.deepcopy(points[-1]),
            {
                'time_from_start_s': 1.0,
                'positions_rad':
                    request['planning']['return_home_positions_rad'],
                'velocities_rad_s': [0.0] * 6,
                'accelerations_rad_s2': [0.0] * 6,
            },
        ]
        home_points[0]['time_from_start_s'] = 0.0
        segments.append({
            'is_return_home': True,
            'points': home_points,
        })
    binding = {
        'request_sha256': request['request_sha256'],
        'plan_kind': request['plan_kind'],
        'target_provenance': request['target_provenance'],
        'model': request['model'],
        'calibration': request['calibration']['hand_eye_sha256'],
        'limits': request['limits'],
        'execution': {
            'effective_speed_percent':
                request['planning']['effective_speed_percent'],
            'command_rate_hz': request['planning']['command_rate_hz'],
            'timing_policy': request['planning']['timing_policy'],
        },
    }
    value = {
        'schema_version': SCHEMA_VERSION,
        'plan_kind': request['plan_kind'],
        'target_provenance': request['target_provenance'],
        'request_id': request['request_id'],
        'request_sha256': request['request_sha256'],
        'status': 'success',
        'deterministic_seed': request['planning']['deterministic_seed'],
        'joint_names': list(JOINT_NAMES),
        'target_center_m': request['scene']['target_center_m'],
        'selected_viewpoints': [{
            'id': request['scene']['candidate_views'][0]['id'],
            'camera_position_m': request['scene']['candidate_views'][0][
                'camera_position_m'],
            'look_direction': request['scene']['candidate_views'][0][
                'look_direction'],
            'roll_rad': 0.0,
        }],
        'segments': segments,
        'trajectory_binding': binding,
        'trajectory_sha256': trajectory_digest(segments, binding),
    }
    return attach_digest(value, 'response_sha256')


def test_request_and_response_hashes_are_fail_closed():
    request = request_fixture()
    assert validate_request(request) is request
    response = response_fixture(request)
    assert validate_response(response, request) is response
    tampered = copy.deepcopy(response)
    tampered['segments'][0]['points'][1]['positions_rad'][5] = 0.5
    with pytest.raises(ContractError, match='response_sha256'):
        validate_response(tampered, request)


def test_response_seed_must_match_the_bound_request():
    request = request_fixture()
    response = response_fixture(request)
    response['deterministic_seed'] = 43
    response = attach_digest(response, 'response_sha256')
    with pytest.raises(ContractError, match='deterministic_seed mismatch'):
        validate_response(response, request)


def test_multiview_response_requires_exact_declared_return_home_endpoint():
    request = request_fixture()
    response = response_fixture(request)
    response['segments'][-1]['is_return_home'] = False
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response = attach_digest(response, 'response_sha256')
    with pytest.raises(ContractError, match='not declared return home'):
        validate_response(response, request)

    response = response_fixture(request)
    response['segments'][-1]['points'][-1]['positions_rad'][5] += 0.01
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response = attach_digest(response, 'response_sha256')
    with pytest.raises(ContractError, match='does not match the request'):
        validate_response(response, request)


def test_request_rejects_the_removed_trajopt_pipeline_label():
    request = request_fixture()
    request['planning']['pipeline'] = 'OMPL_TrajOpt_ISP'
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='planning.pipeline is unsupported'):
        validate_request(request)


def test_request_requires_a_bounded_nonzero_deterministic_seed():
    for invalid in (None, 0, -1, True, 0x100000000):
        request = request_fixture()
        request['planning']['deterministic_seed'] = invalid
        request = attach_digest(request, 'request_sha256')
        with pytest.raises(ContractError, match='deterministic_seed'):
            validate_request(request)


def test_acquisition_request_binds_kind_provenance_and_optional_sweep():
    request = request_fixture('ROUGH_ACQUISITION')
    request['scene']['candidate_views'].extend([
        {
            'id': index,
            'camera_position_m': [0.4, 0.01 * index, 0.3],
            'look_direction': [1.0, 0.0, 0.0],
        }
        for index in range(2, 6)
    ])
    request['planning']['max_viewpoints'] = 5
    request = attach_digest(request, 'request_sha256')

    assert validate_request(request) is request
    response = response_fixture(request)
    assert validate_response(response, request) is response


def test_only_rough_acquisition_may_bind_two_bounded_outside_start_joints():
    request = request_fixture('ROUGH_ACQUISITION')
    request['limits']['position_rad'][2] = [-1.0, 0.0]
    request['limits']['bootstrap_start_limit_tolerance_rad'] = 0.04
    request['start_state']['positions_rad'][2] = 0.0327
    request = attach_digest(request, 'request_sha256')
    assert validate_request(request) is request

    too_far = copy.deepcopy(request)
    too_far['start_state']['positions_rad'][2] = 0.041
    too_far = attach_digest(too_far, 'request_sha256')
    with pytest.raises(ContractError, match='joint3 is outside limits'):
        validate_request(too_far)

    two_joints = copy.deepcopy(request)
    two_joints['limits']['position_rad'][1] = [0.0, 1.0]
    two_joints['start_state']['positions_rad'][1] = -0.01
    two_joints = attach_digest(two_joints, 'request_sha256')
    assert validate_request(two_joints) is two_joints

    three_joints = copy.deepcopy(two_joints)
    three_joints['limits']['position_rad'][0] = [0.0, 1.0]
    three_joints['start_state']['positions_rad'][0] = -0.01
    three_joints = attach_digest(three_joints, 'request_sha256')
    with pytest.raises(ContractError, match='at most two joints'):
        validate_request(three_joints)

    multiview = request_fixture('MULTIVIEW_SCAN')
    multiview['limits']['bootstrap_start_limit_tolerance_rad'] = 0.04
    multiview = attach_digest(multiview, 'request_sha256')
    with pytest.raises(ContractError, match='acquisition-only'):
        validate_request(multiview)


def test_multiview_contract_accepts_exactly_thirteen_views_and_rejects_more():
    request = request_fixture()
    request['scene']['candidate_views'] = [
        {
            'id': index,
            'camera_position_m': [0.4, 0.01 * index, 0.3],
            'look_direction': [1.0, 0.0, 0.0],
        }
        for index in range(13)
    ]
    request['planning']['min_viewpoints'] = 13
    request['planning']['max_viewpoints'] = 13
    request = attach_digest(request, 'request_sha256')

    assert validate_request(request) is request

    request['planning']['max_viewpoints'] = 14
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='planning.max_viewpoints'):
        validate_request(request)


def test_scene_observation_mode_is_bound_to_plan_kind():
    request = request_fixture('ROUGH_ACQUISITION')
    request['scene']['observation_mode'] = 'perception_snapshot'
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='observation_mode'):
        validate_request(request)

    request = request_fixture('ROUGH_ACQUISITION')
    request['scene']['obstacles'] = [{
        'id': 'unexpected',
        'type': 'box',
        'minimum_m': [0.1, 0.1, 0.1],
        'maximum_m': [0.2, 0.2, 0.2],
    }]
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='must not contain perception obstacles'):
        validate_request(request)


def test_plan_kind_and_provenance_must_match():
    request = request_fixture('ROUGH_ACQUISITION')
    request['target_provenance']['source'] = 'tracked_target'
    request['scene']['target_provenance']['source'] = 'tracked_target'
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='does not match plan_kind'):
        validate_request(request)


def test_normal_scan_cannot_relax_minimum_viewpoint_count():
    request = request_fixture()
    request['scene']['candidate_views'].append({
        'id': 2,
        'camera_position_m': [0.4, 0.1, 0.3],
        'look_direction': [1.0, 0.0, 0.0],
    })
    request['planning']['max_viewpoints'] = 2
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='identical min/max'):
        validate_request(request)


def test_response_rejects_kind_or_provenance_substitution():
    request = request_fixture('ROUGH_ACQUISITION')
    response = response_fixture(request)
    response['plan_kind'] = 'MULTIVIEW_SCAN'
    response['target_provenance']['source'] = 'tracked_target'
    response = attach_digest(response, 'response_sha256')
    with pytest.raises(
            ContractError,
            match='plan_kind mismatch|must not carry source_request_id'):
        validate_response(response, request)


def test_spool_atomically_claims_a_private_request(tmp_path):
    spool = Spool(tmp_path / 'spool')
    request = request_fixture()
    spool.write('requests', request['request_id'], request)
    request_id, claimed = spool.claim_next()
    assert request_id == request['request_id']
    assert claimed == request
    assert not spool.path('requests', request_id).exists()
    assert spool.path('processing', request_id).is_file()


def test_spool_rejects_symlink_root(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ContractError, match='symlink'):
        Spool(link)


def test_worker_health_record_is_atomic_and_bounded(tmp_path):
    spool = Spool(tmp_path / 'spool')
    first = {
        'schema_version': SCHEMA_VERSION,
        'generation_id': '1' * 32,
        'written_at_ns': time.time_ns(),
        'worker_ready': True,
        'backend': 'tesseract',
        'backend_version': '0.35.0.6',
        'backend_error': '',
    }
    second = dict(first, written_at_ns=time.time_ns() + 1)
    spool.write_health(first)
    assert spool.read_health() == first
    spool.write_health(second)
    assert spool.read_health() == second


def test_rough_request_requires_bound_source_request_id():
    request = request_fixture('ROUGH_ACQUISITION')
    request['target_provenance'].pop('source_request_id')
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(ContractError, match='source_request_id'):
        validate_request(request)


def test_nonfinite_request_is_rejected():
    request = request_fixture()
    request['start_state']['positions_rad'][0] = float('nan')
    with pytest.raises(ContractError, match='finite JSON'):
        attach_digest(request, 'request_sha256')


def test_worker_fails_closed_when_tesseract_is_unavailable(tmp_path):
    spool = Spool(tmp_path / 'spool')
    request = request_fixture()
    spool.write('requests', request['request_id'], request)
    worker = Worker(
        spool.root,
        tmp_path / 'missing.urdf',
        tmp_path / 'missing.srdf',
        tmp_path / 'missing.yaml',
    )
    assert worker.process_once()
    response = spool.read('responses', request['request_id'])
    assert response['status'] == 'failed'
    assert response['rejection_codes'] == ['BACKEND_UNAVAILABLE']
    assert not spool.path('processing', request['request_id']).exists()
