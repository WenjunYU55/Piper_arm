import copy
import math
import time

import numpy as np
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
        'source': {
            'MULTIVIEW_SCAN': 'tracked_target',
            'ROUGH_ACQUISITION': 'rough_coordinate',
            'RETURN_HOME': 'configured_home',
        }[plan_kind],
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
                'bootstrap_static'
                if plan_kind == 'ROUGH_ACQUISITION'
                else 'perception_snapshot'),
            'candidate_views': ([] if plan_kind == 'RETURN_HOME' else [{
                'id': 1,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
                'required_first': plan_kind == 'ROUGH_ACQUISITION',
            }]),
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
            'min_viewpoints': 0 if plan_kind == 'RETURN_HOME' else 1,
            'max_viewpoints': 0 if plan_kind == 'RETURN_HOME' else 1,
            'max_execution_joint_step_rad': 0.10,
            'effective_speed_percent': 100.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
            'joint_specific_costs': {},
            'return_home_positions_rad': (
                [0.0, 0.0, -0.026, -0.039, 0.346, 0.107]
                if plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME') else []),
        },
    }
    return attach_digest(value, 'request_sha256')


def scheduled_path(knots, maximum_step=0.10):
    positions = [list(knots[0])]
    for goal in knots[1:]:
        start = np.asarray(positions[-1], dtype=float)
        goal = np.asarray(goal, dtype=float)
        steps = max(1, int(math.ceil(
            float(np.max(np.abs(goal - start))) / maximum_step)))
        positions.extend([
            (start + (goal - start) * (index / float(steps))).tolist()
            for index in range(1, steps + 1)
        ])
    return [{
        'time_from_start_s': round(index * 0.5, 9),
        'positions_rad': position,
        'velocities_rad_s': [0.0] * 6,
        'accelerations_rad_s2': [0.0] * 6,
    } for index, position in enumerate(positions)]


def response_fixture(request):
    points = scheduled_path([[0.0] * 6, [0.01] * 6])
    segments = [] if request['plan_kind'] == 'RETURN_HOME' else [{'points': points}]
    if (
            request['plan_kind'] in ('MULTIVIEW_SCAN', 'RETURN_HOME')
            and request['planning'].get('include_return_home', True)):
        home_points = scheduled_path([
            (points[0]['positions_rad']
             if request['plan_kind'] == 'RETURN_HOME'
             else points[-1]['positions_rad']),
            request['planning']['return_home_positions_rad'],
        ])
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
        'selected_viewpoints': ([] if request['plan_kind'] == 'RETURN_HOME' else [{
            'id': request['scene']['candidate_views'][0]['id'],
            'camera_position_m': request['scene']['candidate_views'][0][
                'camera_position_m'],
            'look_direction': request['scene']['candidate_views'][0][
                'look_direction'],
            'roll_rad': 0.0,
        }]),
        'segments': segments,
        'trajectory_binding': binding,
        'trajectory_sha256': trajectory_digest(segments, binding),
    }
    return attach_digest(value, 'response_sha256')


def expanded_ray_request(ray_ids):
    """Return a valid private request with one exact probe per ray ID."""
    request = request_fixture('MULTIVIEW_SCAN')
    template = request['scene']['candidate_views'][0]
    request['scene']['candidate_views'] = [dict(
        template,
        id=1000 + index,
        ray_id=int(ray_id),
        ray_probe_index=index,
    ) for index, ray_id in enumerate(ray_ids)]
    request['planning'].update({
        'shortlisted_ray_count': len(set(ray_ids)),
        'expanded_ray_candidate_count': len(ray_ids),
        'ray_direction_attempt_limit': 6,
    })
    return attach_digest(request, 'request_sha256')


def target_ray_request():
    request = request_fixture('MULTIVIEW_SCAN')
    request['scene']['candidate_views'] = [{
        'id': 100,
        'camera_position_m': [0.80, 0.0, 0.20],
        'look_direction': [-1.0, 0.0, 0.0],
        'candidate_geometry': 'target_ray',
        'ray_id': 12,
        'ray_direction': [1.0, 0.0, 0.0],
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.50,
        'ray_preferred_max_standoff_m': 0.50,
        'ray_scoring_standoff_m': 0.39,
        'ray_standoff_m': 0.30,
        'ray_probe_index': 0,
        'ray_probe_phase': 'interval_search',
    }]
    request['planning'].update({
        'shortlisted_ray_count': 1,
        'expanded_ray_candidate_count': 1,
        'ray_direction_attempt_limit': 6,
    })
    return attach_digest(request, 'request_sha256')


def test_expanded_ray_request_requires_direction_contiguous_attempts():
    assert validate_request(expanded_ray_request([3, 3, 9, 9]))

    with pytest.raises(ContractError, match='direction-contiguous'):
        validate_request(expanded_ray_request([3, 9, 3]))


def test_expanded_ray_request_cannot_exceed_direction_attempt_bound():
    request = expanded_ray_request([1, 2, 3, 4, 5, 6, 7])

    with pytest.raises(ContractError, match='bounds'):
        validate_request(request)


def test_target_ray_response_may_select_another_point_on_bound_interval():
    request = target_ray_request()
    assert validate_request(request) is request
    response = response_fixture(request)
    response['selected_viewpoints'][0].update({
        'camera_position_m': [0.835, 0.0, 0.20],
        'ray_standoff_m': 0.335,
    })
    response = attach_digest(response, 'response_sha256')

    assert validate_response(response, request) is response


@pytest.mark.parametrize('position, standoff', (
    ([0.835, 0.01, 0.20], 0.335),
    ([1.01, 0.0, 0.20], 0.51),
    ([0.835, 0.0, 0.20], 0.34),
))
def test_target_ray_response_cannot_leave_requested_interval(
        position, standoff):
    request = target_ray_request()
    response = response_fixture(request)
    response['selected_viewpoints'][0].update({
        'camera_position_m': position,
        'ray_standoff_m': standoff,
    })
    response = attach_digest(response, 'response_sha256')

    with pytest.raises(ContractError, match='does not match request'):
        validate_response(response, request)


def test_closed_loop_one_view_response_omits_unused_embedded_home():
    request = request_fixture('MULTIVIEW_SCAN')
    request['planning']['include_return_home'] = False
    request = attach_digest(request, 'request_sha256')
    assert validate_request(request) is request

    response = response_fixture(request)
    assert len(response['segments']) == 1
    assert response['segments'][0].get('is_return_home') is not True
    assert validate_response(response, request) is response

    batch = copy.deepcopy(request)
    second = copy.deepcopy(batch['scene']['candidate_views'][0])
    second['id'] = 2
    batch['scene']['candidate_views'].append(second)
    batch['planning']['max_viewpoints'] = 2
    batch['planning']['min_viewpoints'] = 2
    batch = attach_digest(batch, 'request_sha256')
    with pytest.raises(ContractError, match='one closed-loop viewpoint'):
        validate_request(batch)


def rehash_response(response):
    response = copy.deepcopy(response)
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response.pop('response_sha256', None)
    return attach_digest(response, 'response_sha256')


def test_fallback_response_requires_hash_bound_bounded_aim_provenance():
    request = request_fixture('MULTIVIEW_SCAN')
    candidate = request['scene']['candidate_views'][0]
    fallback = [
        math.cos(math.radians(5.0)),
        math.sin(math.radians(5.0)),
        0.0,
    ]
    candidate['look_direction'] = [1.0, 0.0, 0.0]
    candidate['fallback_look_directions'] = [fallback]
    candidate['maximum_final_aim_offset_deg'] = 5.0
    request = attach_digest(request, 'request_sha256')
    assert validate_request(request) is request

    response = response_fixture(request)
    selected = response['selected_viewpoints'][0]
    selected['look_direction'] = fallback
    selected['nominal_look_direction'] = candidate['look_direction']
    selected['aim_fallback_used'] = True
    selected['aim_offset_deg'] = 5.0
    response = rehash_response(response)
    assert validate_response(response, request) is response

    malformed = copy.deepcopy(response)
    malformed['selected_viewpoints'][0].pop('aim_fallback_used')
    malformed = rehash_response(malformed)
    with pytest.raises(ContractError, match='fallback marker'):
        validate_response(malformed, request)


def test_request_rejects_fallback_beyond_five_degrees():
    request = request_fixture('MULTIVIEW_SCAN')
    candidate = request['scene']['candidate_views'][0]
    candidate['look_direction'] = [1.0, 0.0, 0.0]
    candidate['fallback_look_directions'] = [
        [math.cos(math.radians(6.0)), math.sin(math.radians(6.0)), 0.0]]
    candidate['maximum_final_aim_offset_deg'] = 5.0
    request = attach_digest(request, 'request_sha256')

    with pytest.raises(ContractError, match='fallback exceeds'):
        validate_request(request)


def test_request_accepts_later_view_ninety_degree_protocol_bound():
    request = request_fixture('MULTIVIEW_SCAN')
    candidate = request['scene']['candidate_views'][0]
    candidate['look_direction'] = [1.0, 0.0, 0.0]
    candidate['fallback_look_directions'] = [[0.0, 1.0, 0.0]]
    candidate['maximum_final_aim_offset_deg'] = 90.0
    request = attach_digest(request, 'request_sha256')
    assert validate_request(request) is request


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


def test_return_home_is_a_zero_capture_hash_bound_transaction():
    request = request_fixture('RETURN_HOME')
    assert validate_request(request) is request
    response = response_fixture(request)
    assert response['selected_viewpoints'] == []
    assert len(response['segments']) == 1
    assert response['segments'][0]['is_return_home'] is True
    assert validate_response(response, request) is response

    response = response_fixture(request)
    response['segments'][-1]['points'][-1]['positions_rad'][5] += 0.01
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response = attach_digest(response, 'response_sha256')
    with pytest.raises(ContractError, match='does not match the request'):
        validate_response(response, request)


def test_configured_home_collision_bypass_is_exactly_stage_bound():
    request = request_fixture('RETURN_HOME')
    request['planning']['home_stage'] = 'ROUGH_HOME'
    request.pop('request_sha256')
    request = attach_digest(request, 'request_sha256')
    response = response_fixture(request)
    response['segments'][0]['points'] = [
        response['segments'][0]['points'][0],
        response['segments'][0]['points'][-1],
    ]
    response['segments'][0]['points'][-1]['time_from_start_s'] = 0.05
    response['segments'][0].update({
        'configured_home_direct_joint_move': True,
        'configured_home_goal_positions_rad':
            request['planning']['return_home_positions_rad'],
        'collision_validation_bypassed': True,
        'validation': 'configured_home_collision_validation_bypassed',
        'home_stage': 'ROUGH_HOME',
    })
    response = rehash_response(response)

    assert validate_response(response, request) is response

    wrong_stage = copy.deepcopy(response)
    wrong_stage['segments'][0]['home_stage'] = 'STORAGE_WRIST'
    wrong_stage = rehash_response(wrong_stage)
    with pytest.raises(ContractError, match='bypass scope'):
        validate_response(wrong_stage, request)

    leaked = request_fixture('MULTIVIEW_SCAN')
    leaked_response = response_fixture(leaked)
    leaked_response['segments'][0].update({
        'configured_home_direct_joint_move': True,
        'configured_home_goal_positions_rad':
            leaked_response['segments'][0]['points'][-1]['positions_rad'],
        'collision_validation_bypassed': True,
        'validation': 'configured_home_collision_validation_bypassed',
        'home_stage': 'ROUGH_HOME',
    })
    leaked_response = rehash_response(leaked_response)
    with pytest.raises(ContractError, match='bypass scope'):
        validate_response(leaked_response, leaked)

    mismatched_goal = copy.deepcopy(response)
    mismatched_goal['segments'][0][
        'configured_home_goal_positions_rad'][5] += 0.01
    mismatched_goal = rehash_response(mismatched_goal)
    with pytest.raises(ContractError, match='declared goal does not match'):
        validate_response(mismatched_goal, request)


def test_return_home_accepts_only_declared_dual_recovery_targets():
    request = request_fixture('RETURN_HOME')
    response = response_fixture(request)
    segment = response['segments'][0]
    start = copy.deepcopy(segment['points'][0])
    home = copy.deepcopy(segment['points'][-1])
    powered_position = copy.deepcopy(start['positions_rad'])
    powered_position[2] = -0.06
    home_entry_position = copy.deepcopy(home['positions_rad'])
    home_entry_position[2] -= 0.03
    segment['points'] = scheduled_path([
        start['positions_rad'], powered_position,
        home_entry_position, home['positions_rad']])
    powered_end = next(
        index for index, point in enumerate(segment['points'])
        if point['positions_rad'] == powered_position)
    recovery_end = next(
        index for index, point in enumerate(segment['points'])
        if point['positions_rad'] == home_entry_position)
    segment.update({
        'bootstrap_recovery_used': True,
        'bootstrap_recovery_end_point': recovery_end,
        'powered_start_recovery_used': True,
        'powered_start_recovery_end_point': powered_end,
    })
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response = attach_digest(response, 'response_sha256')
    assert validate_response(response, request) is response

    invalid = copy.deepcopy(response)
    invalid['segments'][0]['powered_start_recovery_end_point'] = recovery_end
    invalid['trajectory_sha256'] = trajectory_digest(
        invalid['segments'], invalid['trajectory_binding'])
    invalid = attach_digest(invalid, 'response_sha256')
    with pytest.raises(ContractError, match='declaration is invalid'):
        validate_response(invalid, request)


def test_startup_home_accepts_powered_start_only_recovery_to_clear_home():
    request = request_fixture('RETURN_HOME')
    request['scene']['startup_home_static'] = True
    request = attach_digest(request, 'request_sha256')
    response = response_fixture(request)
    segment = response['segments'][0]
    start = copy.deepcopy(segment['points'][0])
    recovery_position = copy.deepcopy(start['positions_rad'])
    recovery_position[2] = -0.06
    home = copy.deepcopy(segment['points'][-1])
    segment['points'] = scheduled_path([
        start['positions_rad'], recovery_position, home['positions_rad']])
    recovery_end = next(
        index for index, point in enumerate(segment['points'])
        if point['positions_rad'] == recovery_position)
    segment['startup_home_static'] = True
    segment.update({
        'powered_start_recovery_used': True,
        'powered_start_recovery_end_point': recovery_end,
    })
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response = attach_digest(response, 'response_sha256')
    assert validate_response(response, request) is response

    invalid = copy.deepcopy(response)
    invalid['segments'][0]['powered_start_recovery_end_point'] = len(
        invalid['segments'][0]['points']) - 1
    invalid['trajectory_sha256'] = trajectory_digest(
        invalid['segments'], invalid['trajectory_binding'])
    invalid = attach_digest(invalid, 'response_sha256')
    with pytest.raises(ContractError, match='declaration is invalid'):
        validate_response(invalid, request)


def test_startup_home_static_scene_is_empty_and_return_home_only():
    request = request_fixture('RETURN_HOME')
    request['scene']['startup_home_static'] = True
    request = attach_digest(request, 'request_sha256')
    assert validate_request(request) is request

    invalid = request_fixture('MULTIVIEW_SCAN')
    invalid['scene']['startup_home_static'] = True
    invalid = attach_digest(invalid, 'request_sha256')
    with pytest.raises(ContractError, match='RETURN_HOME-only'):
        validate_request(invalid)

    invalid = request_fixture('RETURN_HOME')
    invalid['scene']['startup_home_static'] = True
    invalid['scene']['obstacles'] = [{
        'id': 'unexpected',
        'type': 'box',
        'minimum_m': [0.0, 0.0, 0.0],
        'maximum_m': [0.1, 0.1, 0.1],
    }]
    invalid = attach_digest(invalid, 'request_sha256')
    with pytest.raises(ContractError, match='must not contain'):
        validate_request(invalid)


def test_startup_home_static_is_bound_through_the_response_segment():
    request = request_fixture('RETURN_HOME')
    request['scene']['startup_home_static'] = True
    request = attach_digest(request, 'request_sha256')
    response = response_fixture(request)
    response['segments'][0]['startup_home_static'] = True
    response['trajectory_sha256'] = trajectory_digest(
        response['segments'], response['trajectory_binding'])
    response = attach_digest(response, 'response_sha256')
    assert validate_response(response, request) is response

    invalid = copy.deepcopy(response)
    invalid['segments'][0]['startup_home_static'] = False
    invalid['trajectory_sha256'] = trajectory_digest(
        invalid['segments'], invalid['trajectory_binding'])
    invalid = attach_digest(invalid, 'response_sha256')
    with pytest.raises(ContractError, match='mismatches the request'):
        validate_response(invalid, request)


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


def test_acquisition_request_requires_an_explicit_centered_first_candidate():
    request = request_fixture('ROUGH_ACQUISITION')
    request['scene']['candidate_views'][0]['required_first'] = False
    request = attach_digest(request, 'request_sha256')

    with pytest.raises(ContractError, match='required-first'):
        validate_request(request)


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


def test_direct_return_home_may_recover_a_bounded_post_disable_start():
    request = request_fixture('RETURN_HOME')
    request['limits']['configured_home_start_limit_tolerance_rad'] = 0.3
    request['start_state']['positions_rad'][5] = -1.03168
    request = attach_digest(request, 'request_sha256')
    assert validate_request(request) is request

    response = response_fixture(request)
    response['segments'][0]['points'][0]['positions_rad'][5] = -1.03168
    response['segments'][0]['points'] = [
        response['segments'][0]['points'][0],
        response['segments'][0]['points'][-1],
    ]
    response['segments'][0]['points'][-1]['time_from_start_s'] = 0.05
    response['segments'][0].update({
        'configured_home_direct_joint_move': True,
        'configured_home_goal_positions_rad':
            request['planning']['return_home_positions_rad'],
        'collision_validation_bypassed': True,
        'validation': 'configured_home_collision_validation_bypassed',
        'home_stage': 'CONFIGURED_HOME',
    })
    response['segments'][0]['points'][-1]['positions_rad'] = list(
        request['planning']['return_home_positions_rad'])
    response = rehash_response(response)
    assert validate_response(response, request) is response

    too_far = copy.deepcopy(request)
    too_far['start_state']['positions_rad'][5] = -1.300001
    too_far = attach_digest(too_far, 'request_sha256')
    with pytest.raises(ContractError, match='joint6 is outside limits'):
        validate_request(too_far)

    scan = request_fixture('MULTIVIEW_SCAN')
    scan['limits']['configured_home_start_limit_tolerance_rad'] = 0.3
    scan = attach_digest(scan, 'request_sha256')
    with pytest.raises(ContractError, match='RETURN_HOME-only'):
        validate_request(scan)


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
