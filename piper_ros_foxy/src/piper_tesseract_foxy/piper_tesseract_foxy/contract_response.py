"""Fail-closed schema-v5 Tesseract response validation."""

import math

from piper_tesseract_foxy.contract_core import ContractError
from piper_tesseract_foxy.contract_hashing import verify_digest
from piper_tesseract_foxy.contract_validation import (
    angular_separation_deg,
    COMMAND_RATE_HZ,
    finite_vector,
    JOINT_NAMES,
    MAX_POINTS_PER_SEGMENT,
    MAX_SEGMENTS,
    MOVEJ_NOMINAL_VELOCITY_RAD_S,
    SCHEMA_VERSION,
    target_ray_position_matches,
    trajectory_digest,
    validate_plan_identity,
)


def validate_response(payload, request=None):
    if not isinstance(payload, dict):
        raise ContractError('response must be an object')
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ContractError('unsupported response schema_version')
    verify_digest(payload, 'response_sha256')
    plan_kind, provenance = validate_plan_identity(payload)
    if request is not None:
        if payload.get('request_id') != request.get('request_id'):
            raise ContractError('response request_id mismatch')
        if payload.get('request_sha256') != request.get('request_sha256'):
            raise ContractError('response request hash mismatch')
        if plan_kind != request.get('plan_kind'):
            raise ContractError('response plan_kind mismatch')
        if provenance != request.get('target_provenance'):
            raise ContractError('response target_provenance mismatch')
    if payload.get('status') != 'success':
        return payload
    response_seed = payload.get('deterministic_seed')
    if (
            isinstance(response_seed, bool)
            or not isinstance(response_seed, int)
            or response_seed < 1
            or response_seed > 0xffffffff):
        raise ContractError('successful response has an invalid deterministic_seed')
    if (
            request is not None
            and response_seed != request['planning']['deterministic_seed']):
        raise ContractError('response deterministic_seed mismatch')
    if payload.get('joint_names') != JOINT_NAMES:
        raise ContractError('response joint order must be joint1 through joint6')
    finite_vector(payload.get('target_center_m'), 3, 'response.target_center_m')
    selected = payload.get('selected_viewpoints')
    if not isinstance(selected, list):
        raise ContractError('successful response selected viewpoints are invalid')
    if plan_kind != 'RETURN_HOME' and not selected:
        raise ContractError('successful response has no selected viewpoints')
    if plan_kind == 'RETURN_HOME' and selected:
        raise ContractError('RETURN_HOME response contains capture viewpoints')
    selected_ids = []
    for index, viewpoint in enumerate(selected):
        if not isinstance(viewpoint, dict):
            raise ContractError(
                'selected viewpoint %d must be an object' % index)
        try:
            selected_ids.append(int(viewpoint.get('id')))
        except (TypeError, ValueError):
            raise ContractError(
                'selected viewpoint %d has an invalid id' % index)
        finite_vector(
            viewpoint.get('camera_position_m'), 3,
            'selected viewpoint camera_position_m')
        direction = finite_vector(
            viewpoint.get('look_direction'), 3,
            'selected viewpoint look_direction')
        if sum(value * value for value in direction) <= 1e-12:
            raise ContractError(
                'selected viewpoint look_direction must be non-zero')
    if len(set(selected_ids)) != len(selected_ids):
        raise ContractError('selected viewpoint ids must be unique')
    segments = payload.get('segments')
    if not isinstance(segments, list) or not segments:
        raise ContractError('successful response has no trajectory segments')
    if len(segments) > MAX_SEGMENTS:
        raise ContractError('successful response has too many trajectory segments')
    expected_segments = len(selected)
    if (
            request is not None
            and request['planning'].get(
                'include_return_home',
                bool(request['planning'].get('return_home_positions_rad')))):
        expected_segments += 1
    if len(segments) != expected_segments:
        raise ContractError(
            'selected viewpoint and trajectory segment counts differ from '
            'the capture-plus-home contract')
    for segment_index, segment in enumerate(segments):
        points = segment.get('points') if isinstance(segment, dict) else None
        if not isinstance(points, list) or len(points) < 2:
            raise ContractError(
                'segment %d must contain a scheduled Tesseract path'
                % segment_index)
        if len(points) > MAX_POINTS_PER_SEGMENT:
            raise ContractError('segment %d has too many points' % segment_index)
        recovery_used = bool(segment.get('bootstrap_recovery_used', False))
        powered_start_recovery = bool(
            segment.get('powered_start_recovery_used', False))
        direct_home = bool(
            segment.get('configured_home_direct_joint_move', False))
        collision_bypassed = bool(
            segment.get('collision_validation_bypassed', False))
        segment_home_stage = str(segment.get('home_stage', '')).strip().upper()
        if direct_home:
            expected_home_stage = str(
                request['planning'].get('home_stage', '')
                or 'CONFIGURED_HOME').strip().upper() if request is not None else ''
            declared_home_goal = finite_vector(
                segment.get('configured_home_goal_positions_rad'), 6,
                'configured home direct declared goal')
            if not (
                    plan_kind == 'RETURN_HOME'
                    and segment_index == 0
                    and len(segments) == 1
                    and segment.get('is_return_home') is True
                    and len(points) == 2
                    and collision_bypassed
                    and segment.get('validation') ==
                    'configured_home_collision_validation_bypassed'
                    and segment_home_stage == expected_home_stage):
                raise ContractError(
                    'segment %d configured-home collision bypass scope is invalid'
                    % segment_index)
            planned_home_goal = finite_vector(
                points[-1].get('positions_rad'), 6,
                'configured home direct endpoint')
            if any(
                    abs(float(actual) - float(expected)) > 1e-9
                    for actual, expected in zip(
                        planned_home_goal, declared_home_goal)):
                raise ContractError(
                    'segment %d configured-home declared goal does not match '
                    'the endpoint' % segment_index)
        elif collision_bypassed or segment_home_stage:
            raise ContractError(
                'segment %d has undeclared configured-home bypass metadata'
                % segment_index)
        startup_home_static = segment.get('startup_home_static', False)
        if not isinstance(startup_home_static, bool):
            raise ContractError(
                'segment %d startup_home_static must be boolean'
                % segment_index)
        expected_startup_home_static = bool(
            request is not None
            and request['scene'].get('startup_home_static', False))
        if startup_home_static != expected_startup_home_static:
            raise ContractError(
                'segment %d startup-home static binding mismatches the request'
                % segment_index)
        if startup_home_static and not (
                plan_kind == 'RETURN_HOME'
                and segment_index == 0
                and len(segments) == 1
                and segment.get('is_return_home') is True):
            raise ContractError(
                'segment %d startup-home static scope is invalid'
                % segment_index)
        if direct_home and (recovery_used or powered_start_recovery):
            raise ContractError(
                'segment %d configured-home direct move cannot use recovery targets'
                % segment_index)
        if recovery_used:
            recovery_end = int(segment.get(
                'bootstrap_recovery_end_point', -1))
            powered_end = int(segment.get(
                'powered_start_recovery_end_point', -1))
            ordinary_recovery = bool(
                1 <= recovery_end < len(points) - 1
                and not powered_start_recovery)
            dual_home_recovery = bool(
                plan_kind == 'RETURN_HOME'
                and segment.get('is_return_home') is True
                and 1 <= powered_end < recovery_end < len(points) - 1
                and powered_start_recovery
            )
            if not ordinary_recovery and not dual_home_recovery:
                raise ContractError(
                    'segment %d recovery target declaration is invalid'
                    % segment_index)
        elif powered_start_recovery:
            powered_only_home_recovery = bool(
                plan_kind == 'RETURN_HOME'
                and segment.get('is_return_home') is True
                and startup_home_static
                and 1 <= int(segment.get(
                    'powered_start_recovery_end_point', -1)) < len(points) - 1)
            if not powered_only_home_recovery:
                raise ContractError(
                    'segment %d powered-start-only recovery declaration '
                    'is invalid' % segment_index)
        elif direct_home and len(points) != 2:
            raise ContractError(
                'segment %d direct configured home must have two points'
                % segment_index)
        previous_time = -1.0
        scheduled_positions = []
        scheduled_times = []
        for point_index, point in enumerate(points):
            prefix = 'segment %d point %d' % (segment_index, point_index)
            when = float(point.get('time_from_start_s', -1.0))
            if not math.isfinite(when) or when < 0.0 or when <= previous_time:
                if point_index != 0 or when != 0.0:
                    raise ContractError('%s timestamp is not strictly increasing' % prefix)
            previous_time = when
            positions = finite_vector(
                point.get('positions_rad'), 6, prefix + ' positions')
            scheduled_positions.append(positions)
            scheduled_times.append(when)
            velocities = finite_vector(
                point.get('velocities_rad_s'), 6, prefix + ' velocities')
            accelerations = finite_vector(
                point.get('accelerations_rad_s2'), 6,
                prefix + ' accelerations')
            if (
                    any(abs(value) > 1e-12 for value in velocities)
                    or any(abs(value) > 1e-12 for value in accelerations)):
                raise ContractError(
                    '%s derivatives must be zero transport placeholders'
                    % prefix)
            if point_index:
                period = 1.0 / COMMAND_RATE_HZ
                if when - float(points[point_index - 1].get(
                        'time_from_start_s', 0.0)) < period - 1e-6:
                    raise ContractError(
                        '%s exceeds the scheduled command rate' % prefix)
            if request is not None:
                position_limits = request['limits']['position_rad']
                velocity_limits = request['limits']['max_velocity_rad_s']
                acceleration_limits = request[
                    'limits']['max_acceleration_rad_s2']
                for joint_index in range(6):
                    low, high = position_limits[joint_index]
                    if (
                            positions[joint_index] < float(low) - 1e-9
                            or positions[joint_index] > float(high) + 1e-9):
                        start_tolerance = float(request['limits'].get(
                            'bootstrap_start_limit_tolerance_rad', 0.0))
                        exact_start = request['start_state']['positions_rad']
                        allowed_recovery_start = (
                            request['plan_kind'] == 'ROUGH_ACQUISITION'
                            and recovery_used
                            and segment_index == 0
                            and point_index <= int(segment.get(
                                'bootstrap_recovery_end_point', -1))
                            and (
                                point_index != 0
                                or abs(
                                    positions[joint_index]
                                    - float(exact_start[joint_index])) <= 1e-9)
                            and max(
                                float(low) - positions[joint_index],
                                positions[joint_index] - float(high))
                            <= start_tolerance + 1e-9)
                        configured_home_start_tolerance = float(
                            request['limits'].get(
                                'configured_home_start_limit_tolerance_rad',
                                0.0))
                        allowed_configured_home_start = (
                            request['plan_kind'] == 'RETURN_HOME'
                            and direct_home
                            and segment_index == 0
                            and point_index == 0
                            and abs(
                                positions[joint_index]
                                - float(exact_start[joint_index])) <= 1e-9
                            and max(
                                float(low) - positions[joint_index],
                                positions[joint_index] - float(high))
                            <= configured_home_start_tolerance + 1e-9)
                        if not (
                                allowed_recovery_start
                                or allowed_configured_home_start):
                            raise ContractError(
                                '%s exceeds a position limit' % prefix)
                    if abs(velocities[joint_index]) > (
                            float(velocity_limits[joint_index]) + 1e-9):
                        raise ContractError(
                            '%s exceeds a velocity limit' % prefix)
                    if abs(accelerations[joint_index]) > (
                            float(acceleration_limits[joint_index]) + 1e-9):
                        raise ContractError(
                            '%s exceeds an acceleration limit' % prefix)
                if point_index and not direct_home:
                    maximum_step = float(request['planning'][
                        'max_execution_joint_step_rad'])
                    previous_positions = finite_vector(
                        points[point_index - 1].get('positions_rad'), 6,
                        prefix + ' previous positions')
                    if any(
                            abs(current - previous) > maximum_step + 1e-9
                            for current, previous in zip(
                                positions, previous_positions)):
                        raise ContractError(
                            '%s exceeds the scheduled joint-step ceiling'
                            % prefix)
        if request is not None and not direct_home:
            speed_scale = float(request['planning'][
                'effective_speed_percent']) / 100.0
            # MotionCtrl_2 interprets speed as a percentage of the qualified
            # PiPER MoveJ model (J1-J5 5 rad/s, J6 3 rad/s). The queried
            # motor-limit record is health/hash evidence and must not apply
            # the percentage to the transport schedule a second time.
            velocity_limits = [
                float(value) * speed_scale
                for value in MOVEJ_NOMINAL_VELOCITY_RAD_S]
            for point_index in range(1, len(scheduled_positions)):
                interval = scheduled_times[point_index] - scheduled_times[
                    point_index - 1]
                velocity = [
                    (current - previous) / interval
                    for current, previous in zip(
                        scheduled_positions[point_index],
                        scheduled_positions[point_index - 1],
                    )
                ]
                if any(
                        abs(value) > limit + 1e-6
                        for value, limit in zip(
                            velocity, velocity_limits)):
                    raise ContractError(
                        'segment %d exceeds a speed-scaled MoveJ model '
                        'velocity limit' % segment_index)
        if direct_home and request is not None:
            planned_start = finite_vector(
                points[0].get('positions_rad'), 6,
                'configured home direct start')
            requested_start = request['start_state']['positions_rad']
            if any(
                    abs(float(actual) - float(expected)) > 1e-9
                    for actual, expected in zip(
                        planned_start, requested_start)):
                raise ContractError(
                    'configured-home direct start does not match the request')
    if expected_segments == len(selected) + 1:
        return_segment = segments[-1]
        if return_segment.get('is_return_home') is not True:
            raise ContractError('final segment is not declared return home')
        planned_home = finite_vector(
            return_segment['points'][-1].get('positions_rad'),
            6, 'return home endpoint')
        requested_home = request['planning']['return_home_positions_rad']
        if any(
                abs(float(actual) - float(expected)) > 1e-6
                for actual, expected in zip(planned_home, requested_home)):
            raise ContractError(
                'return home endpoint does not match the request')
    binding = payload.get('trajectory_binding')
    if not isinstance(binding, dict):
        raise ContractError('successful response has no trajectory binding')
    expected = trajectory_digest(segments, binding)
    if payload.get('trajectory_sha256') != expected:
        raise ContractError('trajectory_sha256 mismatch')
    if request is not None:
        if payload['target_center_m'] != request['scene']['target_center_m']:
            raise ContractError('response target center mismatch')
        minimum = int(request['planning']['min_viewpoints'])
        maximum = int(request['planning']['max_viewpoints'])
        if len(selected) < minimum or len(selected) > maximum:
            raise ContractError(
                'response selected viewpoint count violates request')
        candidates = {
            int(item['id']): item for item in request['scene']['candidate_views']}
        for viewpoint in selected:
            candidate = candidates.get(int(viewpoint['id']))
            allowed_looks = (
                [candidate['look_direction']]
                + list(candidate.get('fallback_look_directions', []))
                if candidate is not None else [])
            position_matches = bool(
                candidate is not None
                and (
                    target_ray_position_matches(
                        candidate, viewpoint,
                        request['scene']['target_center_m'])
                    if candidate.get('candidate_geometry') == 'target_ray'
                    else viewpoint['camera_position_m']
                    == candidate['camera_position_m']))
            if candidate is None \
                    or not position_matches \
                    or viewpoint['look_direction'] not in allowed_looks:
                raise ContractError(
                    'response selected viewpoint does not match request')
            nominal = candidate['look_direction']
            reported_nominal = viewpoint.get(
                'nominal_look_direction', nominal)
            if reported_nominal != nominal:
                raise ContractError(
                    'response selected viewpoint nominal aim does not match request')
            expected_offset = angular_separation_deg(
                nominal, viewpoint['look_direction'])
            reported_offset = viewpoint.get(
                'aim_offset_deg', 0.0 if expected_offset <= 1e-6 else None)
            if reported_offset is None or abs(
                    float(reported_offset) - expected_offset) > 1e-6:
                raise ContractError(
                    'response selected viewpoint aim offset does not match request')
            expected_fallback = bool(expected_offset > 1e-6)
            reported_fallback = viewpoint.get(
                'aim_fallback_used', False if not expected_fallback else None)
            if reported_fallback is not expected_fallback:
                raise ContractError(
                    'response selected viewpoint fallback marker is invalid')
        expected_binding = {
            'request_sha256': request['request_sha256'],
            'plan_kind': request['plan_kind'],
            'target_provenance': request['target_provenance'],
            'model': request['model'],
            'calibration': request['calibration']['hand_eye_sha256'],
            'limits': request['limits'],
            'execution': {
                'effective_speed_percent':
                    request['planning']['effective_speed_percent'],
                'command_rate_hz':
                    request['planning']['command_rate_hz'],
                'timing_policy':
                    request['planning']['timing_policy'],
            },
        }
        if binding != expected_binding:
            raise ContractError('trajectory binding does not match the request')
    return payload
