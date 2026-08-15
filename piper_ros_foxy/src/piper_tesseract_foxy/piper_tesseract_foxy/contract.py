"""Versioned, fail-closed filesystem contract for Tesseract planning."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


SCHEMA_VERSION = 5
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
TIMING_POLICY = 'tesseract_stream_v3'
COMMAND_RATE_HZ = 20.0
MOVEJ_NOMINAL_VELOCITY_RAD_S = (5.0, 5.0, 5.0, 5.0, 5.0, 3.0)
MAX_PROTOCOL_VELOCITY_RAD_S = 3.0
MAX_PROTOCOL_ACCELERATION_RAD_S2 = 5.0
MAX_BOOTSTRAP_START_LIMIT_TOLERANCE_RAD = 0.04
MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD = 0.3
PLAN_KINDS = ('MULTIVIEW_SCAN', 'ROUGH_ACQUISITION', 'RETURN_HOME')
PROVENANCE_SOURCES = ('tracked_target', 'rough_coordinate', 'configured_home')
SCENE_OBSERVATION_MODES = ('perception_snapshot', 'bootstrap_static')
SAFE_ID = re.compile(r'^[a-f0-9]{16,64}$')
SOURCE_REQUEST_ID = re.compile(r'^[A-Za-z0-9_.:-]{8,128}$')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_CANDIDATE_VIEWS = 100
MAX_OBSTACLES = 256
MAX_CAPTURE_VIEWPOINTS = 13
MAX_SEGMENTS = MAX_CAPTURE_VIEWPOINTS + 1
MAX_POINTS_PER_SEGMENT = 60000
QUEUE_NAMES = ('requests', 'processing', 'responses', 'failed')
HEALTH_FILENAME = 'worker_health.json'
MAX_HEALTH_BYTES = 16 * 1024


class ContractError(ValueError):
    """Raised when untrusted planning data violates the spool contract."""


def canonical_bytes(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        ).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ContractError('payload is not canonical finite JSON: %s' % error)


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def attach_digest(payload, field):
    value = copy.deepcopy(payload)
    value.pop(field, None)
    value[field] = sha256_value(value)
    return value


def verify_digest(payload, field):
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContractError('%s is missing or invalid' % field)
    value = copy.deepcopy(payload)
    value.pop(field, None)
    if sha256_value(value) != expected:
        raise ContractError('%s does not match canonical payload' % field)


def finite_vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ContractError('%s must contain %d values' % (label, length))
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ContractError('%s contains a boolean' % label)
        number = float(item)
        if not math.isfinite(number):
            raise ContractError('%s contains a non-finite value' % label)
        result.append(number)
    return result


def require_sha256(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[a-f0-9]{64}', value) is None:
        raise ContractError('%s must be a lowercase SHA-256 digest' % label)
    return value


def motion_limits_digest(velocities, accelerations):
    """Return the canonical controller-limit digest also published by the driver."""
    return sha256_value({
        'joint_names': list(JOINT_NAMES),
        'max_velocity_rad_s': [
            round(float(value), 9) for value in velocities],
        'max_acceleration_rad_s2': [
            round(float(value), 9) for value in accelerations],
        'source': 'piper_sdk_controller_feedback',
    })


def validate_motion_limits(limits):
    """Validate controller-derived position, velocity, and acceleration limits."""
    velocities = finite_vector(
        limits.get('max_velocity_rad_s'), 6,
        'limits.max_velocity_rad_s')
    accelerations = finite_vector(
        limits.get('max_acceleration_rad_s2'), 6,
        'limits.max_acceleration_rad_s2')
    if any(
            value <= 0.0 or value > MAX_PROTOCOL_VELOCITY_RAD_S
            for value in velocities):
        raise ContractError('controller velocity limits are invalid')
    if any(
            value <= 0.0 or value > MAX_PROTOCOL_ACCELERATION_RAD_S2
            for value in accelerations):
        raise ContractError('controller acceleration limits are invalid')
    expected = require_sha256(
        limits.get('motion_limits_sha256'),
        'limits.motion_limits_sha256')
    if motion_limits_digest(velocities, accelerations) != expected:
        raise ContractError('limits.motion_limits_sha256 does not match values')
    if limits.get('source') != 'piper_sdk_controller_feedback':
        raise ContractError('limits.source is unsupported')
    return velocities, accelerations


def validate_plan_identity(payload):
    plan_kind = payload.get('plan_kind')
    if plan_kind not in PLAN_KINDS:
        raise ContractError('plan_kind is unsupported')
    provenance = payload.get('target_provenance')
    if not isinstance(provenance, dict):
        raise ContractError('target_provenance must be an object')
    source = provenance.get('source')
    expected_source = {
        'MULTIVIEW_SCAN': 'tracked_target',
        'ROUGH_ACQUISITION': 'rough_coordinate',
        'RETURN_HOME': 'configured_home',
    }[plan_kind]
    if source != expected_source or source not in PROVENANCE_SOURCES:
        raise ContractError(
            'target_provenance.source does not match plan_kind')
    source_request_id = provenance.get('source_request_id', '')
    if plan_kind == 'ROUGH_ACQUISITION':
        if (
                not isinstance(source_request_id, str)
                or SOURCE_REQUEST_ID.fullmatch(source_request_id) is None):
            raise ContractError(
                'rough-coordinate source_request_id is missing or invalid')
    elif source_request_id not in ('', None):
        raise ContractError(
            'non-acquisition provenance must not carry source_request_id')
    frame_id = provenance.get('frame_id')
    if frame_id != 'base_link':
        raise ContractError('target_provenance.frame_id must be base_link')
    stamp = provenance.get('stamp')
    if not isinstance(stamp, dict):
        raise ContractError('target_provenance.stamp must be an object')
    sec = stamp.get('sec')
    nanosec = stamp.get('nanosec')
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        raise ContractError('target_provenance.stamp contains a boolean')
    try:
        sec = int(sec)
        nanosec = int(nanosec)
    except (TypeError, ValueError):
        raise ContractError('target_provenance.stamp is invalid')
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ContractError('target_provenance.stamp is invalid')
    return plan_kind, provenance


def validate_request(payload, now_ns=None):
    if not isinstance(payload, dict):
        raise ContractError('request must be an object')
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ContractError('unsupported request schema_version')
    plan_kind, provenance = validate_plan_identity(payload)
    request_id = payload.get('request_id')
    if not isinstance(request_id, str) or SAFE_ID.fullmatch(request_id) is None:
        raise ContractError('request_id is not a safe canonical identifier')
    verify_digest(payload, 'request_sha256')
    current_ns = time.time_ns() if now_ns is None else int(now_ns)
    created = int(payload.get('created_at_ns', 0))
    expires = int(payload.get('expires_at_ns', 0))
    if created <= 0 or expires <= created or current_ns > expires:
        raise ContractError('request is expired or has invalid timestamps')
    start = payload.get('start_state', {})
    if start.get('joint_names') != JOINT_NAMES:
        raise ContractError('start_state joint order must be joint1 through joint6')
    positions = finite_vector(start.get('positions_rad'), 6, 'start_state.positions_rad')
    scene = payload.get('scene', {})
    finite_vector(scene.get('target_center_m'), 3, 'scene.target_center_m')
    if scene.get('target_provenance') != provenance:
        raise ContractError(
            'scene.target_provenance must match target_provenance')
    observation_mode = scene.get('observation_mode')
    startup_home_static = scene.get('startup_home_static', False)
    if not isinstance(startup_home_static, bool):
        raise ContractError('scene.startup_home_static must be boolean')
    if startup_home_static and plan_kind != 'RETURN_HOME':
        raise ContractError(
            'scene.startup_home_static is RETURN_HOME-only')
    expected_observation_mode = (
        'bootstrap_static'
        if plan_kind == 'ROUGH_ACQUISITION'
        else 'perception_snapshot')
    if (
            observation_mode != expected_observation_mode
            or observation_mode not in SCENE_OBSERVATION_MODES):
        raise ContractError(
            'scene.observation_mode does not match plan_kind')
    candidates = scene.get('candidate_views')
    if not isinstance(candidates, list):
        raise ContractError('scene candidate views must be a list')
    if plan_kind != 'RETURN_HOME' and not candidates:
        raise ContractError('scene requires at least one candidate view')
    if plan_kind == 'RETURN_HOME' and candidates:
        raise ContractError('RETURN_HOME must not contain capture viewpoints')
    if len(candidates) > MAX_CANDIDATE_VIEWS:
        raise ContractError('scene has too many candidate views')
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ContractError('candidate view %d must be an object' % index)
        finite_vector(candidate.get('camera_position_m'), 3, 'candidate camera_position_m')
        direction = finite_vector(
            candidate.get('look_direction'), 3, 'candidate look_direction')
        if sum(value * value for value in direction) <= 1e-12:
            raise ContractError('candidate look_direction must be non-zero')
        required_first = candidate.get('required_first', False)
        if not isinstance(required_first, bool):
            raise ContractError('candidate required_first must be boolean')
        if plan_kind != 'ROUGH_ACQUISITION' and required_first:
            raise ContractError(
                'required_first candidate is acquisition-only')
    if (
            plan_kind == 'ROUGH_ACQUISITION'
            and not any(candidate.get('required_first', False)
                        for candidate in candidates)):
        raise ContractError(
            'ROUGH_ACQUISITION requires a centered required-first candidate')
    obstacles = scene.get('obstacles', [])
    if not isinstance(obstacles, list) or len(obstacles) > MAX_OBSTACLES:
        raise ContractError('scene obstacles must be a bounded list')
    if observation_mode == 'bootstrap_static' and obstacles:
        raise ContractError(
            'bootstrap_static scene must not contain perception obstacles')
    if startup_home_static and obstacles:
        raise ContractError(
            'startup-home static scene must not contain perception obstacles')
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict) or obstacle.get('type') != 'box':
            raise ContractError('obstacle %d is not a supported box' % index)
        minimum = finite_vector(obstacle.get('minimum_m'), 3, 'obstacle minimum')
        maximum = finite_vector(obstacle.get('maximum_m'), 3, 'obstacle maximum')
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ContractError('obstacle %d has empty bounds' % index)
    limit_record = payload.get('limits', {})
    limits = limit_record.get('position_rad')
    if not isinstance(limits, list) or len(limits) != 6:
        raise ContractError('limits.position_rad must contain six ranges')
    start_limit_tolerance = float(
        limit_record.get('bootstrap_start_limit_tolerance_rad', 0.0))
    configured_home_start_tolerance = float(
        limit_record.get(
            'configured_home_start_limit_tolerance_rad', 0.0))
    if (
            not math.isfinite(start_limit_tolerance)
            or start_limit_tolerance < 0.0
            or start_limit_tolerance >
            MAX_BOOTSTRAP_START_LIMIT_TOLERANCE_RAD):
        raise ContractError('bootstrap start limit tolerance is invalid')
    if plan_kind != 'ROUGH_ACQUISITION' and start_limit_tolerance != 0.0:
        raise ContractError(
            'bootstrap start limit tolerance is acquisition-only')
    if (
            not math.isfinite(configured_home_start_tolerance)
            or configured_home_start_tolerance < 0.0
            or configured_home_start_tolerance >
            MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD):
        raise ContractError(
            'configured home start limit tolerance is invalid')
    if (
            plan_kind != 'RETURN_HOME'
            and configured_home_start_tolerance != 0.0):
        raise ContractError(
            'configured home start limit tolerance is RETURN_HOME-only')
    outside_limits = []
    for index, bounds in enumerate(limits):
        low, high = finite_vector(bounds, 2, 'joint%d limits' % (index + 1))
        if low >= high:
            raise ContractError('joint%d limits are empty' % (index + 1))
        if positions[index] < low or positions[index] > high:
            distance = max(low - positions[index], positions[index] - high)
            acquisition_recovery = bool(
                plan_kind == 'ROUGH_ACQUISITION'
                and distance <= start_limit_tolerance)
            configured_home_recovery = bool(
                plan_kind == 'RETURN_HOME'
                and distance <= configured_home_start_tolerance)
            if not (acquisition_recovery or configured_home_recovery):
                raise ContractError(
                    'start_state joint%d is outside limits' % (index + 1))
            outside_limits.append(index)
    if plan_kind == 'ROUGH_ACQUISITION' and len(outside_limits) > 2:
        raise ContractError(
            'bootstrap start may have at most two joints outside limits')
    validate_motion_limits(limit_record)
    model = payload.get('model', {})
    if model.get('mode') != 0:
        raise ContractError('only the qualified mode-0 model is supported')
    for field in ('xacro_sha256', 'srdf_sha256', 'collision_manifest_sha256'):
        require_sha256(model.get(field), 'model.' + field)
    calibration = payload.get('calibration', {})
    require_sha256(calibration.get('hand_eye_sha256'), 'calibration.hand_eye_sha256')
    transform = calibration.get('T_link6_camera')
    if not isinstance(transform, list) or len(transform) != 4:
        raise ContractError('calibration.T_link6_camera must be a 4x4 matrix')
    for row in transform:
        finite_vector(row, 4, 'calibration.T_link6_camera row')
    planning = payload.get('planning', {})
    if planning.get('planner') != 'RRTConnect':
        raise ContractError('planning.planner must be RRTConnect')
    if planning.get('pipeline') != 'OMPL_ISP':
        raise ContractError('planning.pipeline is unsupported')
    if planning.get('timing_policy') != TIMING_POLICY:
        raise ContractError('planning.timing_policy is unsupported')
    command_rate = float(planning.get('command_rate_hz', 0.0))
    if (
            not math.isfinite(command_rate)
            or abs(command_rate - COMMAND_RATE_HZ) > 1e-9):
        raise ContractError(
            'planning.command_rate_hz must be %.0f Hz' % COMMAND_RATE_HZ)
    speed = float(planning.get('effective_speed_percent', 0.0))
    if not math.isfinite(speed) or speed < 1.0 or speed > 100.0:
        raise ContractError(
            'planning.effective_speed_percent must be within 1..100')
    deterministic_seed = planning.get('deterministic_seed')
    if (
            isinstance(deterministic_seed, bool)
            or not isinstance(deterministic_seed, int)
            or deterministic_seed < 1
            or deterministic_seed > 0xffffffff):
        raise ContractError(
            'planning.deterministic_seed must be an integer from 1 through 2^32-1')
    rolls = planning.get('roll_samples_rad')
    if not isinstance(rolls, list) or not rolls or len(rolls) > 100:
        raise ContractError('planning.roll_samples_rad must be a bounded non-empty list')
    finite_vector(rolls, len(rolls), 'planning.roll_samples_rad')
    minimum_views = int(planning.get('min_viewpoints', 0))
    maximum_views = int(planning.get('max_viewpoints', 0))
    include_return_home = planning.get(
        'include_return_home',
        plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME'))
    if not isinstance(include_return_home, bool):
        raise ContractError('planning.include_return_home must be boolean')
    if plan_kind == 'RETURN_HOME':
        if minimum_views != 0 or maximum_views != 0:
            raise ContractError('RETURN_HOME requires zero capture viewpoints')
        if not include_return_home:
            raise ContractError('RETURN_HOME must include its direct home segment')
    elif minimum_views < 1 or minimum_views > MAX_CAPTURE_VIEWPOINTS:
        raise ContractError('planning.min_viewpoints is invalid')
    if plan_kind != 'RETURN_HOME' and (
            maximum_views < minimum_views
            or maximum_views > MAX_CAPTURE_VIEWPOINTS
            or maximum_views > len(candidates)):
        raise ContractError('planning.max_viewpoints is invalid')
    if plan_kind == 'MULTIVIEW_SCAN' and minimum_views != maximum_views:
        raise ContractError(
            'MULTIVIEW_SCAN requires identical min/max viewpoints')
    if plan_kind == 'ROUGH_ACQUISITION' and (
            minimum_views != 1 or maximum_views > 5):
        raise ContractError(
            'ROUGH_ACQUISITION requires min_viewpoints=1 and max_viewpoints<=5')
    if plan_kind == 'ROUGH_ACQUISITION' and include_return_home:
        raise ContractError('ROUGH_ACQUISITION must not include a home segment')
    if (
            plan_kind == 'MULTIVIEW_SCAN'
            and not include_return_home
            and (minimum_views != 1 or maximum_views != 1)):
        raise ContractError(
            'home-free MULTIVIEW_SCAN must be one closed-loop viewpoint')
    home = planning.get('return_home_positions_rad', [])
    home_stage = str(
        planning.get('home_stage', '')
        or ('CONFIGURED_HOME' if plan_kind == 'RETURN_HOME' else '')
    ).strip().upper()
    if plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME'):
        home = finite_vector(
            home, 6, 'planning.return_home_positions_rad')
        for index, value in enumerate(home):
            low, high = limits[index]
            if value < float(low) or value > float(high):
                raise ContractError(
                    'return home joint%d is outside limits' % (index + 1))
        if plan_kind == 'RETURN_HOME':
            if home_stage not in (
                    'CONFIGURED_HOME', 'STARTUP_WRIST', 'ROUGH_HOME',
                    'STORAGE_WRIST'):
                raise ContractError('planning.home_stage is unsupported')
        elif home_stage:
            raise ContractError(
                'planning.home_stage is RETURN_HOME-only')
    elif home not in ([], None):
        raise ContractError('return home is multiview-only')
    elif home_stage:
        raise ContractError('planning.home_stage is RETURN_HOME-only')
    step = float(planning.get('max_execution_joint_step_rad', 0.0))
    if not math.isfinite(step) or step <= 0.0 or step > 0.1:
        raise ContractError('planning.max_execution_joint_step_rad is invalid')
    if planning.get('joint_specific_costs') not in ({}, None):
        raise ContractError('joint-specific costs are not permitted in the six-joint policy')
    return payload


def trajectory_digest(segments, binding):
    return sha256_value({
        'joint_names': JOINT_NAMES,
        'segments': segments,
        'binding': binding,
    })


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
            if candidate is None \
                    or viewpoint['camera_position_m'] != candidate['camera_position_m'] \
                    or viewpoint['look_direction'] != candidate['look_direction']:
                raise ContractError(
                    'response selected viewpoint does not match request')
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


class Spool:
    """Private atomic queues shared by the Foxy bridge and isolated worker."""

    def __init__(self, root):
        self.root = Path(root)
        self._prepare_root()

    def _prepare_root(self):
        if self.root.exists() and self.root.is_symlink():
            raise ContractError('spool root must not be a symlink')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        stat = self.root.stat()
        if stat.st_uid != os.getuid():
            raise ContractError('spool root is not owned by the current user')
        for name in QUEUE_NAMES:
            path = self.root / name
            if path.exists() and path.is_symlink():
                raise ContractError('spool queue must not be a symlink: %s' % name)
            path.mkdir(mode=0o700, exist_ok=True)
            path.chmod(0o700)

    def path(self, queue, request_id):
        if queue not in QUEUE_NAMES:
            raise ContractError('unknown queue')
        if not isinstance(request_id, str) or SAFE_ID.fullmatch(request_id) is None:
            raise ContractError('unsafe request identifier')
        return self.root / queue / (request_id + '.json')

    def write(self, queue, request_id, payload):
        destination = self.path(queue, request_id)
        if destination.exists():
            raise ContractError('spool destination already exists')
        temporary = destination.with_name('.%s.%d.tmp' % (destination.name, os.getpid()))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(temporary), flags, 0o600)
        try:
            data = canonical_bytes(payload)
            if len(data) > MAX_FILE_BYTES:
                raise ContractError('spool payload exceeds size limit')
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(str(temporary), str(destination))
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        directory = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination

    def read(self, queue, request_id):
        path = self.path(queue, request_id)
        stat = path.lstat()
        if not path.is_file() or path.is_symlink() or stat.st_nlink != 1:
            raise ContractError('spool entry is not a regular single-link file')
        if stat.st_size > MAX_FILE_BYTES:
            raise ContractError('spool entry exceeds size limit')
        with open(path, 'r', encoding='utf-8') as stream:
            return json.load(stream)

    def claim_next(self):
        for source in sorted((self.root / 'requests').glob('*.json')):
            request_id = source.stem
            if SAFE_ID.fullmatch(request_id) is None:
                continue
            destination = self.path('processing', request_id)
            if destination.exists():
                continue
            try:
                os.replace(str(source), str(destination))
            except FileNotFoundError:
                continue
            return request_id, self.read('processing', request_id)
        return None, None

    def pending(self, queue):
        if queue not in QUEUE_NAMES:
            raise ContractError('unknown queue')
        return sum(
            1 for path in (self.root / queue).glob('*.json')
            if SAFE_ID.fullmatch(path.stem))

    def write_health(self, payload):
        """Atomically replace the bounded worker liveness record."""
        destination = self.root / HEALTH_FILENAME
        if destination.exists() and (
                destination.is_symlink() or not destination.is_file()):
            raise ContractError('worker health destination is not a regular file')
        temporary = self.root / (
            '.%s.%d.%d.tmp'
            % (HEALTH_FILENAME, os.getpid(), time.time_ns()))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(temporary), flags, 0o600)
        try:
            data = canonical_bytes(payload)
            if len(data) > MAX_HEALTH_BYTES:
                raise ContractError('worker health payload exceeds size limit')
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise OSError('worker health write made no progress')
                offset += written
            os.fsync(descriptor)
        except Exception:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        os.replace(str(temporary), str(destination))
        directory = os.open(str(self.root), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination

    def read_health(self):
        """Read the worker liveness record without following links."""
        path = self.root / HEALTH_FILENAME
        stat = path.lstat()
        if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
            raise ContractError(
                'worker health entry is not a regular single-link file')
        if stat.st_size <= 0 or stat.st_size > MAX_HEALTH_BYTES:
            raise ContractError('worker health entry has an invalid size')
        with open(path, 'r', encoding='utf-8') as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ContractError('worker health entry must be an object')
        return value
