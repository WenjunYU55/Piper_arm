"""Fail-closed schema-v5 Tesseract request validation."""

import math
import time

from piper_tesseract_foxy.contract_core import ContractError
from piper_tesseract_foxy.contract_hashing import verify_digest
from piper_tesseract_foxy.contract_validation import (
    angular_separation_deg,
    COMMAND_RATE_HZ,
    finite_vector,
    JOINT_NAMES,
    MAX_BOOTSTRAP_START_LIMIT_TOLERANCE_RAD,
    MAX_CANDIDATE_VIEWS,
    MAX_CAPTURE_VIEWPOINTS,
    MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD,
    MAX_FINAL_AIM_OFFSET_DEG,
    MAX_OBSTACLES,
    require_sha256,
    SAFE_ID,
    SCENE_OBSERVATION_MODES,
    SCHEMA_VERSION,
    TIMING_POLICY,
    validate_motion_limits,
    validate_plan_identity,
)


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
    target_center = finite_vector(
        scene.get('target_center_m'), 3, 'scene.target_center_m')
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
        fallbacks = candidate.get('fallback_look_directions', [])
        if not isinstance(fallbacks, list) or len(fallbacks) > 1:
            raise ContractError(
                'candidate fallback_look_directions must contain at most one direction')
        maximum_aim_offset = float(candidate.get(
            'maximum_final_aim_offset_deg', 0.0))
        if (
                not math.isfinite(maximum_aim_offset)
                or maximum_aim_offset < 0.0
                or maximum_aim_offset > MAX_FINAL_AIM_OFFSET_DEG):
            raise ContractError('candidate maximum final aim offset is invalid')
        if fallbacks and plan_kind != 'MULTIVIEW_SCAN':
            raise ContractError('aim fallback is MULTIVIEW_SCAN-only')
        for fallback in fallbacks:
            offset = angular_separation_deg(direction, fallback)
            if offset > maximum_aim_offset + 1e-9:
                raise ContractError(
                    'candidate fallback exceeds maximum final aim offset')
        required_first = candidate.get('required_first', False)
        if not isinstance(required_first, bool):
            raise ContractError('candidate required_first must be boolean')
        if plan_kind != 'ROUGH_ACQUISITION' and required_first:
            raise ContractError(
                'required_first candidate is acquisition-only')
        if candidate.get('candidate_geometry') == 'target_ray':
            if plan_kind != 'MULTIVIEW_SCAN':
                raise ContractError('target-ray candidate is scan-only')
            try:
                ray_id = int(candidate.get('ray_id', -1))
                minimum = float(candidate.get('ray_min_standoff_m'))
                maximum = float(candidate.get('ray_max_standoff_m'))
                standoff = float(candidate.get('ray_standoff_m'))
            except (TypeError, ValueError):
                raise ContractError('target-ray interval fields are malformed')
            ray_direction = finite_vector(
                candidate.get('ray_direction'), 3, 'target-ray direction')
            ray_direction_norm = math.sqrt(sum(
                value * value for value in ray_direction))
            if (
                    ray_id < 0
                    or not all(math.isfinite(value) for value in (
                        minimum, maximum, standoff))
                    or minimum <= 0.0 or maximum < minimum
                    or standoff < minimum or standoff > maximum
                    or abs(ray_direction_norm - 1.0) > 1e-6):
                raise ContractError('target-ray interval fields are invalid')
            expected_position = [
                origin + axis * standoff
                for origin, axis in zip(target_center, ray_direction)]
            if any(
                    abs(actual - expected) > 1e-6
                    for actual, expected in zip(
                        finite_vector(
                            candidate.get('camera_position_m'), 3,
                            'target-ray representative position'),
                        expected_position)):
                raise ContractError(
                    'target-ray representative position is inconsistent')
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
    shortlisted_rays = int(planning.get('shortlisted_ray_count', 0))
    expanded_ray_candidates = int(planning.get(
        'expanded_ray_candidate_count', 0))
    ray_attempt_limit = int(planning.get(
        'ray_direction_attempt_limit', 0))
    expanded = [
        candidate for candidate in candidates
        if 'ray_probe_index' in candidate]
    if expanded:
        ray_order = []
        for candidate in expanded:
            ray_id = int(candidate.get('ray_id', -1))
            probe_index = int(candidate.get('ray_probe_index', -1))
            if ray_id < 0 or probe_index < 0:
                raise ContractError('expanded ray candidate identity is invalid')
            if not ray_order or ray_order[-1] != ray_id:
                if ray_id in ray_order:
                    raise ContractError(
                        'expanded ray candidates are not direction-contiguous')
                ray_order.append(ray_id)
        if (
                shortlisted_rays != len(ray_order)
                or expanded_ray_candidates != len(expanded)
                or ray_attempt_limit < 1
                or len(ray_order) > ray_attempt_limit):
            raise ContractError(
                'expanded ray candidate bounds do not match planning metadata')
    elif any(value != 0 for value in (
            shortlisted_rays, expanded_ray_candidates, ray_attempt_limit)):
        raise ContractError(
            'non-ray request contains ray planning metadata')
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
