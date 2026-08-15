"""Repeatable command-free qualification smoke for the pinned worker runtime."""

import argparse
import json
import math
import sys

import numpy as np
import yaml

from piper_mobile_manipulation.scan_motion import PiperScanKinematics
from piper_mobile_manipulation.home_pose import load_home_pose
from piper_mobile_manipulation.target_acquisition import (
    build_acquisition_viewpoints,
)
from piper_tesseract_foxy.contract import ContractError, JOINT_NAMES
from piper_tesseract_foxy.worker import (
    sdk_movej_waypoint_trajectory,
    TesseractBackend,
)


FIXTURE_START = np.asarray([
    0.31895091660115016,
    0.7800870124050843,
    -1.6258884709150951,
    -0.6660237319968092,
    -0.215405288673854,
    0.04035456437181573,
])

POWERED_START_HOME_FIXTURES = (
    np.asarray([-0.007, 0.0, -0.042, -0.036, 0.568, 0.028]),
    np.asarray([0.006, 0.0, -0.026, 0.031, 0.601, 0.036]),
    # Live 2026-08-10 disabled storage readback: J6 relaxed 0.031677 rad
    # beyond -pi after the in-limit -3.139536232-rad storage target.
    np.asarray([
        -0.019565, -0.031905, -0.001710,
        0.042429, 0.493946, -3.173270,
    ]),
)


def acquisition_candidates(viewpoints):
    """Mirror the bridge's schema-v5 centered-first candidate contract."""
    return [{
        'id': int(item['index']),
        'camera_position_m': [
            float(item['desired_camera_position'][axis])
            for axis in ('x', 'y', 'z')
        ],
        'look_direction': [
            float(item['desired_look_at_direction'][axis])
            for axis in ('x', 'y', 'z')
        ],
        'required_first': bool(item.get('keep_object_centered') is True),
    } for item in viewpoints]


def folded_home_return_regression(backend, home_positions):
    """Prove the dedicated zero-capture home transaction reaches saved home."""
    home = np.asarray(home_positions, dtype=float)
    if home.shape != (6,) or not np.all(np.isfinite(home)):
        raise RuntimeError('configured home must contain six finite joints')
    scan_pose = np.asarray([-0.468, 0.638, -0.345, -0.763, 0.700, -1.025])
    selected, segments = backend.plan({
        'plan_kind': 'RETURN_HOME',
        'scene': {
            'observation_mode': 'perception_snapshot',
            'obstacles': [],
            'candidate_views': [],
        },
        'planning': {
            'min_viewpoints': 0,
            'max_viewpoints': 0,
            'max_execution_joint_step_rad': 0.05,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
            'return_home_positions_rad': home.tolist(),
        },
        'limits': {
            'position_rad': backend.execution_position_limits,
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': backend.execution_velocity_limits,
            'max_acceleration_rad_s2': backend.execution_acceleration_limits,
            'bootstrap_start_limit_tolerance_rad': 0.0,
            'configured_home_start_limit_tolerance_rad': 0.3,
        },
        'start_state': {'positions_rad': scan_pose.tolist()},
    })
    if selected or len(segments) != 1:
        raise RuntimeError(
            'dedicated return-home plan did not produce exactly one '
            'zero-capture segment')
    segment = segments[0]
    if segment.get('is_return_home') is not True:
        raise RuntimeError('dedicated home segment is not marked return-home')
    returned = segment['points']
    if not np.allclose(returned[0]['positions_rad'], scan_pose, atol=1e-6):
        raise RuntimeError('terminal return does not start at the scan pose')
    if not np.allclose(returned[-1]['positions_rad'], home, atol=1e-9):
        raise RuntimeError('terminal return does not end at saved home')
    if not (
            segment.get('configured_home_direct_joint_move') is True
            and segment.get('collision_validation_bypassed') is True
            and int(segment.get(
                'external_floor_validation_samples', 0)) >= 2
            and segment.get('home_stage') == 'CONFIGURED_HOME'):
        raise RuntimeError(
            'configured home did not retain its dense external-floor proof')
    return {
        'trajectory_points': len(returned),
        'home_positions_rad': home.tolist(),
        'configured_home_direct_joint_move': bool(
            segment.get('configured_home_direct_joint_move', False)),
        'collision_validation_bypassed': bool(
            segment.get('collision_validation_bypassed', False)),
        'home_stage': str(segment.get('home_stage', '')),
        'external_floor_validation_samples': int(
            segment.get('external_floor_validation_samples', 0)),
        'terminal_home_recovery_used': bool(
            segment.get('bootstrap_recovery_used', False)),
        'bootstrap_recovery_joint': int(
            segment.get('bootstrap_recovery_joint', 0)),
        'bootstrap_recovery_delta_rad': float(
            segment.get('bootstrap_recovery_delta_rad', 0.0)),
        'minimum_clearance_m': float(segment['minimum_clearance_m']),
        'limiting_link_pair': segment['limiting_link_pair'],
    }


def august_11_holder_floor_incident_regression(backend, home_positions):
    """Require the captured 1.25 mm holder-floor pose to fail closed."""
    incident = np.asarray([
        -1.389240160, 2.151002196, -0.657167812,
        1.080463916, 1.230063660, -2.974289220,
    ])
    try:
        backend.plan({
            'plan_kind': 'RETURN_HOME',
            'scene': {
                'observation_mode': 'perception_snapshot',
                'obstacles': [],
                'candidate_views': [],
            },
            'planning': {
                'home_stage': 'ROUGH_HOME',
                'min_viewpoints': 0,
                'max_viewpoints': 0,
                'max_execution_joint_step_rad': 0.05,
                'roll_samples_rad': [0.0],
                'effective_speed_percent': 100.0,
                'command_rate_hz': 20.0,
                'timing_policy': 'tesseract_stream_v3',
                'return_home_positions_rad': list(home_positions),
                'joint_goal_positions_rad': list(home_positions),
            },
            'limits': {
                'position_rad': backend.execution_position_limits,
                'joint_margin_rad': 0.03,
                'max_velocity_rad_s': backend.execution_velocity_limits,
                'max_acceleration_rad_s2': backend.execution_acceleration_limits,
                'bootstrap_start_limit_tolerance_rad': 0.0,
                'configured_home_start_limit_tolerance_rad': 0.3,
            },
            'start_state': {'positions_rad': incident.tolist()},
        })
    except ContractError as error:
        rejection = str(error)
    else:
        raise RuntimeError(
            'August 11 camera-holder floor incident trajectory was accepted')
    if 'floor clearance' not in rejection:
        raise RuntimeError(
            'incident trajectory failed for the wrong reason: ' + rejection)
    return {'rejection': rejection, 'real_arm_motion': False}


def powered_start_home_return_regression(backend, home_positions):
    """Prove both measured gravity-settled starts reach configured home."""
    home = np.asarray(home_positions, dtype=float)
    results = []
    for fixture_index, start in enumerate(POWERED_START_HOME_FIXTURES):
        selected, segments = backend.plan({
            'plan_kind': 'RETURN_HOME',
            'scene': {
                'observation_mode': 'perception_snapshot',
                'startup_home_static': True,
                'obstacles': [],
                'candidate_views': [],
            },
            'planning': {
                'min_viewpoints': 0,
                'max_viewpoints': 0,
                'max_execution_joint_step_rad': 0.05,
                'roll_samples_rad': [0.0],
                'effective_speed_percent': 5.0,
                'command_rate_hz': 20.0,
                'timing_policy': 'tesseract_stream_v3',
                'return_home_positions_rad': home.tolist(),
            },
            'limits': {
                'position_rad': backend.execution_position_limits,
                'joint_margin_rad': 0.03,
                'max_velocity_rad_s': backend.execution_velocity_limits,
                'max_acceleration_rad_s2':
                    backend.execution_acceleration_limits,
                'bootstrap_start_limit_tolerance_rad': 0.0,
                'configured_home_start_limit_tolerance_rad': 0.3,
            },
            'start_state': {'positions_rad': start.tolist()},
        })
        if selected or len(segments) != 1:
            raise RuntimeError(
                'powered-start fixture %d did not produce one home segment'
                % fixture_index)
        segment = segments[0]
        points = segment.get('points', [])
        terminal_recovery_used = bool(
            segment.get('bootstrap_recovery_used', False))
        if (
                len(points) != 2
                or segment.get('configured_home_direct_joint_move') is not True
                or segment.get('collision_validation_bypassed') is not True
                or segment.get('home_stage') != 'CONFIGURED_HOME'):
            raise RuntimeError(
                'powered-start fixture %d lacks its direct-home contract'
                % fixture_index)
        if terminal_recovery_used or segment.get(
                'powered_start_recovery_used', False):
            raise RuntimeError(
                'powered-start fixture %d added a recovery target'
                % fixture_index)
        if not np.allclose(
                points[0]['positions_rad'], start, atol=1e-12):
            raise RuntimeError(
                'powered-start fixture %d lost its measured start'
                % fixture_index)
        if not np.allclose(
                points[-1]['positions_rad'], home, atol=1e-12):
            raise RuntimeError(
                'powered-start fixture %d did not end at configured home'
                % fixture_index)
        results.append({
            'start_positions_rad': start.tolist(),
            'trajectory_points': len(points),
            'configured_home_direct_joint_move': bool(
                segment.get('configured_home_direct_joint_move', False)),
            'collision_validation_bypassed': bool(
                segment.get('collision_validation_bypassed', False)),
            'powered_start_recovery_joint': int(
                segment.get('powered_start_recovery_joint', 0)),
            'powered_start_recovery_delta_rad': float(
                segment.get('powered_start_recovery_delta_rad', 0.0)),
            'powered_start_recovery_minimum_clearance_m': float(
                segment.get(
                    'powered_start_recovery_minimum_clearance_m', -1.0)),
            'middle_minimum_clearance_m': float(
                segment['minimum_clearance_m']),
            'terminal_home_recovery_used': terminal_recovery_used,
            'home_recovery_delta_rad': float(
                segment.get('bootstrap_recovery_delta_rad', 0.0)),
        })
    return results


def staged_wrist_regression(backend, home_profile):
    """Prove the persisted ready/storage J6 stages preserve joints 1-5."""
    rough = np.asarray(home_profile['positions_rad'], dtype=float)
    storage = rough.copy()
    storage[5] = float(home_profile['storage_joint6_rad'])
    results = {}
    cases = (
        ('startup_wrist', storage, rough, 'STARTUP_WRIST', True),
        ('storage_wrist', rough, storage, 'STORAGE_WRIST', False),
    )
    for label, start, goal, home_stage, startup_static in cases:
        backend.reset_scene()
        selected, segments = backend.plan({
            'plan_kind': 'RETURN_HOME',
            'scene': {
                'observation_mode': (
                    'bootstrap_static' if startup_static
                    else 'perception_snapshot'),
                'startup_home_static': bool(startup_static),
                'obstacles': [],
                'candidate_views': [],
            },
            'planning': {
                'min_viewpoints': 0,
                'max_viewpoints': 0,
                'max_execution_joint_step_rad': 0.05,
                'roll_samples_rad': [0.0],
                'effective_speed_percent': 5.0,
                'command_rate_hz': 20.0,
                'timing_policy': 'tesseract_stream_v3',
                'return_home_positions_rad': goal.tolist(),
                'home_stage': home_stage,
            },
            'limits': {
                'position_rad': backend.execution_position_limits,
                'joint_margin_rad': 0.03,
                'max_velocity_rad_s': backend.execution_velocity_limits,
                'max_acceleration_rad_s2':
                    backend.execution_acceleration_limits,
                'bootstrap_start_limit_tolerance_rad': 0.0,
                'configured_home_start_limit_tolerance_rad': 0.3,
            },
            'start_state': {'positions_rad': start.tolist()},
        })
        if selected or len(segments) != 1:
            raise RuntimeError('%s did not produce one zero-capture segment' % label)
        points = segments[0].get('points', [])
        if len(points) < 2:
            raise RuntimeError('%s returned no motion' % label)
        if not np.allclose(points[0]['positions_rad'], start, atol=1e-12):
            raise RuntimeError('%s lost its exact persisted start' % label)
        if not np.allclose(points[-1]['positions_rad'], goal, atol=1e-12):
            raise RuntimeError('%s lost its exact persisted goal' % label)
        first_five = np.asarray([
            point['positions_rad'][:5] for point in points], dtype=float)
        if not np.allclose(first_five, rough[:5], atol=1e-9):
            raise RuntimeError('%s moves a non-J6 joint' % label)
        if not (
                segments[0].get('configured_home_direct_joint_move') is True
                and segments[0].get('collision_validation_bypassed') is True
                and segments[0].get('home_stage') == home_stage):
            raise RuntimeError(
                '%s did not bind its configured direct-home stage' % label)
        results[label] = {
            'start_joint6_rad': float(start[5]),
            'goal_joint6_rad': float(goal[5]),
            'joint6_delta_rad': float(goal[5] - start[5]),
            'trajectory_points': len(points),
            'configured_home_direct_joint_move': bool(
                segments[0].get('configured_home_direct_joint_move', False)),
            'collision_validation_bypassed': bool(
                segments[0].get('collision_validation_bypassed', False)),
            'home_stage': str(segments[0].get('home_stage', '')),
            'minimum_clearance_m': float(
                segments[0]['minimum_clearance_m']),
            'limiting_link_pair': segments[0]['limiting_link_pair'],
        }
    return results


def matching_look_at_roll(transform):
    z_axis = transform[:3, 2]
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z_axis))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    roll = math.atan2(
        float(np.dot(transform[:3, 0], y_axis)),
        float(np.dot(transform[:3, 0], x_axis)),
    )
    return z_axis, roll


def detour_regression(backend):
    """Require the scheduled stream to preserve a collision-free OMPL detour."""
    start = np.asarray([0.0, 0.4, -0.8, 0.0, 0.5, 0.0])
    goal = start.copy()
    goal[1] = 1.2
    midpoint = 0.5 * (start + goal)
    obstacle_center = np.asarray(backend.robot.fk(
        'manipulator', midpoint,
        tip_link='camera_optical_frame').matrix)[:3, 3]
    half_size = np.full(3, 0.003)
    backend.add_obstacles([{
        'id': 'thin_midpoint_regression',
        'minimum_m': (obstacle_center - half_size).tolist(),
        'maximum_m': (obstacle_center + half_size).tolist(),
    }])
    if backend.state_in_collision(start) or backend.state_in_collision(goal):
        raise RuntimeError('detour fixture endpoints are not collision-free')
    if not backend.state_in_collision(midpoint):
        raise RuntimeError('detour fixture midpoint is not blocked')
    points, validation = backend.plan_segment_to_joint_goal(
        start, goal, 0.05)
    if len(points) <= 2:
        raise RuntimeError(
            'scheduled Tesseract detour collapsed to a direct endpoint')
    if any(backend.state_in_collision(point['positions_rad']) for point in points):
        raise RuntimeError('scheduled Tesseract detour contains a collision')
    return {
        'endpoint_collision_free': True,
        'midpoint_collision_detected': True,
        'scheduled_tesseract_detour_accepted': True,
        'trajectory_points': len(points),
        'minimum_clearance_m': float(validation['minimum_clearance_m']),
    }


def zero_start_acquisition_regression(backend):
    """Require a captured cold-start rough target to plan out of arm zero."""
    start = np.zeros(6)
    target = np.asarray([0.33, -0.14, 0.0])
    camera = np.asarray(backend.robot.fk(
        'manipulator', start,
        tip_link='camera_optical_frame').matrix)[:3, 3]
    viewpoints = build_acquisition_viewpoints(
        target, camera, standoff_m=0.45,
        camera_pitch_deg=-10.0, sweep_angle_deg=15.0)
    candidates = acquisition_candidates(viewpoints[:4])
    selected, segments = backend.plan({
        'scene': {
            'obstacles': [],
            'candidate_views': candidates,
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 4,
            'max_execution_joint_step_rad': 0.05,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-math.pi, math.pi],
            ],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
            'bootstrap_start_limit_tolerance_rad': 0.04,
        },
        'start_state': {'positions_rad': start.tolist()},
    })
    if not selected or not segments:
        raise RuntimeError('zero-start rough acquisition returned no plan')
    first = segments[0]
    if not np.allclose(
            np.asarray(first['points'][0]['positions_rad']), start,
            atol=1e-12):
        raise RuntimeError(
            'zero-start rough acquisition did not preserve the exact start')
    return {
        'target_center_m': target.tolist(),
        'selected_viewpoint': int(selected[0]['id']),
        'trajectory_points': len(first['points']),
        'minimum_clearance_m': float(first['minimum_clearance_m']),
        'limiting_link_pair': first['limiting_link_pair'],
        'validation_samples': int(first['discrete_samples']),
    }


def centerline_zero_start_acquisition_regression(backend):
    """Require one compact closed-loop look for a centreline target."""
    start = np.zeros(6)
    target = np.asarray([0.25, 0.0, 0.0])
    camera = np.asarray(backend.robot.fk(
        'manipulator', start,
        tip_link='camera_optical_frame').matrix)[:3, 3]
    viewpoints = build_acquisition_viewpoints(
        target, camera, standoff_m=0.45,
        camera_pitch_deg=-10.0, sweep_angle_deg=15.0,
        fallback_standoff_m=0.28)
    candidates = acquisition_candidates(viewpoints)
    selected, segments = backend.plan({
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'observation_mode': 'bootstrap_static',
            'obstacles': [],
            'candidate_views': candidates,
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'max_execution_joint_step_rad': 0.05,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-math.pi, math.pi],
            ],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
            'bootstrap_start_limit_tolerance_rad': 0.04,
        },
        'start_state': {'positions_rad': start.tolist()},
    })
    if len(selected) != 1 or len(segments) != 1:
        raise RuntimeError(
            'centreline rough acquisition did not return one search pose')
    if int(selected[0]['id']) == 0:
        raise RuntimeError(
            'centreline rough acquisition did not use a compact fallback pose')
    return {
        'target_center_m': target.tolist(),
        'candidate_viewpoints': len(candidates),
        'selected_viewpoints': [int(item['id']) for item in selected],
        'trajectory_points': [
            len(segment['points']) for segment in segments],
        'minimum_clearance_m': min(
            float(segment['minimum_clearance_m']) for segment in segments),
    }


def dual_limit_start_acquisition_regression(backend):
    """Require one validated target to recover live J2/J3 boundary drift."""
    start = np.asarray([
        -0.005302976,
        # Keep J2 deliberately below the current qualified -0.044796192
        # lower bound. The former -0.041377168 fixture stopped exercising
        # dual-limit recovery after that bound was aligned to live feedback.
        -0.050000000,
        0.034748448,
        -0.028363944,
        0.317393580,
        -0.008373120,
    ])
    target = np.asarray([0.25, 0.0, 0.0])
    camera = np.asarray(backend.robot.fk(
        'manipulator', start,
        tip_link='camera_optical_frame').matrix)[:3, 3]
    viewpoints = build_acquisition_viewpoints(
        target, camera, standoff_m=0.45,
        camera_pitch_deg=-10.0, sweep_angle_deg=15.0,
        fallback_standoff_m=0.28)
    candidates = acquisition_candidates(viewpoints)
    selected, segments = backend.plan({
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'observation_mode': 'bootstrap_static',
            'obstacles': [],
            'candidate_views': candidates,
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 5,
            'max_execution_joint_step_rad': 0.05,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-math.pi, math.pi],
            ],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
            'bootstrap_start_limit_tolerance_rad': 0.04,
        },
        'start_state': {'positions_rad': start.tolist()},
    })
    first = segments[0]
    if first.get('bootstrap_recovery_joints') != [2, 3]:
        raise RuntimeError(
            'dual-limit acquisition did not bind J2/J3 recovery')
    recovery_end = int(first.get('bootstrap_recovery_end_point', -1))
    if not (1 <= recovery_end < len(first.get('points', [])) - 1):
        raise RuntimeError(
            'dual-limit acquisition did not bind an internal recovery knot')
    return {
        'target_center_m': target.tolist(),
        'selected_viewpoints': [int(item['id']) for item in selected],
        'bootstrap_recovery_joints':
            first['bootstrap_recovery_joints'],
        'bootstrap_recovery_deltas_rad':
            first['bootstrap_recovery_deltas_rad'],
        'bootstrap_recovery_samples':
            int(first['bootstrap_recovery_samples']),
        'bootstrap_recovery_end_point': recovery_end,
        'trajectory_points': len(first['points']),
        'bootstrap_recovery_minimum_clearance_m':
            float(first['bootstrap_recovery_minimum_clearance_m']),
    }


def compact_start_acquisition_regression(backend):
    """Require automatic bounded recovery from the captured folded live pose."""
    start = np.asarray([
        -0.010100076,
        -0.033632032,
        -0.014356412,
        0.04517996,
        0.533315412,
        -0.052018008,
    ])
    target = np.asarray([0.33, -0.14, 0.0])
    camera = np.asarray(backend.robot.fk(
        'manipulator', start,
        tip_link='camera_optical_frame').matrix)[:3, 3]
    viewpoints = build_acquisition_viewpoints(
        target, camera, standoff_m=0.45,
        camera_pitch_deg=-10.0, sweep_angle_deg=15.0)
    candidates = acquisition_candidates(viewpoints[:4])
    selected, segments = backend.plan({
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'observation_mode': 'bootstrap_static',
            'obstacles': [],
            'candidate_views': candidates,
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 4,
            'max_execution_joint_step_rad': 0.05,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-math.pi, math.pi],
            ],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
            'bootstrap_start_limit_tolerance_rad': 0.04,
        },
        'start_state': {'positions_rad': start.tolist()},
    })
    if not selected or not segments:
        raise RuntimeError('compact-start rough acquisition returned no plan')
    first = segments[0]
    if not first.get('bootstrap_recovery_used'):
        raise RuntimeError('compact-start plan did not bind bootstrap recovery')
    if not np.allclose(
            np.asarray(first['points'][0]['positions_rad']), start,
            atol=1e-12):
        raise RuntimeError(
            'compact-start recovery did not preserve the exact live start')
    return {
        'target_center_m': target.tolist(),
        'selected_viewpoint': int(selected[0]['id']),
        'trajectory_points': len(first['points']),
        'minimum_clearance_m': float(first['minimum_clearance_m']),
        'limiting_link_pair': first['limiting_link_pair'],
        'validation_samples': int(first['discrete_samples']),
        'bootstrap_recovery_end_point': int(
            first['bootstrap_recovery_end_point']),
        'bootstrap_recovery_joint': int(first['bootstrap_recovery_joint']),
        'bootstrap_recovery_delta_rad': float(
            first['bootstrap_recovery_delta_rad']),
        'bootstrap_recovery_samples': int(
            first['bootstrap_recovery_samples']),
        'bootstrap_recovery_minimum_clearance_m': float(
            first['bootstrap_recovery_minimum_clearance_m']),
    }


def run(args, include_compact=True):
    def stage(name):
        print('qualification stage: ' + name, file=sys.stderr, flush=True)

    stage('model')
    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    names = backend.robot.get_joint_names('manipulator')
    if names != JOINT_NAMES:
        raise RuntimeError('unexpected manipulator joint order: %r' % names)

    with open(args.calibration, 'r', encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream)
    oracle = PiperScanKinematics(np.asarray(
        calibration['camera_to_link6']['matrix'], dtype=float))
    tesseract_link6 = np.asarray(backend.robot.fk(
        'manipulator', FIXTURE_START, tip_link='link6').matrix)
    fk_error = float(np.max(np.abs(tesseract_link6 - oracle.forward(FIXTURE_START))))
    if fk_error > 1e-9:
        raise RuntimeError('mode-0 FK mismatch: %.12g' % fk_error)

    stage('six_joint_timing')
    # Direct backend smokes do not pass through the schema-v5 request path, so
    # provide the same explicit protocol-capped execution profile here.
    backend.execution_speed_percent = 100.0
    backend.command_rate_hz = 20.0
    backend.execution_position_limits = [
        [-2.618, 2.168],
        [-0.044796192, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-math.pi, math.pi],
    ]
    backend.execution_velocity_limits = [3.0] * 6
    backend.execution_acceleration_limits = [5.0] * 6
    backend.bootstrap_start_limit_tolerance_rad = 0.0
    target = FIXTURE_START.copy()
    target[5] += 0.35
    target_pose = np.asarray(backend.robot.fk(
        'manipulator', target, tip_link='camera_optical_frame').matrix)
    look, roll = matching_look_at_roll(target_pose)
    points, validation = backend.plan_segment(FIXTURE_START, {
        'camera_position_m': target_pose[:3, 3].tolist(),
        'look_direction': look.tolist(),
    }, roll, 0.05)
    times = np.asarray([point['time_from_start_s'] for point in points])
    if len(points) < 2 or not np.all(np.diff(times) > 0.0):
        raise RuntimeError('planner did not return strictly timed trajectory points')
    for point in points:
        for field in ('positions_rad', 'velocities_rad_s', 'accelerations_rad_s2'):
            values = np.asarray(point[field], dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise RuntimeError('%s is not a finite six-vector' % field)
    max_step = max(float(np.max(np.abs(
        np.asarray(second['positions_rad']) - np.asarray(first['positions_rad'])
    ))) for first, second in zip(points[:-1], points[1:]))
    planned_j6_change = float(points[-1]['positions_rad'][5] - points[0]['positions_rad'][5])
    if abs(planned_j6_change) < 0.30:
        raise RuntimeError('J6 freedom smoke failed: delta=%.6f' % planned_j6_change)
    stage('five_percent_movej_model_timing')
    timing_positions = [FIXTURE_START.copy() for _ in range(3)]
    timing_positions[1][5] += 0.03
    timing_positions[2][5] += 0.06
    raw_timing = backend.time_parameterize_positions(timing_positions)
    source_max_velocity = max(float(np.max(np.abs(
        np.asarray(point['velocities_rad_s'], dtype=float)
    ))) for point in raw_timing)
    movej_model_timing, _ = sdk_movej_waypoint_trajectory(
        raw_timing,
        5.0,
        20.0,
        0.05,
        backend.execution_position_limits,
        [0.3] * 6,
        [0.5] * 6,
    )
    emitted_max_velocity = max(float(np.max(np.abs(
        np.asarray(second['positions_rad'], dtype=float)
        - np.asarray(first['positions_rad'], dtype=float)
    ))) / 0.05 for first, second in zip(
        movej_model_timing[:-1], movej_model_timing[1:]))
    if source_max_velocity <= 0.15:
        raise RuntimeError(
            '5 percent timing regression did not exercise ISP/MoveJ '
            'derivative separation')
    if emitted_max_velocity > 0.15 + 1e-6:
        raise RuntimeError(
            '5 percent MoveJ schedule exceeds the J6 model limit')
    stage('thin_obstacle_detour')
    backend.reset_scene()
    detour = detour_regression(backend)
    stage('zero_start_acquisition')
    backend.reset_scene()
    zero_start = zero_start_acquisition_regression(backend)
    stage('centerline_zero_start_acquisition')
    backend.reset_scene()
    centerline_zero_start = centerline_zero_start_acquisition_regression(
        backend)
    stage('dual_limit_start_acquisition')
    backend.reset_scene()
    dual_limit_start = dual_limit_start_acquisition_regression(backend)
    compact_start = None
    if include_compact:
        stage('compact_start_acquisition')
        backend.reset_scene()
        compact_start = compact_start_acquisition_regression(backend)
    # Keep established fixtures before new recovery qualifications: OMPL's
    # deterministic RNG is process-global, so earlier insertions change their
    # sample streams.
    stage('folded_home_return')
    backend.reset_scene()
    home = load_home_pose(args.home_pose)
    if home is None:
        raise RuntimeError('configured home pose is missing')
    folded_home_return = folded_home_return_regression(
        backend, home['positions_rad'])
    stage('august_11_holder_floor_incident')
    holder_floor_incident = august_11_holder_floor_incident_regression(
        backend, home['positions_rad'])
    stage('powered_start_home_return')
    powered_start_home_return = powered_start_home_return_regression(
        backend, home['positions_rad'])
    stage('staged_wrist_return')
    staged_wrist = staged_wrist_regression(backend, home)
    stage('complete')
    result = {
        'status': 'PASS',
        'backend_version': backend.version,
        'deterministic_seed': backend.deterministic_seed,
        'joint_names': names,
        'mode0_fk_max_matrix_error': fk_error,
        'trajectory_points': len(points),
        'duration_s': float(times[-1]),
        'maximum_joint_step_rad': max_step,
        'minimum_clearance_m': validation['minimum_clearance_m'],
        'limiting_link_pair': validation['limiting_link_pair'],
        'validation_samples': validation['discrete_samples'],
        'default_relative_motion_bound_m': validation[
            'default_relative_motion_bound_m'],
        'j6_start_rad': float(points[0]['positions_rad'][5]),
        'j6_end_rad': float(points[-1]['positions_rad'][5]),
        'j6_change_rad': planned_j6_change,
        'five_percent_movej_model_timing': {
            'source_max_velocity_rad_s': source_max_velocity,
            'emitted_max_velocity_rad_s': emitted_max_velocity,
            'trajectory_points': len(movej_model_timing),
            'duration_s': float(
                movej_model_timing[-1]['time_from_start_s']),
        },
        'collision_model_qualified_for_hardware': bool(
            backend.manifest.get('qualified_for_hardware', False)),
        'thin_obstacle_detour': detour,
        'folded_home_return': folded_home_return,
        'august_11_holder_floor_incident': holder_floor_incident,
        'powered_start_home_return': powered_start_home_return,
        'staged_wrist': staged_wrist,
        'zero_start_rough_acquisition': zero_start,
        'centerline_zero_start_rough_acquisition': centerline_zero_start,
        'dual_limit_start_rough_acquisition': dual_limit_start,
        'real_arm_motion': False,
    }
    if compact_start is not None:
        result['compact_start_rough_acquisition'] = compact_start
    return result


def run_compact(args):
    print(
        'qualification stage: compact_start_acquisition',
        file=sys.stderr,
        flush=True,
    )
    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    return {
        'status': 'PASS',
        'backend_version': backend.version,
        'collision_model_qualified_for_hardware': bool(
            backend.manifest.get('qualified_for_hardware', False)),
        'compact_start_rough_acquisition':
            compact_start_acquisition_regression(backend),
        'real_arm_motion': False,
    }


def run_folded_home(args):
    """Run only the saved-home terminal-recovery hardware-model contract."""
    print(
        'qualification stage: folded_home_return',
        file=sys.stderr,
        flush=True,
    )
    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    backend.execution_speed_percent = 5.0
    backend.command_rate_hz = 20.0
    backend.execution_position_limits = [
        [-2.618, 2.168],
        [-0.044796192, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-math.pi, math.pi],
    ]
    backend.execution_velocity_limits = [3.0] * 6
    backend.execution_acceleration_limits = [5.0] * 6
    backend.bootstrap_start_limit_tolerance_rad = 0.0
    return {
        'status': 'PASS',
        'backend_version': backend.version,
        'collision_model_qualified_for_hardware': bool(
            backend.manifest.get('qualified_for_hardware', False)),
        'folded_home_return': folded_home_return_regression(
            backend, load_home_pose(args.home_pose)['positions_rad']),
        'real_arm_motion': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--home-pose', required=True)
    parser.add_argument(
        '--suite', choices=('all', 'core', 'compact', 'folded_home'),
        default='all')
    args = parser.parse_args(argv)
    if args.suite == 'compact':
        result = run_compact(args)
    elif args.suite == 'folded_home':
        result = run_folded_home(args)
    else:
        result = run(args, include_compact=args.suite == 'all')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
