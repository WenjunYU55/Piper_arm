"""Repeatable command-free qualification smoke for the pinned worker runtime."""

import argparse
import json
import math
import sys

import numpy as np
import yaml

from piper_mobile_manipulation.scan_motion import PiperScanKinematics
from piper_mobile_manipulation.target_acquisition import (
    build_acquisition_viewpoints,
)
from piper_tesseract_foxy.contract import ContractError, JOINT_NAMES
from piper_tesseract_foxy.worker import TesseractBackend


FIXTURE_START = np.asarray([
    0.31895091660115016,
    0.7800870124050843,
    -1.6258884709150951,
    -0.6660237319968092,
    -0.215405288673854,
    0.04035456437181573,
])


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
    """Require one-target execution to reject a path needing an OMPL detour."""
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
    try:
        backend.plan_segment_to_joint_goal(start, goal, 0.025)
    except ContractError as error:
        rejection = str(error)
    else:
        raise RuntimeError(
            'one-target SDK MoveJ plan accepted a blocked direct interpolation')
    return {
        'endpoint_collision_free': True,
        'midpoint_collision_detected': True,
        'direct_sdk_movej_rejected': True,
        'rejection': rejection,
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
    candidates = [{
        'id': int(item['index']),
        'camera_position_m': [
            float(item['desired_camera_position'][axis])
            for axis in ('x', 'y', 'z')
        ],
        'look_direction': [
            float(item['desired_look_at_direction'][axis])
            for axis in ('x', 'y', 'z')
        ],
    } for item in viewpoints[:4]]
    selected, segments = backend.plan({
        'scene': {
            'obstacles': [],
            'candidate_views': candidates,
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 4,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-2.0944, 2.0944],
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
    """Require the compact fallback to cover an absent centreline target."""
    start = np.zeros(6)
    target = np.asarray([0.25, 0.0, 0.0])
    camera = np.asarray(backend.robot.fk(
        'manipulator', start,
        tip_link='camera_optical_frame').matrix)[:3, 3]
    viewpoints = build_acquisition_viewpoints(
        target, camera, standoff_m=0.45,
        camera_pitch_deg=-10.0, sweep_angle_deg=15.0,
        fallback_standoff_m=0.30)
    candidates = [{
        'id': int(item['index']),
        'camera_position_m': [
            float(item['desired_camera_position'][axis])
            for axis in ('x', 'y', 'z')
        ],
        'look_direction': [
            float(item['desired_look_at_direction'][axis])
            for axis in ('x', 'y', 'z')
        ],
    } for item in viewpoints]
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
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-2.0944, 2.0944],
            ],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
            'bootstrap_start_limit_tolerance_rad': 0.04,
        },
        'start_state': {'positions_rad': start.tolist()},
    })
    if len(selected) != 5 or len(segments) != 5:
        raise RuntimeError(
            'centreline rough acquisition did not return five search poses')
    if not any(int(item['id']) >= 5 for item in selected):
        raise RuntimeError(
            'centreline rough acquisition did not use compact fallback poses')
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
        fallback_standoff_m=0.30)
    candidates = [{
        'id': int(item['index']),
        'camera_position_m': [
            float(item['desired_camera_position'][axis])
            for axis in ('x', 'y', 'z')
        ],
        'look_direction': [
            float(item['desired_look_at_direction'][axis])
            for axis in ('x', 'y', 'z')
        ],
    } for item in viewpoints]
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
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-2.0944, 2.0944],
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
    if len(first.get('points', [])) != 3:
        raise RuntimeError(
            'dual-limit acquisition did not bind one recovery target')
    return {
        'target_center_m': target.tolist(),
        'selected_viewpoints': [int(item['id']) for item in selected],
        'bootstrap_recovery_joints':
            first['bootstrap_recovery_joints'],
        'bootstrap_recovery_deltas_rad':
            first['bootstrap_recovery_deltas_rad'],
        'bootstrap_recovery_samples':
            int(first['bootstrap_recovery_samples']),
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
    candidates = [{
        'id': int(item['index']),
        'camera_position_m': [
            float(item['desired_camera_position'][axis])
            for axis in ('x', 'y', 'z')
        ],
        'look_direction': [
            float(item['desired_look_at_direction'][axis])
            for axis in ('x', 'y', 'z')
        ],
    } for item in viewpoints[:4]]
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
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
        },
        'limits': {
            'position_rad': [
                [-2.618, 2.168],
                [-0.044796192, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-2.0944, 2.0944],
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
    backend.command_rate_hz = 100.0
    backend.execution_position_limits = [
        [-2.618, 2.168],
        [-0.044796192, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.0944, 2.0944],
    ]
    backend.execution_velocity_limits = [3.0] * 6
    backend.execution_acceleration_limits = [5.0] * 6
    target = FIXTURE_START.copy()
    target[5] += 0.35
    target_pose = np.asarray(backend.robot.fk(
        'manipulator', target, tip_link='camera_optical_frame').matrix)
    look, roll = matching_look_at_roll(target_pose)
    points, validation = backend.plan_segment(FIXTURE_START, {
        'camera_position_m': target_pose[:3, 3].tolist(),
        'look_direction': look.tolist(),
    }, roll, 0.025)
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
        'collision_model_qualified_for_hardware': bool(
            backend.manifest.get('qualified_for_hardware', False)),
        'thin_obstacle_detour': detour,
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument(
        '--suite', choices=('all', 'core', 'compact'), default='all')
    args = parser.parse_args(argv)
    if args.suite == 'compact':
        result = run_compact(args)
    else:
        result = run(args, include_compact=args.suite == 'all')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
