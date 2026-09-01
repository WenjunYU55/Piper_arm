#!/usr/bin/env python3
"""Create command-free planner fixtures from recorded achieved scan poses."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(
    ROOT / 'piper_ros_foxy/src/piper_tesseract_foxy'))

from motion_planning.benchmarking import (  # noqa: E402
    attach_digest, scenario_sha256, sha256_value)
from piper_tesseract_foxy.protocol.contract import (  # noqa: E402
    JOINT_NAMES, motion_limits_digest)


POSITION_LIMITS = [
    [-2.6180, 2.1680],
    [-0.044796192, 3.1400],
    [-2.9670, 0.0000],
    [-1.7450, 1.7450],
    [-1.2200, 1.2200],
    [-math.pi, math.pi],
]
REFERENCE_NEUTRAL = [0.0, 0.8, -0.7, 0.0, 0.7, 0.0]
REFERENCE_HOME = [
    -0.010187296, 0.0, -0.01692068,
    0.068485144, 0.441280868, 0.012594568,
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_vector(value, length, label):
    result = [float(item) for item in value]
    if (
            len(result) != length
            or not all(math.isfinite(item) for item in result)):
        raise ValueError('%s must contain %d finite values' % (label, length))
    return result


def normalized(value, label):
    result = finite_vector(value, 3, label)
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1e-9:
        raise ValueError('%s is zero' % label)
    return [item / norm for item in result]


def model_record(root, calibration_path):
    xacro = root / (
        'piper_ros_foxy/src/piper_description/urdf/'
        'piper_description.xacro')
    srdf = root / (
        'piper_ros_foxy/src/piper_tesseract_foxy/model/'
        'piper_bunker.srdf')
    manifest = root / (
        'piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml')
    calibration = yaml.safe_load(calibration_path.read_text(encoding='utf-8'))
    matrix = calibration.get('camera_to_link6', {}).get('matrix')
    if calibration.get('status') != 'accepted':
        raise ValueError('hand-eye calibration is not accepted')
    if len(matrix or ()) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError('hand-eye matrix is malformed')
    return ({
        'mode': 0,
        'xacro_sha256': sha256_file(xacro),
        'srdf_sha256': sha256_file(srdf),
        'collision_manifest_sha256': sha256_file(manifest),
    }, {
        'hand_eye_sha256': sha256_file(calibration_path),
        'T_link6_camera': [[float(value) for value in row] for row in matrix],
    })


def base_request(root, calibration_path, plan_kind):
    model, calibration = model_record(root, calibration_path)
    source = {
        'MULTIVIEW_SCAN': 'tracked_target',
        'ROUGH_ACQUISITION': 'rough_coordinate',
        'RETURN_HOME': 'configured_home',
    }[plan_kind]
    provenance = {
        'source': source,
        'frame_id': 'base_link',
        'stamp': {'sec': 1, 'nanosec': 0},
    }
    if plan_kind == 'ROUGH_ACQUISITION':
        provenance['source_request_id'] = 'benchmark-acquisition-reference'
    velocities = [3.0] * 6
    accelerations = [5.0] * 6
    now = time.time_ns()
    request = {
        'schema_version': 5,
        'planner_backend': 'tesseract',
        'plan_kind': plan_kind,
        'target_provenance': provenance,
        'request_id': '0' * 32,
        'created_at_ns': now,
        'expires_at_ns': now + 600_000_000_000,
        'start_state': {
            'joint_names': list(JOINT_NAMES),
            'positions_rad': list(REFERENCE_NEUTRAL),
        },
        'scene': {
            'target_center_m': [0.0, 0.0, 0.0],
            'target_provenance': provenance,
            'observation_mode': (
                'bootstrap_static'
                if plan_kind == 'ROUGH_ACQUISITION'
                else 'perception_snapshot'),
            'candidate_views': [],
            'obstacles': [],
        },
        'model': model,
        'calibration': calibration,
        'limits': {
            'position_rad': copy.deepcopy(POSITION_LIMITS),
            'max_velocity_rad_s': velocities,
            'max_acceleration_rad_s2': accelerations,
            'motion_limits_sha256': motion_limits_digest(
                velocities, accelerations),
            'source': 'piper_sdk_controller_feedback',
            'joint_margin_rad': 0.03,
            'bootstrap_start_limit_tolerance_rad': (
                0.04 if plan_kind == 'ROUGH_ACQUISITION' else 0.0),
            'configured_home_start_limit_tolerance_rad': (
                0.3 if plan_kind == 'RETURN_HOME' else 0.0),
        },
        'planning': {
            'planner': 'RRTConnect',
            'pipeline': 'OMPL_ISP',
            'deterministic_seed': 42,
            'roll_samples_rad': [
                -2.094395102, -1.047197551, 0.0,
                1.047197551, 2.094395102, 3.141592654,
            ],
            'min_viewpoints': 0 if plan_kind == 'RETURN_HOME' else 1,
            'max_viewpoints': 0 if plan_kind == 'RETURN_HOME' else 1,
            'max_execution_joint_step_rad': 0.05,
            'effective_speed_percent': 5.0,
            'command_rate_hz': 20.0,
            'timing_policy': 'timed_stream_v1',
            'joint_specific_costs': {},
            'include_return_home': plan_kind == 'RETURN_HOME',
            'return_home_positions_rad': (
                list(REFERENCE_HOME)
                if plan_kind == 'MULTIVIEW_SCAN' else []),
        },
    }
    return attach_digest(request, 'request_sha256')


def recorded_ray(generation):
    rays = generation.get('rays', [])
    if not rays:
        raise ValueError('recorded generation has no accepted ray')
    ray = rays[0]
    target = finite_vector(generation['target_center_m'], 3, 'target center')
    camera = finite_vector(ray['camera_position_m'], 3, 'camera position')
    look = normalized(ray['look_direction'], 'look direction')
    relative = [camera[i] - target[i] for i in range(3)]
    standoff = math.sqrt(sum(value * value for value in relative))
    direction = normalized(relative, 'target ray direction')
    return {
        'id': 1000000 + int(ray['ray_id']),
        'camera_position_m': camera,
        'look_direction': look,
        'candidate_geometry': 'target_ray',
        'ray_id': int(ray['ray_id']),
        'ray_direction': direction,
        'ray_min_standoff_m': min(0.28, standoff),
        'ray_max_standoff_m': max(0.50, standoff),
        'ray_preferred_max_standoff_m': max(0.50, standoff),
        'ray_scoring_standoff_m': standoff,
        'ray_standoff_m': standoff,
        'ray_probe_index': 0,
        'ray_probe_phase': 'interval_search',
        'maximum_final_aim_offset_deg': 5.0,
        'nbv_rank': 1,
    }


def finalized(request):
    return attach_digest(request, 'request_sha256')


def build_corpus(root, diagnostics_path, calibration_path, home_positions):
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    generations = diagnostics.get('generations', [])
    if len(generations) < 2:
        raise ValueError('at least two achieved ray generations are required')
    fixtures = []

    acquisition = base_request(root, calibration_path, 'ROUGH_ACQUISITION')
    first = recorded_ray(generations[0])
    first.pop('candidate_geometry', None)
    for field in tuple(first):
        if (
                field.startswith('ray_')
                or field in ('maximum_final_aim_offset_deg', 'nbv_rank')):
            first.pop(field, None)
    first['required_first'] = True
    acquisition['scene']['target_center_m'] = finite_vector(
        generations[0]['target_center_m'], 3, 'acquisition target')
    acquisition['scene']['candidate_views'] = [first]
    fixtures.append(('recorded_rough_acquisition', 'ROUGH_ACQUISITION',
                     finalized(acquisition)))

    maximum_transitions = min(len(generations) - 1, 4)
    for index in range(maximum_transitions):
        request = base_request(root, calibration_path, 'MULTIVIEW_SCAN')
        request['start_state']['positions_rad'] = finite_vector(
            generations[index]['rays'][0]['achieved_joint_positions_rad'],
            6, 'multiview start joints')
        request['scene']['target_center_m'] = finite_vector(
            generations[index + 1]['target_center_m'], 3,
            'multiview target')
        request['scene']['candidate_views'] = [
            recorded_ray(generations[index + 1])]
        request['planning'].update({
            'shortlisted_ray_count': 1,
            'expanded_ray_candidate_count': 1,
            'ray_direction_attempt_limit': 6,
        })
        fixtures.append((
            'recorded_multiview_%02d_to_%02d' % (index, index + 1),
            'MULTIVIEW_SCAN', finalized(request)))

    blocked = copy.deepcopy(fixtures[1][2])
    blocked['scene']['obstacles'] = [{
        'id': 'benchmark_deliberately_blocking_volume',
        'type': 'box',
        'minimum_m': [-2.0, -2.0, -2.0],
        'maximum_m': [2.0, 2.0, 2.0],
    }]
    fixtures.append((
        'deliberately_blocked_multiview', 'MULTIVIEW_SCAN',
        finalized(blocked)))

    terminal = base_request(root, calibration_path, 'RETURN_HOME')
    terminal['start_state']['positions_rad'] = finite_vector(
        generations[-1]['rays'][0]['achieved_joint_positions_rad'],
        6, 'return-home start joints')
    terminal['planning']['return_home_positions_rad'] = finite_vector(
        home_positions, 6, 'return-home goal')
    fixtures.append((
        'recorded_return_home', 'RETURN_HOME', finalized(terminal)))

    records = []
    for name, kind, request in fixtures:
        records.append({
            'name': name,
            'plan_kind': kind,
            'expected_role': (
                'negative_control' if name.startswith('deliberately_blocked')
                else 'policy_control' if kind == 'RETURN_HOME'
                else 'recorded_achieved_geometry'),
            'scenario_sha256': scenario_sha256(request),
            'request_template': request,
        })
    source_hash = sha256_file(diagnostics_path)
    corpus = {
        'schema_version': 1,
        'comparison_strength': 'CONTROLLED_REPLAY',
        'real_arm_motion': False,
        'source': {
            'ray_diagnostics_path': str(diagnostics_path),
            'ray_diagnostics_sha256': source_hash,
            'calibration_path': str(calibration_path),
            'calibration_sha256': sha256_file(calibration_path),
        },
        'fixture_count': len(records),
        'fixtures': records,
    }
    corpus['corpus_sha256'] = sha256_value(corpus)
    return corpus


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ray-diagnostics', required=True, type=Path)
    parser.add_argument('--calibration', type=Path, default=ROOT / (
        'L515_camera/calibration/hand_eye/'
        'session_20260808_straight_mount/calibration_result.yaml'))
    parser.add_argument(
        '--home', nargs=6, type=float, default=list(REFERENCE_HOME))
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    corpus = build_corpus(
        ROOT, args.ray_diagnostics.resolve(), args.calibration.resolve(),
        args.home)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(corpus, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(args.output)
    print('fixtures=%d corpus_sha256=%s real_arm_motion=false' % (
        corpus['fixture_count'], corpus['corpus_sha256']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
