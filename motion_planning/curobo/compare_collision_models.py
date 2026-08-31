#!/usr/bin/env python3
"""Compare PiPER sphere self-collision decisions with exact Tesseract meshes.

This diagnostic is command-free and CUDA-free.  It is intended to run inside
the existing rootless Tesseract environment, where both the exact collision
bindings and the repository's pure URDF/sphere checker are available.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from motion_planning.curobo.collision_qualification import sphere_overlaps
from piper_mobile_manipulation.scan_motion import URDF_JOINT_LIMITS
from piper_tesseract_foxy.worker import TesseractBackend


JOINT_NAMES = tuple('joint%d' % index for index in range(1, 7))
REFERENCE_POSES = {
    'zero': [0.0] * 6,
    'neutral': [0.0, 0.8, -0.7, 0.0, 0.7, 0.0],
    'qualified_scan': [
        0.3189509166,
        0.7800870124,
        -1.6258884709,
        -0.6660237320,
        -0.2154052887,
        0.0403545644,
    ],
    'known_retract_collision': [0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0],
}


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--curobo-config', required=True)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260831)
    parser.add_argument('--maximum-examples', type=int, default=20)
    return parser.parse_args()


def _pair_key(first, second):
    return tuple(sorted((str(first), str(second))))


def _sphere_pairs(overlaps):
    return sorted({_pair_key(item.first_link, item.second_link) for item in overlaps})


def _exact_self_pairs(minimums, collision_links):
    return sorted(
        pair for pair, distance in minimums.items()
        if pair[0] in collision_links
        and pair[1] in collision_links
        and float(distance) < 0.0
    )


def _record(name, joints, backend, kinematics, report_distance):
    minimums = backend.contact_minimums(joints, report_distance)
    exact_pairs = _exact_self_pairs(
        minimums, set(kinematics['collision_link_names']))
    overlaps = sphere_overlaps(
        kinematics['urdf_path'],
        kinematics,
        dict(zip(JOINT_NAMES, joints)),
    )
    sphere_pairs = _sphere_pairs(overlaps)
    return {
        'name': name,
        'joints_rad': [float(value) for value in joints],
        'exact_collision': bool(exact_pairs),
        'sphere_collision': bool(sphere_pairs),
        'exact_pairs': ['/'.join(pair) for pair in exact_pairs],
        'sphere_pairs': ['/'.join(pair) for pair in sphere_pairs],
        'maximum_sphere_penetration_m': (
            max(float(item.penetration_m) for item in overlaps)
            if overlaps else 0.0
        ),
    }


def compare(args):
    if args.samples < 0 or args.maximum_examples < 1:
        raise ValueError('sample and example counts are invalid')
    if args.seed < 1 or args.seed > 0xffffffff:
        raise ValueError('seed must be within 1..2^32-1')

    with open(args.curobo_config, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    kinematics = config['robot_cfg']['kinematics']
    if tuple(kinematics['cspace']['joint_names']) != JOINT_NAMES:
        raise ValueError('cuRobo joint order is not the PiPER six-joint order')
    if Path(kinematics['urdf_path']).resolve() != Path(args.urdf).resolve():
        raise ValueError('cuRobo and Tesseract must use the same planning URDF')

    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    _default, report_distance, _maximum_l1, _overrides = (
        backend.collision_policy())
    rng = np.random.default_rng(args.seed)

    records = [
        _record(name, joints, backend, kinematics, report_distance)
        for name, joints in REFERENCE_POSES.items()
    ]
    for index in range(args.samples):
        joints = rng.uniform(URDF_JOINT_LIMITS[:, 0], URDF_JOINT_LIMITS[:, 1])
        records.append(_record(
            'random_%06d' % index,
            joints,
            backend,
            kinematics,
            report_distance,
        ))

    counts = {
        'true_clear': 0,
        'true_collision': 0,
        'false_negative': 0,
        'false_positive': 0,
    }
    examples = {key: [] for key in counts}
    for record in records:
        exact = record['exact_collision']
        sphere = record['sphere_collision']
        if exact and sphere:
            outcome = 'true_collision'
        elif exact:
            outcome = 'false_negative'
        elif sphere:
            outcome = 'false_positive'
        else:
            outcome = 'true_clear'
        counts[outcome] += 1
        if len(examples[outcome]) < args.maximum_examples:
            examples[outcome].append(record)

    if sum(counts.values()) != len(records) or not all(
            math.isfinite(record['maximum_sphere_penetration_m'])
            for record in records):
        raise RuntimeError('collision comparison produced an invalid report')
    return {
        'schema_version': 1,
        'real_arm_motion': False,
        'comparison_scope': (
            'articulated sphere self-collision links; fixed-world meshes are '
            'qualified by command-free GPU world tests'
        ),
        'random_seed': args.seed,
        'random_sample_count': args.samples,
        'total_pose_count': len(records),
        'counts': counts,
        'reference_poses': records[:len(REFERENCE_POSES)],
        'examples': examples,
    }


def main():
    print(json.dumps(compare(_arguments()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
