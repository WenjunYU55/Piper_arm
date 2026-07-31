#!/usr/bin/env python3
"""Command-free rootless diagnostic for one Cartesian viewpoint candidate."""

import argparse
import json
import math

import numpy as np

from piper_mobile_manipulation.scan_motion import URDF_JOINT_LIMITS
from piper_tesseract_foxy.worker import TesseractBackend, look_at_quaternion


def collision_details(backend, joints, margin_m=0.0):
    from tesseract_robotics.tesseract_collision import ContactResultVector

    manager = backend.robot.env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(backend.robot.env.getActiveLinkNames())
    manager.setDefaultCollisionMargin(float(margin_m))
    backend.robot.env.setState(
        ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        np.asarray(joints, dtype=float),
    )
    state = backend.robot.env.getState()
    manager.setCollisionObjectsTransform(state.link_transforms)
    contacts = backend.api['ContactResultMap']()
    manager.contactTest(
        contacts,
        backend.api['ContactRequest'](backend.api['ContactTestType_ALL']),
    )
    flattened = ContactResultVector()
    contacts.flattenMoveResults(flattened)
    return [
        {
            'links': [str(item.link_names[0]), str(item.link_names[1])],
            'distance_m': float(item.distance),
        }
        for item in (flattened[index] for index in range(len(flattened)))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--start', nargs=6, type=float, required=True)
    parser.add_argument('--position', nargs=3, type=float, required=True)
    parser.add_argument('--look', nargs=3, type=float, required=True)
    parser.add_argument(
        '--rolls', nargs='+', type=float,
        default=[-2.094395102, -1.047197551, 0.0,
                 1.047197551, 2.094395102, 3.141592654],
    )
    args = parser.parse_args()

    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    start = np.asarray(args.start, dtype=float)
    rng = np.random.default_rng(42)
    seeds = [
        ('start', start),
        ('neutral', np.asarray([0.0, 0.8, -0.7, 0.0, 0.7, 0.0])),
        ('qualified_fixture', np.asarray([
            0.3189509166, 0.7800870124, -1.6258884709,
            -0.6660237320, -0.2154052887, 0.0403545644,
        ])),
    ]
    for index in range(12):
        seed = rng.uniform(URDF_JOINT_LIMITS[:, 0], URDF_JOINT_LIMITS[:, 1])
        seeds.append(('random_%02d' % index, seed))

    report = {
        'start_contacts': collision_details(backend, start),
        'rolls': [],
        'real_arm_motion': False,
    }
    for roll in args.rolls:
        quaternion = look_at_quaternion(args.look, roll)
        pose = backend.api['Pose'].from_xyz_quat(*(args.position + quaternion))
        item = {'roll_rad': float(roll), 'attempts': []}
        for label, seed in seeds:
            attempt = {'seed': label}
            try:
                solution = backend.robot.ik(
                    'manipulator', pose, seed=seed,
                    tip_link='camera_optical_frame', all_solutions=False,
                )
                if solution is None:
                    attempt['result'] = 'NO_IK'
                else:
                    solution = np.asarray(solution, dtype=float)
                    attempt['solution_rad'] = solution.tolist()
                    attempt['finite'] = bool(np.all(np.isfinite(solution)))
                    attempt['inside_urdf_limits'] = bool(np.all(
                        solution >= URDF_JOINT_LIMITS[:, 0]) and np.all(
                        solution <= URDF_JOINT_LIMITS[:, 1]))
                    attempt['contacts'] = collision_details(backend, solution)
                    actual = np.asarray(backend.robot.fk(
                        'manipulator', solution,
                        tip_link='camera_optical_frame').matrix)
                    desired = np.asarray(pose.matrix)
                    attempt['position_error_m'] = float(np.linalg.norm(
                        actual[:3, 3] - desired[:3, 3]))
                    rotation_error = actual[:3, :3].T @ desired[:3, :3]
                    attempt['rotation_error_rad'] = float(math.acos(np.clip(
                        (np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0)))
                    attempt['result'] = 'IK'
            except Exception as error:  # diagnostic must report binding failures
                attempt['result'] = 'ERROR'
                attempt['error'] = str(error)
            item['attempts'].append(attempt)
        report['rolls'].append(item)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
