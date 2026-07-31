#!/usr/bin/env python3
"""Search command-free Tesseract IK for practical cube-viewpoint geometry."""

import argparse
import json

import numpy as np

from diagnose_candidate_ik import collision_details
from piper_mobile_manipulation.scan_motion import URDF_JOINT_LIMITS, orbit_camera_view
from piper_tesseract_foxy.worker import TesseractBackend, look_at_quaternion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--start', nargs=6, type=float, required=True)
    parser.add_argument('--target', nargs=3, type=float, required=True)
    args = parser.parse_args()

    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    start = np.asarray(args.start, dtype=float)
    target = np.asarray(args.target, dtype=float)
    seeds = [
        start,
        np.asarray([0.0, 0.8, -0.7, 0.0, 0.7, 0.0]),
        np.asarray([0.3189509166, 0.7800870124, -1.6258884709,
                    -0.6660237320, -0.2154052887, 0.0403545644]),
    ]
    rolls = [-2.094395102, -1.047197551, 0.0,
             1.047197551, 2.094395102, 3.141592654]
    successes = []
    attempts = 0
    for radius in (0.30, 0.34, 0.38, 0.42, 0.45):
        for pitch in (-15.0, -25.0, -35.0, -45.0, -55.0, -65.0):
            for angle in range(-180, 181, 15):
                position, look = orbit_camera_view(target, angle, radius, pitch)
                accepted = None
                for roll in rolls:
                    quaternion = look_at_quaternion(look, roll)
                    pose = backend.api['Pose'].from_xyz_quat(*(
                        position.tolist() + quaternion))
                    for seed_index, seed in enumerate(seeds):
                        attempts += 1
                        solution = backend.robot.ik(
                            'manipulator', pose, seed=seed,
                            tip_link='camera_optical_frame', all_solutions=False,
                        )
                        if solution is None:
                            continue
                        solution = np.asarray(solution, dtype=float)
                        if (not np.all(np.isfinite(solution))
                                or np.any(solution < URDF_JOINT_LIMITS[:, 0])
                                or np.any(solution > URDF_JOINT_LIMITS[:, 1])):
                            continue
                        contacts = collision_details(backend, solution)
                        if contacts:
                            continue
                        accepted = {
                            'radius_m': radius,
                            'pitch_deg': pitch,
                            'angle_deg': angle,
                            'roll_rad': roll,
                            'seed_index': seed_index,
                            'camera_position_m': position.tolist(),
                            'look_direction': look.tolist(),
                            'solution_rad': solution.tolist(),
                            'max_start_delta_rad': float(np.max(np.abs(solution - start))),
                            'l2_start_delta_rad': float(np.linalg.norm(solution - start)),
                        }
                        break
                    if accepted is not None:
                        break
                if accepted is not None:
                    successes.append(accepted)
    successes.sort(key=lambda item: (
        item['max_start_delta_rad'], item['radius_m'], item['pitch_deg'],
        abs(item['angle_deg'])))
    groups = {}
    for item in successes:
        key = 'radius_%.2f_pitch_%+.0f' % (
            item['radius_m'], item['pitch_deg'])
        group = groups.setdefault(key, {
            'radius_m': item['radius_m'],
            'pitch_deg': item['pitch_deg'],
            'angles_deg': [],
            'minimum_max_start_delta_rad': item['max_start_delta_rad'],
        })
        group['angles_deg'].append(item['angle_deg'])
        group['minimum_max_start_delta_rad'] = min(
            group['minimum_max_start_delta_rad'],
            item['max_start_delta_rad'])
    group_summary = sorted(groups.values(), key=lambda item: (
        -len(item['angles_deg']), item['minimum_max_start_delta_rad']))
    for item in group_summary:
        item['count'] = len(item['angles_deg'])
    print(json.dumps({
        'attempts': attempts,
        'success_count': len(successes),
        'ring_summary': group_summary,
        'nearest_successes': successes[:5],
        'real_arm_motion': False,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
