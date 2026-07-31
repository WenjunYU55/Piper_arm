#!/usr/bin/env python3
"""Search the proposal model for nearby collision-free single-joint states."""

import argparse
import json

import numpy as np

from diagnose_candidate_ik import collision_details
from piper_mobile_manipulation.scan_motion import URDF_JOINT_LIMITS
from piper_tesseract_foxy.worker import TesseractBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--start', nargs=6, type=float, required=True)
    parser.add_argument('--max-delta', type=float, default=0.6)
    parser.add_argument('--step', type=float, default=0.025)
    args = parser.parse_args()

    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    start = np.asarray(args.start, dtype=float)
    results = []
    steps = int(round(args.max_delta / args.step))
    for joint_index in range(6):
        for signed_step in range(-steps, steps + 1):
            if signed_step == 0:
                continue
            candidate = start.copy()
            candidate[joint_index] += signed_step * args.step
            if not (URDF_JOINT_LIMITS[joint_index, 0]
                    <= candidate[joint_index]
                    <= URDF_JOINT_LIMITS[joint_index, 1]):
                continue
            contacts = collision_details(backend, candidate)
            if contacts:
                continue
            camera = np.asarray(backend.robot.fk(
                'manipulator', candidate,
                tip_link='camera_optical_frame').matrix)
            results.append({
                'joint': joint_index + 1,
                'delta_rad': float(candidate[joint_index] - start[joint_index]),
                'positions_rad': candidate.tolist(),
                'camera_position_m': camera[:3, 3].tolist(),
                'camera_optical_z': camera[:3, 2].tolist(),
            })
    results.sort(key=lambda item: (abs(item['delta_rad']), item['joint']))
    print(json.dumps({
        'start_contacts': collision_details(backend, start),
        'nearest_single_joint_states': results[:20],
        'real_arm_motion': False,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
