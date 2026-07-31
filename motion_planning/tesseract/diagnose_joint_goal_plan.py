#!/usr/bin/env python3
"""Compare raw OMPL and the full Tesseract freespace pipeline for one goal."""

import argparse
import json

import numpy as np

from piper_tesseract_foxy.contract import JOINT_NAMES
from piper_tesseract_foxy.worker import TesseractBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--view-index', type=int, default=0)
    args = parser.parse_args()
    with open(args.request, 'r', encoding='utf-8') as stream:
        request = json.load(stream)
    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    backend.add_obstacles(request['scene'].get('obstacles', []))
    start = np.asarray(request['start_state']['positions_rad'], dtype=float)
    candidate = request['scene']['candidate_views'][args.view_index]
    ranked = []
    for roll in request['planning']['roll_samples_rad']:
        goals = backend.ik_joint_goals(
            start, candidate, roll, request['limits']['position_rad'],
            request['limits'].get('joint_margin_rad', 0.0))
        for goal in goals[:2]:
            ranked.append((float(np.max(np.abs(goal - start))), float(roll), goal))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise RuntimeError('no validated IK goal')
    _, roll, goal = ranked[0]
    program = backend.api['MotionProgram'](
        'manipulator', tcp_frame='camera_optical_frame').set_joint_names(JOINT_NAMES)
    program.start_at(backend.api['JointTarget'](start))
    program.move_to(backend.api['JointTarget'](goal))
    results = {}
    for pipeline in ('OMPLPipeline', 'FreespacePipeline'):
        result = backend.api['plan_ompl'](
            backend.robot, program, pipeline=pipeline,
            profiles=backend.planning_profiles())
        results[pipeline] = {
            'successful': bool(result.successful),
            'message': str(result.message),
            'trajectory_points': len(result.trajectory),
        }
    print(json.dumps({
        'request_id': request['request_id'],
        'view_id': candidate['id'],
        'roll_rad': roll,
        'start_rad': start.tolist(),
        'goal_rad': goal.tolist(),
        'results': results,
        'real_arm_motion': False,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
