#!/usr/bin/env python3
"""Summarize validated Tesseract IK branches for one frozen spool request."""

import argparse
import json

import numpy as np

from diagnose_candidate_ik import collision_details
from piper_tesseract_foxy.worker import TesseractBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()

    with open(args.request, 'r', encoding='utf-8') as stream:
        request = json.load(stream)
    backend = TesseractBackend(args.urdf, args.srdf, args.manifest)
    backend.reset_scene()
    backend.add_obstacles(request['scene'].get('obstacles', []))
    start = np.asarray(request['start_state']['positions_rad'], dtype=float)
    limits = request['limits']['position_rad']
    margin = float(request['limits'].get('joint_margin_rad', 0.0))
    rolls = request['planning']['roll_samples_rad']
    report = []
    for candidate in request['scene']['candidate_views']:
        item = {'id': int(candidate['id']), 'rolls': []}
        for roll in rolls:
            goals = backend.ik_joint_goals(
                start, candidate, roll, limits, margin)
            entry = {'roll_rad': float(roll), 'valid_goal_count': len(goals)}
            if goals:
                entry['nearest_goal_rad'] = goals[0].tolist()
                entry['maximum_start_delta_rad'] = float(np.max(
                    np.abs(goals[0] - start)))
            item['rolls'].append(entry)
        item['total_valid_goals'] = sum(
            entry['valid_goal_count'] for entry in item['rolls'])
        report.append(item)
    print(json.dumps({
        'request_id': request['request_id'],
        'start_state_rad': start.tolist(),
        'start_contacts_at_zero_margin': collision_details(
            backend, start, 0.0),
        'start_contacts_at_proposal_margin': collision_details(
            backend, start,
            float(backend.manifest['proposal_collision_margin_m'])),
        'viewpoints': report,
        'real_arm_motion': False,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
