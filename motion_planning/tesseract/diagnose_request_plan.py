#!/usr/bin/env python3
"""Run the ROS-free backend on an archived request without spool/TTL handling."""

import argparse
import json

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
    selected, segments = backend.plan(request)
    print(json.dumps({
        'request_id': request['request_id'],
        'selected': [
            {'id': item['id'], 'roll_rad': item['roll_rad']}
            for item in selected
        ],
        'segments': [
            {
                'from': item['from_viewpoint'],
                'to': item['to_viewpoint'],
                'points': len(item['points']),
                'duration_s': item['points'][-1]['time_from_start_s'],
            }
            for item in segments
        ],
        'real_arm_motion': False,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
