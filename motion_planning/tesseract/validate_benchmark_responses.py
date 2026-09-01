#!/usr/bin/env python3
"""Apply the exact Tesseract collision policy to benchmark trajectories."""

import argparse
import json
import os
from pathlib import Path
import time

from piper_tesseract_foxy.protocol.contract import validate_response
from piper_tesseract_foxy.worker import TesseractBackend


def validate_item(backend, item):
    request = item['request']
    response = item['response']
    validate_response(response, request)
    backend.reset_scene()
    backend.add_obstacles(request['scene'].get('obstacles', []))
    backend.execution_position_limits = request['limits']['position_rad']
    backend.execution_velocity_limits = request['limits']['max_velocity_rad_s']
    backend.execution_acceleration_limits = request[
        'limits']['max_acceleration_rad_s2']
    backend.bootstrap_start_limit_tolerance_rad = float(
        request['limits'].get('bootstrap_start_limit_tolerance_rad', 0.0))
    backend.execution_speed_percent = float(
        request['planning']['effective_speed_percent'])
    backend.command_rate_hz = float(request['planning']['command_rate_hz'])
    backend.planning_deadline_monotonic = None
    segment_reports = []
    started = time.monotonic()
    status = 'passed'
    reason = 'all trajectory segments passed exact dense Tesseract validation'
    try:
        for index, segment in enumerate(response.get('segments', [])):
            result = backend.final_validate(segment.get('points', []))
            segment_reports.append({
                'segment_index': index,
                'status': 'passed',
                **result,
            })
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        status = 'failed'
        reason = str(error)
        segment_reports.append({
            'segment_index': len(segment_reports),
            'status': 'failed',
            'reason': str(error),
        })
    elapsed = time.monotonic() - started
    return {
        'trial_key': item['trial_key'],
        'backend': response.get('backend', ''),
        'plan_kind': request.get('plan_kind', ''),
        'status': status,
        'reason': reason,
        'validation_wall_sec': elapsed,
        'segment_reports': segment_reports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--urdf', default=os.environ.get(
        'PIPER_TESSERACT_URDF', '/models/piper_planning.urdf'))
    parser.add_argument('--srdf', default=os.environ.get(
        'PIPER_TESSERACT_SRDF', '/models/piper_bunker.srdf'))
    parser.add_argument('--collision-manifest', default=os.environ.get(
        'PIPER_TESSERACT_COLLISION_MANIFEST', '/models/collision_model.yaml'))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    backend = TesseractBackend(
        args.urdf, args.srdf, args.collision_manifest)
    results = [validate_item(backend, item) for item in payload['items']]
    result = {
        'schema_version': 1,
        'validator': 'tesseract_exact_dense_collision_policy',
        'validator_backend_version': backend.version,
        'real_arm_motion': False,
        'result_count': len(results),
        'results': results,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(',', ':')),
        encoding='utf-8')
    return 0 if all(item['status'] == 'passed' for item in results) else 4


if __name__ == '__main__':
    raise SystemExit(main())
