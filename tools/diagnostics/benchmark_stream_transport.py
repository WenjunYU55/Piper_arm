#!/usr/bin/env python3
"""Offline transport-density comparison; never imports ROS, CUDA or the SDK."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'piper_ros_foxy/src/piper_mobile_manipulation'))

from motion_planning.curobo.adapter import normalize_trajectory  # noqa: E402
from piper_mobile_manipulation.execution.trajectory import TrajectoryRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('responses', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory already exists; choose a new path')
    rows = []
    for path in args.responses:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get('status') != 'success':
            continue
        execution = payload['trajectory_binding']['execution']
        if float(execution['command_rate_hz']) != 20.:
            raise ValueError('comparison requires recorded 20 Hz inputs')
        speed = float(execution['effective_speed_percent'])
        for segment_id, segment in enumerate(payload['segments']):
            if segment.get('configured_home_direct_joint_move'):
                continue
            original = segment['points']
            # Only uniform recorded schedules give a controlled density-only
            # comparison; do not silently retime historical sparse output.
            if any(abs(b['time_from_start_s'] - a['time_from_start_s'] - .05) > 1e-8
                   for a, b in zip(original, original[1:])):
                raise ValueError('%s segment %d is not uniform 20 Hz' % (path, segment_id))
            positions = [point['positions_rad'] for point in original]
            for rate in (20., 100., 200.):
                points = normalize_trajectory(positions, .05, speed, command_rate_hz=rate)
                factor = int(rate / 20)
                if [p['positions_rad'] for p in points[::factor]] != positions:
                    raise ValueError('source point retention check failed')
                times = tuple(point['time_from_start_s'] for point in points)
                if abs(times[-1] - original[-1]['time_from_start_s']) > 1e-8:
                    raise ValueError('source duration preservation check failed')
                costs = []
                for _ in range(1000):
                    started = time.perf_counter_ns()
                    TrajectoryRunner.stream_decision(len(times) // 2, times, times[len(times) // 2])
                    costs.append((time.perf_counter_ns() - started) / 1000)
                maximum_step = max(abs(a - b)
                    for left, right in zip(points, points[1:])
                    for a, b in zip(left['positions_rad'], right['positions_rad']))
                rows.append(dict(source=str(path.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                                 segment=segment_id, rate_hz=rate, points=len(points),
                                 duration_sec=times[-1], maximum_command_step_rad=maximum_step,
                                 scheduler_median_us=statistics.median(costs),
                                 scheduler_p99_us=sorted(costs)[989],
                                 evidence='OFFLINE_TRANSPORT_REPLAY_NOT_PHYSICAL'))
    if not rows:
        raise ValueError('no eligible recorded segments')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'transport.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'README.md').write_text(
        '# Offline transport comparison\n\n'
        'Recorded 20 Hz positions replayed at 20/100/200 Hz. Source vertices and '
        'total duration verified unchanged. Intermediate samples remain on each '
        'source segment. Scheduler timing excludes ROS, CAN, safety checks and '
        'physical tracking: this does not prove achieved rate or smoothness.\n\n'
        'Command: ' + repr(sys.argv) + '\n', encoding='utf-8')
    print(args.output.resolve())
    print('Compared %d segments at three rates.' % (len(rows) // 3))


if __name__ == '__main__':
    main()
