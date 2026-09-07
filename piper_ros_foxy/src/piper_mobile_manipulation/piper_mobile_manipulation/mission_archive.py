#!/usr/bin/env python3
"""Archive whole-scan sensor evidence in a non-critical, separate rosbag process."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import uuid

import yaml

from piper_mobile_manipulation.visibility import DEFAULT_VISIBILITY_ANGLE_DEG, visibility_tolerance

TOPICS = (
    '/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
    '/camera/color/camera_info', '/piper/sam2_target_mask', '/tf', '/tf_static',
    '/piper/reconstruction_keyframe', '/piper/scan_session_history',
    '/piper/scan_execution_status', '/piper/supervised_workflow_status',
)


def recording_command(root):
    """Disable Foxy's unflushed tail cache; bounded DDS queues isolate this writer."""
    return ['ros2', 'bag', 'record', '--output', str(root / 'bag'),
            '--max-cache-size', '0', '--max-bag-size', str(1024**3),
            '--qos-profile-overrides-path', str(root / 'qos.yaml')] + list(TOPICS)


def write_json(path, data):
    """Keep inspection metadata readable across ordinary interruption."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def record(output_root, minimum_free_bytes, task_id, config_dir=None, stop=None,
           visibility_angle=DEFAULT_VISIBILITY_ANGLE_DEG):
    """Persist a closed or incomplete archive; failure never contacts motion nodes."""
    visibility_angle = visibility_tolerance(visibility_angle)
    stop = stop or threading.Event()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(str(root)).free < minimum_free_bytes:
        raise RuntimeError('Scan recording not started: free disk below configured reserve')
    active = root / ('scan_recording_' + uuid.uuid4().hex + '.inprogress')
    active.mkdir()
    metadata = dict(schema_version=1, task_id=task_id, state='recording',
                    target_visibility_angle_tolerance_deg=visibility_angle,
                    evidence_class='ROS_SENSOR_ARCHIVE_ORIGIN_UNVERIFIED',
                    started_unix_ns=time.time_ns(), topics=list(TOPICS),
                    minimum_free_bytes=minimum_free_bytes,
                    mission_outcome=('Read recorded mission events; '
                                     'archive closure is not mission success'),
                    dropped_messages=None,
                    drop_reporting=(
                        'Foxy does not expose reliable per-session transport drop totals'))
    write_json(active / 'recording.json', metadata)
    qos = {name: dict(history='keep_last', depth=8, reliability='best_effort',
                      durability='volatile') for name in TOPICS}
    qos['/tf_static'].update(durability='transient_local', reliability='reliable', depth=100)
    (active / 'qos.yaml').write_text(yaml.safe_dump(qos))
    if config_dir is not None:
        (active / 'config').mkdir()
        for name in ('reconstruction_recording_params.yaml', 'scan_execution_params.yaml',
                     'scan_quality_params.yaml'):
            shutil.copyfile(str(Path(config_dir) / name), str(active / 'config' / name))
    process = None
    reason = 'interrupted'
    try:
        with (active / 'rosbag.log').open('w') as log:
            process = subprocess.Popen(recording_command(active), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            print('Whole-scan recording: ' + str(active), flush=True)
            while not stop.wait(.5):
                if process.poll() is not None:
                    reason = 'recorder_exited'
                    break
                if shutil.disk_usage(str(active)).free < minimum_free_bytes:
                    reason = 'low_disk_space'
                    break
            else:
                reason = 'shutdown'
    except (OSError, RuntimeError) as exc:
        reason = 'recorder_error: ' + str(exc)
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                reason = 'forced_stop'
        bag_metadata = active / 'bag/metadata.yaml'
        # Foxy ros2cli returns signal.SIGINT (2) after a handled KeyboardInterrupt.
        closed = (reason == 'shutdown' and bag_metadata.is_file()
                  and process.returncode in (0, signal.SIGINT))
        metadata.update(state='closed' if closed else 'incomplete', stop_reason=reason,
                        ended_unix_ns=time.time_ns(),
                        recorder_exit_code=process.returncode if process else None)
        if bag_metadata.is_file():
            try:
                info = yaml.safe_load(bag_metadata.read_text())['rosbag2_bagfile_information']
                actual_count = 0
                for filename in info['relative_file_paths']:
                    database = (active / 'bag' / filename).resolve()
                    database.relative_to((active / 'bag').resolve())
                    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
                        actual_count += connection.execute(
                            'SELECT COUNT(*) FROM messages').fetchone()[0]
                metadata['stored_messages'] = actual_count
                if actual_count != info['message_count']:
                    closed = False
                    metadata['metadata_error'] = (
                        'Bag metadata count differs from stored SQLite rows')
                metadata['recorded_messages'] = info['message_count']
                metadata['topic_counts'] = {
                    item['topic_metadata']['name']: item['message_count']
                    for item in info['topics_with_message_count']}
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, sqlite3.Error):
                closed = False
                metadata['metadata_error'] = 'Unable to read rosbag topic counts'
        metadata['state'] = 'closed' if closed else 'incomplete'
        write_json(active / 'recording.json', metadata)
        final = active.with_suffix('.closed' if closed else '.incomplete')
        active.rename(final)
        print('Scan archive %s: %s (%s)' % (metadata['state'], final, reason), flush=True)
    return final


def main():
    """Follow launch shutdown with SIGINT so rosbag can close its database metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--minimum-free-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--task-id', default='')
    parser.add_argument('--config-dir')
    parser.add_argument('--visibility-angle', type=float, default=DEFAULT_VISIBILITY_ANGLE_DEG)
    args = parser.parse_args()
    if args.minimum_free_bytes < 0:
        parser.error('minimum free bytes must be non-negative')
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signal, _frame: stop.set())
    try:
        record(args.output_root, args.minimum_free_bytes, args.task_id, args.config_dir, stop,
               args.visibility_angle)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
