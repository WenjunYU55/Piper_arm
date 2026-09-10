#!/usr/bin/env python3
"""Destructively compact closed RGB-D bags, retaining verified movement framesets.

Run in the sourced Foxy environment. Default is a read-only plan; --apply replaces
segments after content verification. A durable plan supports interrupted resumes.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from contextlib import closing, contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess

import yaml
from piper_mobile_manipulation.movement_recording import movement_intervals

RGB = '/camera/color/image_raw'
DEPTH = '/camera/aligned_depth_to_color/image_raw'
MASK = '/piper/sam2_target_mask'
EXECUTION = '/piper/scan_execution_status'


def atomic(path, content):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def read_db(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        yield db
    finally:
        db.close()


def image_stamp(prefix):
    # ROS 2 CDR encapsulation followed by the first Image field, Header.stamp.
    if len(prefix) != 12 or prefix[:2] not in (b'\x00\x01', b'\x00\x00'):
        raise ValueError('Unsupported Image CDR encoding')
    sec, nano = struct.unpack('<iI' if prefix[1] else '>iI', prefix[4:12])
    if not 0 <= nano < 1_000_000_000:
        raise ValueError('Invalid Image stamp')
    return sec * 1_000_000_000 + nano


def inspect(archive):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    status_type = get_message('piper_mobile_manipulation/msg/ScanExecutionStatus')
    metadata = yaml.safe_load((archive / 'bag/metadata.yaml').read_text())
    info = metadata['rosbag2_bagfile_information']
    if info.get('version') != 4 or info.get('storage_identifier') != 'sqlite3' or 'files' in info:
        raise ValueError('Unsupported bag metadata version/storage')
    paths = info['relative_file_paths']
    frames = {RGB: [], DEPTH: [], MASK: []}
    events, end, segments = [], 0, {}
    for name in paths:
        path = archive / 'bag' / name
        if path.resolve().parent != (archive / 'bag').resolve():
            raise ValueError('Escaping database path')
        if any(Path(str(path) + suffix).exists() and Path(str(path) + suffix).stat().st_size
               for suffix in ('-wal', '-journal')):
            raise ValueError('Database has an outstanding journal; refusing mutation')
        with read_db(path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables != {'topics', 'messages'}:
                raise ValueError('Unsupported database schema')
            topics = {r[0]: (r[1], r[2]) for r in db.execute('SELECT id,name,type FROM topics')}
            end = max(end, db.execute('SELECT MAX(timestamp) FROM messages').fetchone()[0] or 0)
            segments[name] = {'original_bytes': path.stat().st_size, 'keep_ids': [], 'payload_bytes': 0}
            for topic_id, (topic, kind) in topics.items():
                if topic == EXECUTION:
                    for ts, data in db.execute('SELECT timestamp,data FROM messages WHERE topic_id=?', (topic_id,)):
                        events.append((ts, deserialize_message(data, status_type).state))
                if topic in frames:
                    if kind != 'sensor_msgs/msg/Image':
                        raise ValueError('Unexpected sensor type')
                    for row, ts, prefix, size in db.execute(
                            'SELECT id,timestamp,substr(data,1,12),length(data) FROM messages WHERE topic_id=?', (topic_id,)):
                        frames[topic].append((name, row, ts, image_stamp(prefix), size))
                else:
                    for row, size in db.execute('SELECT id,length(data) FROM messages WHERE topic_id=?', (topic_id,)):
                        segments[name]['keep_ids'].append(row)
                        segments[name]['payload_bytes'] += size
    if not events:
        raise ValueError('No execution events: cannot establish movement windows')
    windows = movement_intervals(events, end)
    if not windows:
        raise ValueError('No recorded MOVING interval: leave archive intact')
    starts = [w[0] for w in windows]

    def inside(ts):
        i = bisect_right(starts, ts) - 1
        return i >= 0 and ts <= windows[i][1]

    rgb = {f[3] for f in frames[RGB]}
    depth = {f[3] for f in frames[DEPTH]}
    selected = {f[3] for topic in (RGB, DEPTH) for f in frames[topic] if inside(f[2])}
    paired = selected & rgb & depth
    for topic, entries in frames.items():
        for name, row, ts, stamp, size in entries:
            if stamp in paired or (topic == MASK and inside(ts)):
                segments[name]['keep_ids'].append(row)
                segments[name]['payload_bytes'] += size
    for segment in segments.values():
        segment['keep_ids'].sort()
    return dict(state='planned', schema_version=1, windows_ns=windows,
                paired_frames=len(paired), selected_unpaired_stamps=len(selected - paired),
                original_sensor_messages=sum(map(len, frames.values())), segments=segments,
                original_metadata=metadata,
                original_manifest=json.loads((archive / 'recording.json').read_text()))


def digest(db):
    result = hashlib.sha256()
    count = 0
    for row, topic, stamp, data in db.execute('SELECT id,topic_id,timestamp,data FROM messages ORDER BY id'):
        result.update(struct.pack('<qqqQ', row, topic, stamp, len(data)))
        result.update(data)
        count += 1
    return count, result.hexdigest()


def replace_segment(path, segment):
    if any(Path(str(path) + suffix).exists() and Path(str(path) + suffix).stat().st_size
           for suffix in ('-wal', '-journal')):
        raise ValueError('Database acquired a pending journal; refusing replacement')
    temporary = path.with_name(path.name + '.compact.tmp')
    # Only remove this tool's incomplete scratch output, never source data.
    temporary.unlink(missing_ok=True)
    Path(str(temporary) + '-journal').unlink(missing_ok=True)
    estimate = segment['payload_bytes'] * 1.05 + len(segment['keep_ids']) * 128 + 16 * 1024**2
    if shutil.disk_usage(path.parent).free < estimate + 64 * 1024**2:
        raise ValueError('Insufficient scratch space for ' + path.name)
    with read_db(path) as source, closing(sqlite3.connect(str(temporary))) as dest:
        schema = source.execute("SELECT type,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type DESC").fetchall()
        for kind, sql in schema:
            if kind == 'table':
                dest.execute(sql)
        topics = source.execute('SELECT * FROM topics ORDER BY id').fetchall()
        dest.executemany('INSERT INTO topics VALUES (?,?,?,?,?)', topics)
        expected = hashlib.sha256()
        for row in segment['keep_ids']:
            record = source.execute('SELECT id,topic_id,timestamp,data FROM messages WHERE id=?', (row,)).fetchone()
            if record is None:
                raise ValueError('Planned retained message missing')
            expected.update(struct.pack('<qqqQ', *record[:3], len(record[3])))
            expected.update(record[3])
            dest.execute('INSERT INTO messages VALUES (?,?,?,?)', record)
        for kind, sql in schema:
            if kind == 'index':
                dest.execute(sql)
        dest.commit()
        if dest.execute('PRAGMA quick_check').fetchone() != ('ok',):
            raise ValueError('Compacted database integrity check failed')
        count, actual = digest(dest)
        if count != len(segment['keep_ids']) or actual != expected.hexdigest():
            raise ValueError('Retained payload verification failed')
        if dest.execute('SELECT * FROM topics ORDER BY id').fetchall() != topics:
            raise ValueError('Topic metadata verification failed')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    for suffix in ('-wal', '-shm'):
        Path(str(path) + suffix).unlink(missing_ok=True)
    sync_directory(path.parent)
    segment.update(new_bytes=path.stat().st_size, retained_messages=count, sha256_retained_rows=actual)


def finish(archive, report):
    metadata = deepcopy(report['original_metadata'])
    info = metadata['rosbag2_bagfile_information']
    counts, first, last = Counter(), None, None
    for name in info['relative_file_paths']:
        with read_db(archive / 'bag' / name) as db:
            for topic, count in db.execute('SELECT topics.name,COUNT(messages.id) FROM topics LEFT JOIN messages ON topics.id=messages.topic_id GROUP BY topics.id'):
                counts[topic] += count
            low, high = db.execute('SELECT MIN(timestamp),MAX(timestamp) FROM messages').fetchone()
            if low is not None:
                first = low if first is None else min(low, first)
                last = high if last is None else max(high, last)
    info['message_count'] = sum(counts.values())
    info['starting_time'] = {'nanoseconds_since_epoch': first or 0}
    info['duration'] = {'nanoseconds': (last - first) if first is not None else 0}
    for topic in info['topics_with_message_count']:
        topic['message_count'] = counts[topic['topic_metadata']['name']]
    atomic(archive / 'bag/metadata.yaml', yaml.safe_dump(metadata, sort_keys=False))
    manifest = dict(report['original_manifest'])
    manifest.update(stored_messages=sum(counts.values()), recorded_messages=sum(counts.values()), topic_counts=dict(counts),
                    offline_compaction='movement_compaction.json')
    atomic(archive / 'recording.json', json.dumps(manifest, indent=2) + '\n')
    report.update(state='complete', saved_bytes=sum(s['original_bytes'] - s['new_bytes'] for s in report['segments'].values()))


def compact(archive, apply=False):
    archive = Path(archive)
    manifest = json.loads((archive / 'recording.json').read_text())
    if manifest.get('state') not in ('closed', 'incomplete'):
        raise ValueError('Live/unclosed archive: skipped')
    report_path = archive / 'movement_compaction.json'
    with (archive / '.compaction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads(report_path.read_text()) if report_path.exists() else inspect(archive)
        if any(Path(name).name != name for name in report['segments']):
            raise ValueError('Escaping database path in compaction report')
        if report['state'] == 'complete' or not apply:
            return report
        paths = list((archive / 'bag').glob('*.db3*'))
        opened = subprocess.run(['fuser'] + [str(p) for p in paths], capture_output=True)
        if opened.returncode != 1:
            raise ValueError('Archive is open in another process, or open-file check failed')
        report['state'] = 'compacting'
        atomic(report_path, json.dumps(report) + '\n')
        # Smallest retained output first frees scratch space for larger segments.
        for name, segment in sorted(report['segments'].items(), key=lambda item: item[1]['payload_bytes']):
            if 'new_bytes' in segment:
                continue
            replace_segment(archive / 'bag' / name, segment)
            atomic(report_path, json.dumps(report) + '\n')
            print(archive.name + '/' + name + ': verified and replaced', flush=True)
        finish(archive, report)
        atomic(report_path, json.dumps(report) + '\n')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    parser.add_argument('--apply', action='store_true', help='Permanently delete omitted sensor data after verification')
    args = parser.parse_args()
    results = []
    for archive in args.archives:
        try:
            report = compact(archive, args.apply)
            item = dict(archive=str(archive), state=report['state'], paired_frames=report['paired_frames'],
                        saved_bytes=report.get('saved_bytes'), windows=len(report['windows_ns']))
        except Exception as exc:
            item = dict(archive=str(archive), state='skipped_or_interrupted', reason=str(exc))
        results.append(item)
        print(json.dumps(item), flush=True)
    print(json.dumps({'results': results}), flush=True)


if __name__ == '__main__':
    main()
