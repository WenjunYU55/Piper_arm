#!/usr/bin/env python3
"""Extract timestamp-ordered RGB-D after an accepted seed, without ROS nodes or mission gates."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, fields
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
import yaml

from dataset import atomic_json, intrinsics_valid, distortion_valid
from piper_mobile_manipulation.reconstruction_recording.core import RecorderConfig
from piper_mobile_manipulation.visibility import validate_transform


RGB = '/camera/color/image_raw'
DEPTH = '/camera/aligned_depth_to_color/image_raw'
INFO = '/camera/color/camera_info'
PIN = '/piper/reconstruction_keyframe'
HISTORY = '/piper/scan_session_history'
WORKFLOW = '/piper/supervised_workflow_status'
TOPICS = (RGB, DEPTH, INFO, PIN, HISTORY, WORKFLOW, '/tf', '/tf_static')


def stamp(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def match_stamps(rgb_stamps, depth_stamps, slop_ns):
    """Prefer exact matches globally; never reuse a depth frame or depend on arrival order."""
    colors, depths = sorted(set(rgb_stamps)), sorted(set(depth_stamps))
    if slop_ns < 0:
        raise ValueError('negative synchronization slop')
    available = set(depths)
    result = {s: s for s in colors if s in available}
    available.difference_update(result.values())
    remaining = sorted(available)
    for color in colors:
        if color in result:
            continue
        i = bisect_left(remaining, color)
        candidates = remaining[max(0, i-1):i+1]
        if candidates:
            chosen = min(candidates, key=lambda d: (abs(d-color), d))
            if abs(chosen-color) <= slop_ns:
                result[color] = chosen
                remaining.remove(chosen)
    return result


def calibration(info):
    return dict(K=np.asarray(info.k).reshape(3, 3).tolist(), width=info.width, height=info.height,
                distortion_model=info.distortion_model, distortion_coefficients=list(info.d),
                camera_optical_frame=info.header.frame_id)


def extract(bag, output):
    # Deserialization and TF live only in Foxy. No node, subscribers or publishers.
    from cv_bridge import CvBridge
    from rclpy.duration import Duration
    from rclpy.time import Time
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from scipy.spatial.transform import Rotation
    from tf2_ros import Buffer

    bag, output = Path(bag).resolve(), Path(output).resolve()
    if (bag / 'recording.json').is_file():
        bag = bag / 'bag'
    if output.exists():
        raise FileExistsError('Refusing existing extraction output')
    archive = json.loads((bag.parent / 'recording.json').read_text())
    if archive.get('state') not in ('closed', 'incomplete'):
        raise ValueError('Select a closed recording, not a live archive')
    config = asdict(RecorderConfig())
    keys = {f.name for f in fields(RecorderConfig)}
    for name in ('scan_quality_params.yaml', 'reconstruction_recording_params.yaml'):
        path = bag.parent / 'config' / name
        if path.is_file():
            values = yaml.safe_load(path.read_text())['/**']['ros__parameters']
            config.update({k: v for k, v in values.items() if k in keys})
    config['target_visibility_angle_tolerance_deg'] = archive['target_visibility_angle_tolerance_deg']
    RecorderConfig(**config)  # Same authoritative quality settings, no live sampling applied below.
    metadata = yaml.safe_load((bag / 'metadata.yaml').read_text())
    buffer = Buffer(cache_time=Duration(seconds=86400))
    images = {RGB: {}, DEPTH: {}}
    infos, pins, histories, terminals = {}, {}, [], []
    counts, rejects = Counter(), Counter()
    for relative in metadata['rosbag2_bagfile_information']['relative_file_paths']:
        path = (bag / relative).resolve()
        if path.parent != bag or not path.is_file():
            raise ValueError('missing or escaping bag database')
        with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as db:
            topics = {i: (name, get_message(kind)) for i, name, kind in
                      db.execute('SELECT id,name,type FROM topics') if name in TOPICS}
            for row, topic_id, recorded_ns, data in db.execute(
                    'SELECT id,topic_id,timestamp,data FROM messages ORDER BY timestamp,id'):
                if topic_id not in topics:
                    continue
                topic, kind = topics[topic_id]
                msg = deserialize_message(data, kind)
                counts[topic] += 1
                if topic in images:
                    key = stamp(msg)
                    if key in images[topic]:
                        rejects['duplicate_' + ('rgb' if topic == RGB else 'depth')] += 1
                    else:
                        images[topic][key] = (path, row, kind)
                elif topic == INFO:
                    infos[stamp(msg)] = calibration(msg)
                elif topic == PIN and msg.task_id == archive['task_id']:
                    pins[stamp(msg.rgb)] = msg
                elif topic in (HISTORY, WORKFLOW):
                    payload = json.loads(msg.data)
                    if topic == HISTORY and payload.get('accepted_views', 0) > 0:
                        histories.append(payload)
                    elif topic == WORKFLOW and payload.get('state') in ('COMPLETE', 'ABORTED'):
                        terminals.append((recorded_ns, payload))
                elif topic in ('/tf', '/tf_static'):
                    for tf in msg.transforms:
                        if topic == '/tf_static':
                            buffer.set_transform_static(tf, 'recorded_archive')
                        else:
                            buffer.set_transform(tf, 'recorded_archive')
        print('Indexed ' + relative, flush=True)
    sessions = {h['session_id'] for h in histories}
    if len(sessions) != 1:
        raise ValueError('Need exactly one accepted mission session in archive')
    history = max(histories, key=lambda h: int(h['accepted_views']))
    bound_angle = history.get('target_visibility_angle_tolerance_deg')
    if bound_angle is not None and float(bound_angle) != config['target_visibility_angle_tolerance_deg']:
        raise ValueError('Recorded visibility configuration disagrees with accepted history')
    accepted = {int(e['reconstruction_stamp_ns']): int(e['accepted_view'])
                for e in history['entries'] if e.get('reconstruction_stamp_ns')}
    first = next((s for s, view in accepted.items() if view == 1), None)
    if first is None or first not in pins:
        raise ValueError('Missing exact raw first accepted capture and target mask')
    seed = pins[first]
    expected = calibration(seed.camera_info)
    target = [history['coverage_target_center'][axis] for axis in ('x', 'y', 'z')]
    if not intrinsics_valid(expected['K'], expected['width'], expected['height']) or not distortion_valid(
            expected['distortion_model'], expected['distortion_coefficients']):
        raise ValueError('Invalid accepted-seed calibration')
    matching_terminal = [(t, p) for t, p in terminals
                         if p.get('session_id') in (None, history['session_id']) and t >= first]
    end_ns, terminal = min(matching_terminal, default=(2**63-1, {}), key=lambda pair: pair[0])
    bridge = CvBridge()
    seed_mask = bridge.imgmsg_to_cv2(seed.mask, 'mono8')
    if (stamp(seed.mask) != first or not seed_mask.any()
            or seed_mask.shape != (expected['height'], expected['width'])
            or seed.mask.header.frame_id != expected['camera_optical_frame']):
        raise ValueError('Invalid accepted-seed mask')
    output.mkdir(parents=True)
    atomic_json(output / 'extraction.json', dict(state='inprogress', source_bag=str(bag)))
    records = []
    connections = {}

    def read(ref):
        path, row, kind = ref
        if path not in connections:
            connections[path] = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        data = connections[path].execute('SELECT data FROM messages WHERE id=?', (row,)).fetchone()[0]
        return deserialize_message(data, kind)

    # Reserve accepted pins; each has a synchronized raw tuple even if the image topic dropped it.
    pinned_stamps = set(accepted) & set(pins)
    color_stamps = sorted(s for s in images[RGB] if first < s <= end_ns and s not in pinned_stamps)
    reserved_depth = {stamp(pins[s].depth) for s in accepted if s in pins}
    depth_stamps = [s for s in images[DEPTH] if s not in reserved_depth]
    pairs = match_stamps(color_stamps, depth_stamps, round(config['synchronization_slop_sec']*1e9))
    rejects['rgb_depth_unsynchronised'] += len(color_stamps) - len(pairs)
    info_stamps = sorted(infos)
    candidates = sorted(set(pairs) | {s for s in accepted if s in pins and first <= s <= end_ns})
    try:
        for s in candidates:
            if s in accepted and s in pins:
                pin = pins[s]
                rgb, depth, info = pin.rgb, pin.depth, calibration(pin.camera_info)
            else:
                rgb, depth = read(images[RGB][s]), read(images[DEPTH][pairs[s]])
                i = bisect_right(info_stamps, s) - 1
                info = infos[info_stamps[i]] if i >= 0 else None
            if info != expected:
                raise ValueError('Camera intrinsics/resolution changed during sequence')
            if any(m.header.frame_id != expected['camera_optical_frame'] for m in (rgb, depth)):
                raise ValueError('RGB/depth optical-frame mismatch')
            if abs(s-stamp(depth)) > round(config['synchronization_slop_sec']*1e9):
                raise ValueError('Accepted pin is not synchronized')
            color = bridge.imgmsg_to_cv2(rgb, 'bgr8')
            z = bridge.imgmsg_to_cv2(depth, 'passthrough')
            if color.shape != (expected['height'], expected['width'], 3) or z.shape != color.shape[:2]:
                raise ValueError('RGB/depth resolution mismatch')
            if (depth.encoding, str(z.dtype)) not in (('16UC1', 'uint16'), ('32FC1', 'float32')):
                raise ValueError('Invalid depth encoding')
            transform, tf_error = None, None
            try:
                tf = buffer.lookup_transform(
                    'base_link', expected['camera_optical_frame'], Time(nanoseconds=s))
                q, t = tf.transform.rotation, tf.transform.translation
                quaternion = np.array([q.x, q.y, q.z, q.w])
                if (stamp(tf) != s or not np.isfinite(quaternion).all()
                        or abs(np.linalg.norm(quaternion)-1) > 1e-5):
                    raise ValueError('stale or invalid TF')
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
                matrix[:3, 3] = [t.x, t.y, t.z]
                transform = validate_transform(matrix).tolist()
            except Exception:
                tf_error = 'missing_or_invalid_tf'
            name = '%08d' % len(records)
            folder = output / name
            folder.mkdir()
            if not cv2.imwrite(str(folder / 'rgb.png'), color):
                raise OSError('RGB encoding failed')
            np.save(str(folder / 'depth.npy'), z, allow_pickle=False)
            if s == first and not cv2.imwrite(str(folder / 'seed.png'), seed_mask):
                raise OSError('Seed mask encoding failed')
            record = dict(index=len(records), rgb=name+'/rgb.png', depth=name+'/depth.npy',
                          rgb_stamp_ns=s, depth_stamp_ns=stamp(depth), tf_stamp_ns=s,
                          T_base_camera=transform, tf_error=tf_error, base_frame='base_link',
                          depth_encoding=depth.encoding, depth_dtype=str(z.dtype),
                          depth_scale_m=1. if depth.encoding == '32FC1' else config['depth_scale_m'],
                          accepted_view=accepted.get(s, 0), **expected)
            records.append(record)
        if not records or records[0]['rgb_stamp_ns'] != first:
            raise ValueError('Initial accepted frame not recovered')
        report = dict(schema_version=1, dataset_type='piper_unsegmented_rgbd_video', state='ready',
                      source_bag=str(bag), task_id=archive['task_id'], session_id=history['session_id'],
                      complete=terminal.get('state') == 'COMPLETE',
                      mission_outcome=terminal.get('state', 'UNKNOWN'),
                      mission_capture_count=int(history['accepted_views']), configuration=config,
                      coverage_target_center=target, frames=records, seed_mask='00000000/seed.png',
                      source_message_counts=dict(counts), rejected_by_reason=dict(rejects),
                      selection='Every synchronized RGB-D after accepted seed; no MOVING or rate gate')
        atomic_json(output / 'extraction.json', report)
        print('Extracted %d synchronized frames' % len(records), flush=True)
        return report
    except BaseException as error:
        atomic_json(output / 'extraction.json', dict(
            state='failed', error=str(error), source_bag=str(bag), frames=records))
        raise
    finally:
        for connection in connections.values():
            connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    extract(args.bag, args.output)
