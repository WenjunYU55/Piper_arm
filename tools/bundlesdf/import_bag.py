#!/usr/bin/env python3
"""Recover a recorded ROS bag through the recorder callbacks without DDS transport loss."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import time

import rclpy
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml

from piper_mobile_manipulation.reconstruction_recorder_node import ReconstructionRecorderNode


def drain(node):
    """Bound recovery backpressure outside any robot or ROS subscription callback."""
    deadline = time.monotonic() + 15
    while node.images.qsize() > 2 or node.events.qsize() > 0:
        if not node.worker.is_alive() or time.monotonic() > deadline:
            raise RuntimeError('Recorder admission stopped or failed to drain')
        time.sleep(0.001)


def recover(bag, output):
    """Feed original stamped messages through real synchronization and session admission."""
    if os.environ.get('ROS_DOMAIN_ID', '0') in ('0', '42', ''):
        raise ValueError('Set ROS_DOMAIN_ID to an unused isolated offline domain')
    bag, output = Path(bag).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing existing recovery output')
    metadata = yaml.safe_load((bag / 'metadata.yaml').read_text())
    archive = json.loads((bag.parent / 'recording.json').read_text())
    parameters = []
    for name in ('scan_quality_params.yaml', 'reconstruction_recording_params.yaml'):
        path = bag.parent / 'config' / name
        if path.is_file():
            values = yaml.safe_load(path.read_text())['/**']['ros__parameters']
            parameters.extend(Parameter(k, value=v) for k, v in values.items())
    parameters.extend([
        Parameter('output_root', value=str(output)),
        Parameter('task_id', value=archive['task_id']),
        Parameter('target_visibility_angle_tolerance_deg',
                  value=float(archive['target_visibility_angle_tolerance_deg'])),
    ])
    output.mkdir(parents=True)
    counts = Counter()
    rclpy.init()
    node = ReconstructionRecorderNode(parameter_overrides=parameters)
    callbacks = {
        '/camera/color/image_raw': node.rgb_sub.signalMessage,
        '/camera/aligned_depth_to_color/image_raw': node.depth_sub.signalMessage,
        '/camera/color/camera_info': node.info_cb,
        '/piper/sam2_target_mask': node.mask_cb,
        '/piper/reconstruction_keyframe': node.keyframe_cb,
        '/piper/scan_session_history': node.history_cb,
        '/piper/supervised_workflow_status': node.workflow_cb,
        '/piper/scan_execution_status': node.execution_cb,
        '/tf': node.tf_listener.callback,
        '/tf_static': node.tf_listener.static_callback,
    }
    try:
        for relative in metadata['rosbag2_bagfile_information']['relative_file_paths']:
            path = (bag / relative).resolve()
            if path.parent != bag:
                raise ValueError('Bag database escapes input directory')
            with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as connection:
                topics = {i: (name, get_message(kind)) for i, name, kind in
                          connection.execute('SELECT id,name,type FROM topics')
                          if name in callbacks}
                for topic_id, data in connection.execute(
                        'SELECT topic_id,data FROM messages ORDER BY timestamp,id'):
                    if topic_id not in topics:
                        continue
                    topic, kind = topics[topic_id]
                    drain(node)
                    callbacks[topic](deserialize_message(data, kind))
                    counts[topic] += 1
            print('Read ' + relative, flush=True)
        drain(node)
        # Resolve final pending pin/history events and finish through node shutdown.
        time.sleep(0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        report = dict(
            evidence_class='RECORDED_ROS_BAG_CALLBACK_RECOVERY_NOT_DDS_OR_NEW_PHYSICAL_TEST',
            source_bag=str(bag), task_id=archive['task_id'], messages=dict(counts),
            sessions=[str(s.final_path or s.path) for s in node.completed_sessions],
            note=(
                'Original image stamps and TF; raw callbacks, synchronizer and gates retained. '
                'Offline input waits for admission capacity; '
                'no robot processes or motion publishers.'),
        )
        (output / 'recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    if not sum(s.accepted for s in node.completed_sessions):
        raise RuntimeError('No accepted frames; inspect recovered manifests')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    recover(args.bag, args.output)
