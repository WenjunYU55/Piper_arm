"""Record movement RGB-D with pre/post-roll; never publish or command anything."""

import argparse
import json
from pathlib import Path
import time

from piper_mobile_manipulation.movement_recording import MovementWindow, PreRoll, SQLiteBagWriter


SENSOR_TOPICS = {
    '/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
    '/piper/sam2_target_mask',
}
STATUS = '/piper/scan_execution_status'


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from piper_mobile_manipulation.mission_archive import TOPICS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-cache-size', default='0')
    args = parser.parse_args()
    rclpy.init()
    node = Node('movement_bag_recorder')
    writer = SQLiteBagWriter(args.output)
    window, pending = MovementWindow(), PreRoll()
    subscriptions = {}
    counters = {'saved_sensor_messages': 0, 'buffered_idle_messages': 0}

    def save(item):
        topic, data, stamp = item
        writer.write(topic, data, stamp)
        if topic in SENSOR_TOPICS:
            counters['saved_sensor_messages'] += 1

    def callback(topic, kind, data):
        now = time.monotonic()
        stamp = node.get_clock().now().nanoseconds
        item = (topic, data, stamp)
        if topic == STATUS:
            message = deserialize_message(data, kind)
            if window.update(message.state, now):
                for buffered in pending.take(now):
                    save(buffered)
        if topic not in SENSOR_TOPICS or window.active(now):
            save(item)
        else:
            counters['buffered_idle_messages'] += 1
            pending.add(now, item)

    def discover():
        for topic, types in node.get_topic_names_and_types():
            if topic not in TOPICS or topic in subscriptions or len(types) != 1:
                continue
            kind = get_message(types[0])
            static = topic == '/tf_static'
            qos = QoSProfile(depth=100 if static else 8,
                             reliability=ReliabilityPolicy.RELIABLE if static
                             else ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL if static
                             else DurabilityPolicy.VOLATILE)
            writer.create_topic(topic, types[0])
            subscriptions[topic] = node.create_subscription(
                kind, topic, lambda data, t=topic, k=kind: callback(t, k, data),
                qos, raw=True)

    node.create_timer(.25, discover)
    # Stop stale pre-roll retaining memory while the camera is silent.
    node.create_timer(.25, lambda: pending.expire(time.monotonic()))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        writer.close()  # Close SQLite before parent verifies bag metadata.
        counters['pre_roll_overflow_messages'] = pending.overflow
        counters['selection'] = 'movement plus one second before and after; capture pins retained'
        (Path(args.output).parent / 'movement_recording.json').write_text(
            json.dumps(counters, indent=2))
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
