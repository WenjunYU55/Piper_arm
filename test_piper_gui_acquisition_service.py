"""Command-free GUI-to-acquisition service transport regression."""

import os
from pathlib import Path
import queue
import threading
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from piper_gui_native import PiperGuiRos
from piper_mobile_manipulation.scan_target_acquisition_node import (
    ScanTargetAcquisitionNode,
)


def test_real_prepare_service_round_trip_enables_async_plan_wait(monkeypatch):
    """The real service must acknowledge the GUI request before its deadline."""
    repository = Path(__file__).resolve().parent
    monkeypatch.setenv(
        'FASTRTPS_DEFAULT_PROFILES_FILE',
        str(repository / 'fastdds_gui_udp_only.xml'),
    )
    monkeypatch.setenv('RMW_FASTRTPS_USE_QOS_FROM_XML', '0')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '0')
    assert os.environ['RMW_FASTRTPS_USE_QOS_FROM_XML'] == '0'

    events = queue.Queue()
    rclpy.init()
    gui = PiperGuiRos(events)
    acquisition = ScanTargetAcquisitionNode()
    tf_node = Node('piper_gui_prepare_service_test_tf')
    broadcaster = StaticTransformBroadcaster(tf_node)
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (gui, acquisition, tf_node):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        transform = TransformStamped()
        transform.header.stamp = tf_node.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'camera_color_optical_frame'
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.45
        transform.transform.rotation.w = 1.0
        broadcaster.sendTransform(transform)

        # Allow the static transform and service endpoint to traverse the same
        # loopback-DDS discovery path used by the managed GUI stack.
        deadline = time.monotonic() + 5.0
        while (
                not gui.acquisition_prepare_client.service_is_ready()
                and time.monotonic() < deadline):
            time.sleep(0.05)
        assert gui.acquisition_prepare_client.service_is_ready()

        gui.publish_rough_target_and_start(
            (0.40, 0.0, 0.20),
            'acq-service-roundtrip-0001',
            stack_generation=7,
            attempt_generation=11,
        )

        event_deadline = time.monotonic() + 10.0
        result = None
        while time.monotonic() < event_deadline:
            try:
                name, payload = events.get(timeout=0.25)
            except queue.Empty:
                continue
            if name == 'acquisition_start':
                result = payload
                break
        assert result is not None
        stack_generation, attempt_generation, outcome, message = result
        assert stack_generation == 7
        assert attempt_generation == 11
        assert outcome == 'accepted', message
        assert 'asynchronous' in message
    finally:
        executor.shutdown()
        for node in (gui, acquisition, tf_node):
            node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
