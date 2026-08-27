"""Real DDS regression for the GUI stack's controller-limit message."""

import os
import multiprocessing
from pathlib import Path
import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from piper_msgs.msg import PiperMotionLimits


def motion_limits_message():
    message = PiperMotionLimits()
    message.header.frame_id = 'transport_test_invalid'
    message.joint_names = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    message.max_velocity_rad_s = [0.0] * 6
    message.max_acceleration_rad_s2 = [0.0] * 6
    message.valid = False
    message.limits_sha256 = '0' * 64
    message.source = 'transport_test'
    message.reason = 'invalid transport probe'
    return message


def cross_process_publisher(stop_event):
    rclpy.init()
    node = Node('motion_limits_cross_process_publisher')
    publisher = node.create_publisher(
        PiperMotionLimits, '/piper/motion_limits_cross_process_test', 10)
    message = motion_limits_message()
    try:
        while not stop_event.is_set():
            message.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(message)
            time.sleep(0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cross_process_subscriber(result_queue):
    rclpy.init()
    node = Node('motion_limits_cross_process_subscriber')
    received = []
    node.create_subscription(
        PiperMotionLimits,
        '/piper/motion_limits_cross_process_test',
        received.append,
        10,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        result_queue.put(bool(received))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def configure_transport(monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        'FASTRTPS_DEFAULT_PROFILES_FILE',
        str(repository / 'fastdds_gui_udp_only.xml'),
    )
    monkeypatch.setenv('RMW_FASTRTPS_USE_QOS_FROM_XML', '0')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '0')


def test_motion_limits_reach_an_independent_ros_node(monkeypatch):
    configure_transport(monkeypatch)

    rclpy.init()
    publisher_node = Node('motion_limits_transport_publisher')
    subscriber_node = Node('motion_limits_transport_subscriber')
    publisher = publisher_node.create_publisher(
        PiperMotionLimits, '/piper/motion_limits_transport_test', 10)
    received = []
    subscriber_node.create_subscription(
        PiperMotionLimits,
        '/piper/motion_limits_transport_test',
        received.append,
        10,
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(publisher_node)
    executor.add_node(subscriber_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        message = motion_limits_message()

        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            message.header.stamp = publisher_node.get_clock().now().to_msg()
            publisher.publish(message)
            time.sleep(0.05)

        assert received
        assert not received[-1].valid
        assert received[-1].joint_names == message.joint_names
        assert received[-1].limits_sha256 == '0' * 64
    finally:
        executor.shutdown()
        publisher_node.destroy_node()
        subscriber_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def test_motion_limits_cross_process_transport(monkeypatch):
    configure_transport(monkeypatch)
    context = multiprocessing.get_context('spawn')
    result_queue = context.Queue()
    stop_event = context.Event()
    subscriber = context.Process(
        target=cross_process_subscriber, args=(result_queue,))
    publisher = context.Process(
        target=cross_process_publisher, args=(stop_event,))
    subscriber.start()
    time.sleep(0.5)
    publisher.start()
    try:
        subscriber.join(timeout=8.0)
        assert subscriber.exitcode == 0
        assert result_queue.get(timeout=1.0)
    finally:
        stop_event.set()
        publisher.join(timeout=3.0)
        if publisher.is_alive():
            publisher.terminate()
            publisher.join(timeout=2.0)


@pytest.mark.skipif(
    os.environ.get('PIPER_LIVE_DRIVER_TEST') != '1',
    reason='requires the separately started, disabled PiPER driver',
)
def test_live_driver_publishes_valid_motion_limits(monkeypatch):
    configure_transport(monkeypatch)
    rclpy.init()
    node = Node('motion_limits_live_driver_subscriber')
    received = []
    node.create_subscription(
        PiperMotionLimits, '/piper/motion_limits', received.append, 10)
    try:
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert received
        assert received[-1].valid, received[-1].reason
        assert len(received[-1].max_velocity_rad_s) == 6
        assert len(received[-1].max_acceleration_rad_s2) == 6
        assert len(received[-1].limits_sha256) == 64
    finally:
        node.destroy_node()
        rclpy.shutdown()
