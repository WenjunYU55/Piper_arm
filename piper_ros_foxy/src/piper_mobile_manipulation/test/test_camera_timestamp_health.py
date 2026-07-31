from pathlib import Path
import shutil
from types import SimpleNamespace
import os
import time

import pytest
import rclpy
import yaml
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, JointState

from piper_mobile_manipulation.camera_timestamp_health import TimestampHealthMonitor
from piper_mobile_manipulation.camera_timestamp_watchdog_node import (
    CameraTimestampWatchdogNode,
)
from piper_mobile_manipulation.sam2_live_bridge_node import Sam2LiveBridgeNode


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]


def test_timestamp_monitor_requires_consecutive_healthy_frames():
    monitor = TimestampHealthMonitor(max_offset_sec=0.5, healthy_frames_required=3)
    first = monitor.evaluate(100.0, 100.1)
    second = monitor.evaluate(100.1, 100.2)
    third = monitor.evaluate(100.2, 100.3)

    assert first.state == 'STARTING' and not first.healthy
    assert second.consecutive_healthy_frames == 2 and not second.healthy
    assert third.state == 'HEALTHY' and third.healthy


def test_large_clock_offset_fails_closed_and_resets_stability():
    monitor = TimestampHealthMonitor(max_offset_sec=0.5, healthy_frames_required=2)
    assert not monitor.evaluate(10.0, 10.1).healthy
    assert monitor.evaluate(10.1, 10.2).healthy

    stale = monitor.evaluate(10.2, 4307.2)

    assert stale.state == 'CLOCK_OFFSET'
    assert stale.offset_sec == pytest.approx(4297.0)
    assert not stale.healthy
    assert stale.consecutive_healthy_frames == 0
    assert not monitor.evaluate(4307.3, 4307.4).healthy


def test_backwards_camera_timestamp_is_rejected():
    monitor = TimestampHealthMonitor(backward_tolerance_sec=0.001)
    monitor.evaluate(20.0, 20.0)

    result = monitor.evaluate(19.5, 20.1)

    assert result.state == 'NON_MONOTONIC'
    assert not result.monotonic
    assert not result.healthy


def test_sam_bridge_rejects_missing_stale_and_unhealthy_clock():
    parameters = {'camera_timestamp_health_timeout_sec': 1.0}
    bridge = SimpleNamespace(
        latest_camera_timestamp_health=None,
        camera_timestamp_health_at=0.0,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    assert 'missing' in Sam2LiveBridgeNode.camera_timestamp_rejection_reason(bridge, 10.0)

    bridge.latest_camera_timestamp_health = SimpleNamespace(
        healthy=True, state='HEALTHY', reason='ok')
    bridge.camera_timestamp_health_at = 8.0
    assert 'stale' in Sam2LiveBridgeNode.camera_timestamp_rejection_reason(bridge, 10.0)

    bridge.latest_camera_timestamp_health = SimpleNamespace(
        healthy=False, state='CLOCK_OFFSET', reason='camera timestamp is stale')
    bridge.camera_timestamp_health_at = 10.0
    assert 'CLOCK_OFFSET' in Sam2LiveBridgeNode.camera_timestamp_rejection_reason(bridge, 10.0)


def test_motion_executor_and_pipeline_have_independent_clock_gates():
    executor = (
        PACKAGE_ROOT / 'piper_mobile_manipulation' /
        'scan_viewpoint_executor_node.py').read_text()
    pipeline = (REPO_ROOT / 'L515_camera' / 'run_gpu_vision_pipeline.sh').read_text()

    assert "self.mark('camera_clock')" in executor
    assert "not camera_health.healthy" in executor
    assert 'Approval repeats the fresh camera-clock gate' in executor
    assert 'Preserve the' in executor
    assert 'immutable proposal while that transient blocks approval' in executor
    assert 'Camera timestamp fault confirmed while arm is stationary.' in pipeline
    assert 'recovery_backoff' in pipeline
    assert 'stop_processes' in pipeline


def test_recovery_request_waits_for_stationary_arm():
    root = Path('/tmp/piper_vision_recovery/pytest_%d' % os.getpid())
    request = root / 'request.yaml'
    shutil.rmtree(str(root), ignore_errors=True)
    old_log_dir = os.environ.get('ROS_LOG_DIR')
    os.environ['ROS_LOG_DIR'] = str(root / 'ros_log')
    rclpy.init()
    node = CameraTimestampWatchdogNode()
    try:
        node.set_parameters([
            Parameter(
                'recovery_request_path', Parameter.Type.STRING, str(request)),
            Parameter('startup_grace_sec', Parameter.Type.DOUBLE, 0.0),
            Parameter('stationary_duration_sec', Parameter.Type.DOUBLE, 0.0),
            Parameter(
                'unhealthy_frames_before_recovery', Parameter.Type.INTEGER, 2),
        ])
        node.started_at = time.monotonic() - 10.0
        stale = CameraInfo()
        stale.header.stamp.sec = 1
        moving = JointState()
        moving.velocity = [0.2] * 6
        node.joint_cb(moving)
        node.image_cb(stale)
        node.image_cb(stale)
        node.tick()
        assert not request.exists()

        stationary = JointState()
        stationary.velocity = [0.0] * 6
        node.joint_cb(stationary)
        node.stationary_since = time.monotonic() - 1.0
        node.tick()
        assert request.exists()
        assert 'arm_stationary: true' in request.read_text()
        assert node.recovery_latched
    finally:
        node.destroy_node()
        rclpy.shutdown()
        shutil.rmtree(str(root), ignore_errors=True)
        if old_log_dir is None:
            os.environ.pop('ROS_LOG_DIR', None)
        else:
            os.environ['ROS_LOG_DIR'] = old_log_dir


def test_stationary_detection_uses_positions_despite_noisy_velocity():
    node = SimpleNamespace(
        last_joint_at=None,
        stationary_anchor_positions=None,
        stationary_since=None,
        arm_stationary=False,
        get_parameter=lambda name: SimpleNamespace(value={
            'stationary_position_tolerance_rad': 0.001,
            'stationary_velocity_rad_s': 0.03,
            'joint_state_timeout_sec': 1.0,
            'stationary_duration_sec': 0.75,
        }[name]),
    )
    node.update_arm_stationary = (
        lambda now: CameraTimestampWatchdogNode.update_arm_stationary(node, now))
    feedback = JointState()
    feedback.position = [0.1] * 6
    feedback.velocity = [0.09] * 6

    CameraTimestampWatchdogNode.joint_cb(node, feedback)
    node.stationary_since = time.monotonic() - 1.0
    CameraTimestampWatchdogNode.joint_cb(node, feedback)

    assert node.arm_stationary

    feedback.position[3] += 0.01
    CameraTimestampWatchdogNode.joint_cb(node, feedback)
    assert not node.arm_stationary


def test_watchdog_uses_small_per_frame_camera_info_clock_sample():
    source = (
        PACKAGE_ROOT / 'piper_mobile_manipulation' /
        'camera_timestamp_watchdog_node.py').read_text()
    config = yaml.safe_load((
        PACKAGE_ROOT / 'config' /
        'camera_timestamp_watchdog_params.yaml').read_text())

    assert 'CameraInfo, self.get_parameter' in source
    assert 'MultiThreadedExecutor' not in source
    assert 'rclpy.spin(node)' in source
    assert config['/**']['ros__parameters']['image_topic'] == (
        '/camera/color/camera_info')
