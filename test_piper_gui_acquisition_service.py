"""Command-free GUI ROS ownership regression for Phase 8."""

import queue

import rclpy

from piper_gui_native import PiperGuiRos
from piper_gui.ros_node import PiperGuiRos as PiperGuiRosOwner
from piper_gui.view_model import validate_mission_request


def test_gui_exposes_action_client_without_legacy_acquisition_services():
    assert PiperGuiRos is PiperGuiRosOwner
    events = queue.Queue()
    rclpy.init()
    gui = PiperGuiRos(events)
    try:
        assert gui.mission_action_client is not None
        assert gui.mission_client is not None
        assert not hasattr(gui, 'acquisition_prepare_client')
        assert not hasattr(gui, 'multiview_plan_client')
        assert not hasattr(gui, 'workflow_start_client')
        assert not hasattr(gui, 'scan_approve_client')
        assert not hasattr(gui, 'scan_cancel_client')

        goal = gui._build_mission_goal(
            'gui-sim-contract',
            validate_mission_request((0.4, -0.1, 0.2), 'green cube'),
        )
        assert goal.task_id == 'gui-sim-contract'
        assert goal.task_type == 'SCAN_3D'
        assert goal.target_label == 'green cube'
        assert goal.target_profile == ''
        assert goal.target_confidence == 1.0
        assert goal.deadline_sec == 1200.0
        assert goal.rough_target.header.frame_id == 'base_link'
        point = goal.rough_target.pose.pose.position
        assert (point.x, point.y, point.z) == (0.4, -0.1, 0.2)
        covariance = goal.rough_target.pose.covariance
        assert covariance[0] == covariance[7] == covariance[14] == 0.01
    finally:
        gui.destroy_node()
        rclpy.shutdown()
