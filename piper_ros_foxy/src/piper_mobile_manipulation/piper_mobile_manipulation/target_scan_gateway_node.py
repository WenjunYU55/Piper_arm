#!/usr/bin/env python3
"""Network-facing action/TF gateway for the loopback-isolated PiPER stack."""

import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.mission_core import validate_goal_payload
from piper_mobile_manipulation.mission_spool import MissionSpool
from piper_mobile_manipulation.scan_capture import rigid_transform_matrix
from piper_mobile_manipulation.srv import GetTargetScanResult


class TargetScanGatewayNode(Node):
    def __init__(self):
        super().__init__('target_scan_gateway')
        self.declare_parameter(
            'mission_spool_root', os.path.join(
                os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                'piper_target_scan_missions'))
        self.declare_parameter('piper_base_frame', 'piper_base_link')
        self.declare_parameter('local_base_frame', 'base_link')
        self.spool = MissionSpool(
            self.get_parameter('mission_spool_root').value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.callback_group = ReentrantCallbackGroup()
        self.active_task_id = ''
        self.active_mission_sha256 = ''
        self.lock = threading.RLock()
        self.server = ActionServer(
            self, RunTargetScan, '/piper/run_target_scan',
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.callback_group)
        self.create_service(
            GetTargetScanResult, '/piper/get_target_scan_result',
            self.get_result_cb, callback_group=self.callback_group)

    @staticmethod
    def goal_payload(goal):
        stamp = goal.rough_target.header.stamp
        return {
            'task_id': str(goal.task_id),
            'task_type': str(goal.task_type),
            'target_label': str(goal.target_label),
            'target_profile': str(goal.target_profile),
            'target_confidence': float(goal.target_confidence),
            'deadline_sec': float(goal.deadline_sec),
            'rough_target': {
                'frame_id': str(goal.rough_target.header.frame_id),
                'stamp_sec': float(stamp.sec) + float(stamp.nanosec) * 1e-9,
                'position': [
                    float(goal.rough_target.pose.pose.position.x),
                    float(goal.rough_target.pose.pose.position.y),
                    float(goal.rough_target.pose.pose.position.z),
                ],
                'covariance': list(goal.rough_target.pose.covariance),
            },
        }

    def goal_cb(self, goal):
        try:
            normalized = validate_goal_payload(self.goal_payload(goal))
        except (TypeError, ValueError) as exc:
            self.get_logger().warn('gateway rejected target-scan goal: %s' % exc)
            return GoalResponse.REJECT
        task_id = normalized['task_id']
        try:
            cached = self.spool.read('results', task_id)
        except (FileNotFoundError, OSError, ValueError):
            cached = None
        if cached is not None:
            return (
                GoalResponse.ACCEPT
                if self.persisted_gateway_hash(task_id)
                == normalized['mission_sha256'] else GoalResponse.REJECT)
        with self.lock:
            if self.active_task_id:
                if (self.active_task_id != task_id
                        or self.active_mission_sha256
                        != normalized['mission_sha256']):
                    return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def persisted_gateway_hash(self, task_id):
        try:
            goal = self.spool.read('goals', task_id)
        except (FileNotFoundError, OSError, ValueError):
            return ''
        return str(goal.get('gateway_mission_sha256', ''))

    def cancel_cb(self, _goal_handle):
        return CancelResponse.ACCEPT

    def get_result_cb(self, request, response):
        try:
            payload = self.spool.read('results', request.task_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            response.found = False
            response.result_json = ''
            response.message = 'task result not found: %s' % exc
            return response
        response.found = True
        response.result_json = json.dumps(payload, sort_keys=True)
        response.message = 'durable result returned'
        return response

    def execute_cb(self, goal_handle):
        goal = goal_handle.request
        try:
            normalized = validate_goal_payload(self.goal_payload(goal))
        except (TypeError, ValueError) as exc:
            goal_handle.abort()
            return self.result_message({'outcome': 'FAILED', 'reason': str(exc)})
        task_id = normalized['task_id']
        try:
            cached = self.spool.read('results', task_id)
        except (FileNotFoundError, OSError, ValueError):
            cached = None
        if cached is not None:
            if (self.persisted_gateway_hash(task_id)
                    != normalized['mission_sha256']):
                goal_handle.abort()
                return self.result_message({
                    'outcome': 'FAILED',
                    'reason': 'task ID conflicts with a different durable goal',
                })
            goal_handle.succeed() if cached.get('outcome') == 'SUCCEEDED' \
                else goal_handle.abort()
            return self.result_message(cached)
        with self.lock:
            if self.active_task_id and self.active_task_id != task_id:
                goal_handle.abort()
                return self.result_message({
                    'outcome': 'BUSY',
                    'reason': 'another target-scan task is active',
                })
            self.active_task_id = task_id
            self.active_mission_sha256 = normalized['mission_sha256']
        try:
            local_goal = self.transform_goal(normalized, goal)
            self.spool.write('goals', task_id, local_goal)
            deadline = time.monotonic() + float(normalized['deadline_sec']) + 90.0
            while time.monotonic() < deadline:
                cancelled = bool(goal_handle.is_cancel_requested)
                self.spool.write('heartbeat', task_id, {
                    'task_id': task_id,
                    'mission_sha256': local_goal['mission_sha256'],
                    'wall_time_sec': time.time(),
                    'cancel_requested': cancelled,
                })
                try:
                    result = self.spool.read('results', task_id)
                except (FileNotFoundError, OSError, ValueError):
                    result = None
                if result is not None:
                    if result.get('outcome') == 'SUCCEEDED':
                        goal_handle.succeed()
                    else:
                        goal_handle.abort()
                    return self.result_message(result)
                self.publish_status_feedback(goal_handle, task_id)
                time.sleep(0.5)
            goal_handle.abort()
            return self.result_message({
                'outcome': 'FAILED',
                'reason': 'gateway did not receive a safe-shutdown result before timeout',
            })
        finally:
            with self.lock:
                if self.active_task_id == task_id:
                    self.active_task_id = ''
                    self.active_mission_sha256 = ''

    def transform_goal(self, normalized, goal):
        source = normalized['rough_target']['frame_id']
        if source == 'base_link':
            raise ValueError(
                'network gateway requires odom input; base_link is local-only')
        piper_base = str(self.get_parameter('piper_base_frame').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                piper_base, source,
                Time.from_msg(goal.rough_target.header.stamp))
        except TransformException as exc:
            raise ValueError(
                'cannot snapshot %s to %s: %s' % (source, piper_base, exc))
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = rigid_transform_matrix(
            [translation.x, translation.y, translation.z],
            [rotation.x, rotation.y, rotation.z, rotation.w])
        point = np.asarray(normalized['rough_target']['position'] + [1.0])
        transformed = (matrix @ point)[:3].tolist()
        covariance = np.asarray(
            normalized['rough_target']['covariance'], dtype=float).reshape(6, 6)
        rotation_3x3 = matrix[:3, :3]
        pose_rotation = np.zeros((6, 6), dtype=float)
        pose_rotation[:3, :3] = rotation_3x3
        pose_rotation[3:, 3:] = rotation_3x3
        transformed_covariance = (
            pose_rotation @ covariance @ pose_rotation.T).reshape(-1).tolist()
        local = dict(normalized)
        local.pop('mission_sha256', None)
        local['rough_target'] = dict(normalized['rough_target'])
        local['rough_target']['frame_id'] = str(
            self.get_parameter('local_base_frame').value)
        local['rough_target']['position'] = transformed
        local['rough_target']['covariance'] = transformed_covariance
        local['gateway_mission_sha256'] = normalized['mission_sha256']
        local['source_transform'] = {
            'target_frame': piper_base,
            'source_frame': source,
            'stamp_sec': normalized['rough_target']['stamp_sec'],
            'matrix_4x4': matrix.tolist(),
        }
        # Mission identity is recomputed over the local immutable task. The
        # source transform remains independently hashed by the spool envelope.
        from piper_mobile_manipulation.mission_core import sha256_value
        identity = {
            key: local[key] for key in (
                'task_id', 'task_type', 'target_label', 'target_profile',
                'target_confidence', 'deadline_sec', 'rough_target')
        }
        local['mission_sha256'] = sha256_value(identity)
        return local

    def publish_status_feedback(self, goal_handle, task_id):
        try:
            status = self.spool.read('status', task_id)
        except (FileNotFoundError, OSError, ValueError):
            return
        feedback = RunTargetScan.Feedback()
        feedback.phase = str(status.get('phase', 'GOAL_LATCHED'))
        feedback.reason = str(status.get('reason', 'waiting for local orchestrator'))
        feedback.elapsed_sec = float(status.get('elapsed_sec', 0.0))
        feedback.remaining_sec = float(status.get('remaining_sec', 0.0))
        feedback.accepted_captures = int(status.get('accepted_captures', 0))
        feedback.required_captures = int(status.get('required_captures', 13))
        feedback.occlusion_action_limit = 6
        feedback.process_health_json = json.dumps(
            status.get('processes', {}), sort_keys=True)
        feedback.shutdown_phase = feedback.phase if feedback.phase in (
            'RETURNING_HOME', 'HOLDING', 'DISABLING', 'STOPPING') else ''
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def result_message(payload):
        result = RunTargetScan.Result()
        outcomes = {
            # Constants declared before the first separator in a ROS action
            # belong to the generated Goal class, not the Result instance.
            'SUCCEEDED': RunTargetScan.Goal.OUTCOME_SUCCEEDED,
            'FAILED': RunTargetScan.Goal.OUTCOME_FAILED,
            'CANCELLED': RunTargetScan.Goal.OUTCOME_CANCELLED,
            'BUSY': RunTargetScan.Goal.OUTCOME_BUSY,
            'UNSUPPORTED_TARGET_PROFILE': (
                RunTargetScan.Goal.OUTCOME_UNSUPPORTED_TARGET_PROFILE),
            'NEEDS_OPERATOR': RunTargetScan.Goal.OUTCOME_NEEDS_OPERATOR,
        }
        result.outcome = outcomes.get(
            str(payload.get('outcome', 'FAILED')),
            RunTargetScan.Goal.OUTCOME_FAILED)
        result.reason = str(payload.get('reason', ''))
        result.safe_shutdown = bool(payload.get('safe_shutdown', False))
        result.dataset_path = str(payload.get('dataset_path', ''))
        result.manifest_sha256 = str(payload.get('manifest_sha256', ''))
        result.capture_count = int(payload.get('capture_count', 0))
        result.mesh_job_id = str(payload.get('mesh_job_id', ''))
        result.action_summary_json = json.dumps(
            payload.get('action_summary', {}), sort_keys=True)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = TargetScanGatewayNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.server.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
