#!/usr/bin/env python3
"""Network-facing action/TF gateway for the loopback-isolated PiPER stack."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.mission_core import (
    MAX_PENDING_MISSIONS,
    validate_goal_payload,
)
from piper_mobile_manipulation.mission_spool import MissionSpool
from piper_mobile_manipulation.msg import MeshJobStatus
from piper_mobile_manipulation.reconstruction_jobs import (
    transition_job,
    validate_home_report,
    waiting_job,
)
from piper_mobile_manipulation.scan_capture import rigid_transform_matrix
from piper_mobile_manipulation.srv import (
    GetMeshJobResult,
    GetTargetScanResult,
    ReportTrackedRobotHomed,
)


class TargetScanGatewayNode(Node):
    def __init__(self):
        super().__init__('target_scan_gateway')
        self.declare_parameter(
            'mission_spool_root', os.path.join(
                os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                'piper_target_scan_missions'))
        self.declare_parameter('piper_base_frame', 'piper_base_link')
        self.declare_parameter('local_base_frame', 'base_link')
        self.declare_parameter('project_root', '/home/prl/Piper_arm')
        self.declare_parameter('reconstruction_python', '')
        self.declare_parameter('reconstruction_timeout_sec', 1800.0)
        self.declare_parameter('max_pending_missions', MAX_PENDING_MISSIONS)
        self.spool = MissionSpool(
            self.get_parameter('mission_spool_root').value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.callback_group = ReentrantCallbackGroup()
        self.admitted_tasks = {}
        self.lock = threading.RLock()
        self.active_mesh_jobs = set()
        self.server = ActionServer(
            self, RunTargetScan, '/piper/run_target_scan',
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.callback_group)
        self.create_service(
            GetTargetScanResult, '/piper/get_target_scan_result',
            self.get_result_cb, callback_group=self.callback_group)
        self.create_service(
            ReportTrackedRobotHomed, '/piper/report_tracked_robot_homed',
            self.report_tracked_robot_homed_cb,
            callback_group=self.callback_group)
        self.create_service(
            GetMeshJobResult, '/piper/get_mesh_job_result',
            self.get_mesh_job_result_cb,
            callback_group=self.callback_group)
        self.mesh_status_pub = self.create_publisher(
            MeshJobStatus, '/piper/mesh_job_status',
            QoSProfile(
                depth=10, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))

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
            if task_id in self.admitted_tasks:
                return GoalResponse.REJECT
            maximum = int(self.get_parameter('max_pending_missions').value)
            if maximum < 1 or len(self.admitted_tasks) >= maximum:
                self.get_logger().warn(
                    'gateway rejected target-scan goal: bounded queue is full')
                return GoalResponse.REJECT
            self.admitted_tasks[task_id] = {
                'mission_sha256': normalized['mission_sha256'],
                'admitted_wall_time_sec': time.time(),
            }
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

    def get_mesh_job_result_cb(self, request, response):
        try:
            payload = self.spool.read('mesh_jobs', request.mesh_job_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            response.found = False
            response.result_json = ''
            response.message = 'mesh job not found: %s' % exc
            return response
        response.found = True
        response.result_json = json.dumps(payload, sort_keys=True)
        response.message = 'durable mesh job returned'
        return response

    def report_tracked_robot_homed_cb(self, request, response):
        try:
            result = self.spool.read('results', request.task_id)
            stamp = (
                float(request.homed_at.sec)
                + float(request.homed_at.nanosec) * 1e-9)
            job_id = validate_home_report(
                request.task_id, request.mesh_job_id,
                request.manifest_sha256, stamp, result)
            try:
                job = self.spool.read('mesh_jobs', job_id)
            except FileNotFoundError:
                job = waiting_job(result)
            state = str(job.get('state', ''))
            if state == 'WAITING_FOR_BASE_HOME':
                self.spool.write('base_home', request.task_id, {
                    'task_id': str(request.task_id),
                    'mesh_job_id': job_id,
                    'manifest_sha256': str(request.manifest_sha256),
                    'homed_at_sec': stamp,
                })
                job = transition_job(
                    job, 'QUEUED',
                    'matching tracked-robot home report accepted')
                self.spool.write('mesh_jobs', job_id, job)
                self.publish_mesh_status(job)
                self.start_reconstruction_job(job)
            elif state in ('QUEUED', 'RUNNING', 'SUCCEEDED'):
                # An identical retry after an uncertain service response is
                # idempotent and never launches a second worker.
                self.publish_mesh_status(job)
            else:
                raise ValueError(
                    'mesh job is terminal FAILED; use a separate retry job')
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            response.accepted = False
            response.mesh_job_id = str(request.mesh_job_id)
            response.state = 'REJECTED'
            response.message = str(exc)
            return response
        response.accepted = True
        response.mesh_job_id = job_id
        response.state = str(job.get('state', 'QUEUED'))
        response.message = str(job.get('reason', 'home report accepted'))
        return response

    def start_reconstruction_job(self, job):
        job_id = str(job['mesh_job_id'])
        with self.lock:
            if job_id in self.active_mesh_jobs:
                return
            self.active_mesh_jobs.add(job_id)
        threading.Thread(
            target=self.run_reconstruction_job,
            args=(dict(job),), daemon=True).start()

    def reconstruction_python(self):
        configured = str(
            self.get_parameter('reconstruction_python').value).strip()
        if configured:
            return configured
        root = Path(str(self.get_parameter('project_root').value)).resolve()
        isolated = root / '.venv-reconstruction' / 'bin' / 'python'
        return str(isolated if isolated.is_file() else Path(sys.executable))

    def run_reconstruction_job(self, queued):
        job_id = str(queued['mesh_job_id'])
        job = dict(queued)
        try:
            job = transition_job(
                job, 'RUNNING', 'reconstruction worker started')
            self.spool.write('mesh_jobs', job_id, job)
            self.publish_mesh_status(job)
            root = Path(str(self.get_parameter('project_root').value)).resolve()
            script = root / 'reconstruction' / 'tsdf_reconstruct.py'
            dataset = Path(str(job['dataset_path'])).resolve()
            output = dataset / 'reconstruction' / 'target_mesh.ply'
            completed = subprocess.run(
                [
                    self.reconstruction_python(), str(script), str(dataset),
                    '--output', str(output),
                ],
                check=False, capture_output=True, text=True,
                timeout=float(self.get_parameter(
                    'reconstruction_timeout_sec').value),
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    'reconstruction worker exited %d: %s'
                    % (completed.returncode, detail[-2000:]))
            report = json.loads(completed.stdout)
            if not isinstance(report, dict):
                raise ValueError(
                    'reconstruction worker result is not an object')
            if Path(str(report.get('mesh_path', ''))).resolve() != output.resolve():
                raise ValueError(
                    'reconstruction worker returned a different mesh path')
            mesh_hash = str(report.get('mesh_sha256', ''))
            if len(mesh_hash) != 64 or not output.is_file():
                raise ValueError(
                    'reconstruction worker did not produce a hashed mesh')
            job = transition_job(
                job, 'SUCCEEDED', 'target mesh reconstruction completed',
                mesh_path=str(output), mesh_sha256=mesh_hash,
                quality_report=report)
        except (
                json.JSONDecodeError, OSError, RuntimeError,
                subprocess.SubprocessError, TypeError, ValueError) as exc:
            if str(job.get('state')) == 'QUEUED':
                job = transition_job(
                    job, 'RUNNING',
                    'reconstruction worker failed during startup')
            job = transition_job(
                job, 'FAILED', 'reconstruction failed: %s' % exc)
        finally:
            with self.lock:
                self.active_mesh_jobs.discard(job_id)
        self.spool.write('mesh_jobs', job_id, job)
        self.publish_mesh_status(job)

    def publish_mesh_status(self, job):
        msg = MeshJobStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.task_id = str(job.get('task_id', ''))
        msg.mesh_job_id = str(job.get('mesh_job_id', ''))
        msg.state = str(job.get('state', ''))
        msg.reason = str(job.get('reason', ''))
        msg.dataset_path = str(job.get('dataset_path', ''))
        msg.manifest_sha256 = str(job.get('manifest_sha256', ''))
        msg.mesh_path = str(job.get('mesh_path', ''))
        msg.mesh_sha256 = str(job.get('mesh_sha256', ''))
        msg.quality_report_json = json.dumps(
            job.get('quality_report', {}), sort_keys=True)
        self.mesh_status_pub.publish(msg)

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
        try:
            local_goal = self.transform_goal(normalized, goal)
            with self.lock:
                admission = dict(self.admitted_tasks.get(task_id, {}))
            local_goal['queue_admitted_wall_time_sec'] = float(
                admission.get('admitted_wall_time_sec', time.time()))
            self.spool.write('goals', task_id, local_goal)
            # A task's execution deadline starts only when it becomes the
            # active physical mission.  Allow bounded predecessors to finish
            # while this action remains queued and heartbeating.
            maximum = int(self.get_parameter('max_pending_missions').value)
            deadline = time.monotonic() + (
                float(normalized['deadline_sec']) * max(1, maximum) + 90.0)
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
                    elif result.get('outcome') == 'CANCELLED':
                        if goal_handle.is_cancel_requested:
                            goal_handle.canceled()
                        else:
                            goal_handle.abort()
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
                self.admitted_tasks.pop(task_id, None)

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
                'target_prompt', 'target_confidence', 'deadline_sec',
                'rough_target')
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
        feedback.required_captures = int(status.get('required_captures', 8))
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
            'REPOSITION_REQUIRED': (
                RunTargetScan.Goal.OUTCOME_REPOSITION_REQUIRED),
        }
        result.outcome = outcomes.get(
            str(payload.get('outcome', 'FAILED')),
            RunTargetScan.Goal.OUTCOME_FAILED)
        result.reason = str(payload.get('reason', ''))
        result.failure_code = str(payload.get('failure_code', ''))
        result.retryable = bool(payload.get('retryable', False))
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
    executor = MultiThreadedExecutor(num_threads=12)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.server.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
