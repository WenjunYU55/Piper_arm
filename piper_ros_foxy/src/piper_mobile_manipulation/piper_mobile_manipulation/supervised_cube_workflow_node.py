#!/usr/bin/env python3
"""Coordinate obstacle removal and adaptive cube scanning without moving the arm."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

from piper_mobile_manipulation.msg import (
    ObstacleInstance3DArray, RemovalPlan, ScanStatus, SceneObject, SceneObjectArray,
)
from piper_mobile_manipulation.srv import ScanCommand
from piper_mobile_manipulation.supervised_workflow import (
    choose_removal_plan, cloud_model, distance, point,
)
from piper_mobile_manipulation.utils.piper_kinematics import forward_matrix, solve_link6_pose


class SupervisedCubeWorkflowNode(Node):
    def __init__(self):
        super().__init__('supervised_cube_workflow')
        defaults = {
            'obstacle_topic': '/piper/obstacle_instances_3d',
            'landmark_topic': '/piper/target_landmark',
            'landmark_status_topic': '/piper/target_landmark_status',
            'scan_quality_topic': '/piper/scan_quality',
            'cloud_topic': '/piper/target_cloud',
            'cloud_status_topic': '/piper/target_cloud_status',
            'cloud_request_topic': '/piper/target_cloud_request',
            'status_topic': '/piper/supervised_workflow_status',
            'plan_topic': '/piper/removal_plan',
            'target_model_topic': '/piper/target_model',
            'marker_topic': '/piper/supervised_workflow_markers',
            'typed_plan_topic': '/piper/removal_plan_typed',
            'typed_status_topic': '/piper/scan_status',
            'scene_map_topic': '/piper/scene_objects',
            'movable_whitelist': ['pen'],
            'protected_labels': [
                'person', 'hand', 'finger', 'wire', 'cable', 'tool', 'electronics',
                'unknown object'],
            'ground_labels': ['ground'],
            'min_survey_views': 3,
            'joint_states_topic': '/joint_states_single',
            'joint_bounds_path': '/home/prl/Piper_arm/piper_joint_bounds.json',
            'min_views': 5, 'max_views': 8, 'min_quality_score': 0.40,
            'center_convergence_m': 0.005, 'target_motion_abort_m': 0.020,
            'obstacle_displacement_m': 0.050, 'data_timeout_sec': 2.0,
            'target_clearance_m': 0.040, 'drop_target_clearance_m': 0.120,
            'drop_obstacle_clearance_m': 0.080, 'drop_search_radius_m': 0.180,
            'max_grasp_width_m': 0.070, 'approach_height_m': 0.100,
            'pre_push_offset_m': 0.080, 'push_distance_m': 0.060,
            'workspace_x_min': 0.10, 'workspace_x_max': 0.70,
            'workspace_y_min': -0.40, 'workspace_y_max': 0.40,
            'workspace_z_min': 0.02, 'workspace_z_max': 0.60,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.state = 'IDLE'
        self.landmark = None
        self.initial_landmark = None
        self.landmark_locked = False
        self.obstacles = None
        self.obstacles_at_plan = {}
        self.plan = None
        self.quality = None
        self.cloud_points = []
        self.cloud_frame = ''
        self.cloud_status = None
        self.accepted_views = 0
        self.modeled_views = 0
        self.centers = []
        self.target_center = None
        self.updated = {}
        self.survey_views = 0
        self.survey_object_observations = {}
        self.current_viewpoint = 0
        self.latest_joint_state = None
        self.joint_lower, self.joint_upper = self.load_joint_bounds()
        self.service_group = MutuallyExclusiveCallbackGroup()

        self.status_pub = self.create_publisher(String, defaults['status_topic'], 10)
        self.plan_pub = self.create_publisher(String, defaults['plan_topic'], 10)
        self.typed_plan_pub = self.create_publisher(
            RemovalPlan, defaults['typed_plan_topic'], 10)
        self.typed_status_pub = self.create_publisher(
            ScanStatus, defaults['typed_status_topic'], 10)
        self.scene_map_pub = self.create_publisher(
            SceneObjectArray, defaults['scene_map_topic'], 10)
        self.model_pub = self.create_publisher(String, defaults['target_model_topic'], 10)
        self.marker_pub = self.create_publisher(MarkerArray, defaults['marker_topic'], 10)
        self.target_tf = TransformBroadcaster(self)
        self.cloud_request_pub = self.create_publisher(String, defaults['cloud_request_topic'], 10)
        self.create_subscription(
            ObstacleInstance3DArray, defaults['obstacle_topic'], self.obstacle_cb, 10)
        self.create_subscription(PointStamped, defaults['landmark_topic'], self.landmark_cb, 10)
        self.create_subscription(
            String, defaults['landmark_status_topic'], self.landmark_status_cb, 10)
        self.create_subscription(String, defaults['scan_quality_topic'], self.quality_cb, 10)
        self.create_subscription(PointCloud2, defaults['cloud_topic'], self.cloud_cb, 10)
        self.create_subscription(String, defaults['cloud_status_topic'], self.cloud_status_cb, 10)
        self.create_subscription(
            JointState, defaults['joint_states_topic'], self.joint_state_cb, 10)
        self.create_service(
            Trigger, '~/start', self.start_cb, callback_group=self.service_group)
        self.create_service(
            Trigger, '~/approve_plan', self.approve_cb,
            callback_group=self.service_group)
        self.create_service(
            Trigger, '~/confirm_action_complete', self.confirm_action_cb,
            callback_group=self.service_group)
        self.create_service(
            Trigger, '~/capture_view', self.capture_view_cb,
            callback_group=self.service_group)
        self.create_service(
            Trigger, '~/finish_scan', self.finish_scan_cb,
            callback_group=self.service_group)
        self.create_service(
            Trigger, '~/abort', self.abort_cb, callback_group=self.service_group)
        self.create_service(
            Trigger, '~/capture_survey', self.capture_survey_cb,
            callback_group=self.service_group)
        self.create_service(
            ScanCommand, '~/command', self.command_cb,
            callback_group=self.service_group)
        self.create_timer(0.5, self.tick)
        self.publish_status('dry-run coordinator ready; no arm command publisher exists')

    def now(self):
        return time.monotonic()

    def mark(self, key):
        self.updated[key] = self.now()

    def fresh(self, key):
        timeout = float(self.get_parameter('data_timeout_sec').value)
        return self.now() - self.updated.get(key, -1e9) <= timeout

    def obstacle_cb(self, msg):
        self.obstacles = msg
        self.mark('obstacles')
        self.publish_scene_map(msg)

    def joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def landmark_cb(self, msg):
        self.landmark = point(msg.point)
        self.mark('landmark')
        if self.initial_landmark and distance(self.landmark, self.initial_landmark) > float(
                self.get_parameter('target_motion_abort_m').value):
            self.abort('cube landmark moved beyond tolerance')

    def landmark_status_cb(self, msg):
        payload = self.parse(msg)
        self.landmark_locked = str(payload.get('state', '')).upper() == 'LOCKED'
        self.mark('landmark_status')

    def quality_cb(self, msg):
        self.quality = self.parse(msg)
        self.mark('quality')

    def cloud_status_cb(self, msg):
        self.cloud_status = self.parse(msg)
        self.mark('cloud_status')
        if self.state == 'WAIT_CAPTURE' and self.cloud_status.get('state') == 'accumulating' and \
                self.cloud_status.get('mask_source') == 'full_resolution_refinement':
            self.accepted_views += 1
            self.current_viewpoint += 1
            self.state = 'WAIT_MODEL'
            self.publish_status('full-resolution view accepted; waiting for cloud model')

    def cloud_cb(self, msg):
        # Continuous clouds can be large. Decode only the first cloud needed
        # to model a newly accepted view so service callbacks remain responsive.
        if self.accepted_views <= self.modeled_views:
            return
        if msg.header.frame_id != 'base_link':
            self.abort('target cloud frame is not base_link')
            return
        self.cloud_points = self.read_xyz(msg)
        self.cloud_frame = msg.header.frame_id
        self.mark('cloud')
        if self.accepted_views > self.modeled_views:
            self.publish_model()
            self.modeled_views = self.accepted_views
            if self.state == 'WAIT_MODEL':
                self.state = 'SCAN_READY'
                self.publish_status('accepted cloud modeled; ready for next scan view')

    def start_cb(self, _request, response):
        if self.state not in ('IDLE', 'COMPLETE', 'ABORTED'):
            return self.reply(response, False, 'workflow already active')
        self.accepted_views, self.modeled_views, self.centers = 0, 0, []
        self.target_center = None
        self.plan = None
        self.initial_landmark = None
        self.survey_views = 0
        self.survey_object_observations = {}
        self.state = 'SURVEYING_SCENE'
        self.publish_status(
            'operator-guided survey required; capture at least %d viewpoints' %
            int(self.get_parameter('min_survey_views').value))
        return self.reply(response, True, 'workflow started')

    def capture_survey_cb(self, _request, response):
        ok, message = self.capture_survey()
        return self.reply(response, ok, message)

    def capture_survey(self):
        if self.state != 'SURVEYING_SCENE':
            return False, 'workflow is not surveying the scene'
        if not (self.landmark_locked and self.landmark and self.fresh('landmark') and
                self.fresh('obstacles') and self.obstacles is not None):
            return False, 'locked landmark and fresh obstacle geometry are required'
        if self.initial_landmark is None:
            self.initial_landmark = self.landmark
        for item in self.obstacles.instances:
            if not item.valid:
                continue
            self.survey_object_observations.setdefault(int(item.object_id), []).append(
                point(item.base_centroid))
        self.survey_views += 1
        minimum = int(self.get_parameter('min_survey_views').value)
        if self.survey_views >= minimum:
            self.state = 'ASSESSING_SCENE'
            self.publish_status('survey complete; assessing protected and movable objects')
            self.assess_scene()
            return True, 'survey accepted and scene assessment started'
        self.publish_status('survey view %d/%d accepted' % (self.survey_views, minimum))
        return True, 'survey view accepted'

    def approve_cb(self, _request, response):
        if self.state != 'PLAN_READY' or not self.plan or not self.plan.get('valid'):
            return self.reply(response, False, 'no valid removal plan ready')
        self.state = 'WAIT_OPERATOR_ACTION'
        self.publish_status('plan approved; operator may perform the displayed action')
        return self.reply(response, True, 'dry-run plan approved; no motion was commanded')

    def confirm_action_cb(self, _request, response):
        if self.state != 'WAIT_OPERATOR_ACTION':
            return self.reply(response, False, 'not waiting for operator action')
        self.state = 'VERIFY_ACTION'
        self.publish_status('waiting for fresh post-action perception')
        return self.reply(response, True, 'post-action verification started')

    def capture_view_cb(self, _request, response):
        if self.state != 'SCAN_READY':
            return self.reply(response, False, 'scan is not ready for a view')
        if self.modeled_views < self.accepted_views:
            return self.reply(response, False, 'waiting for the accepted cloud to be modeled')
        if self.accepted_views >= int(self.get_parameter('max_views').value):
            return self.reply(response, False, 'maximum view count reached; finish the scan')
        if not self.fresh('quality') or not self.quality:
            return self.reply(response, False, 'scan quality is missing or stale')
        label = str(self.quality.get('quality_label', self.quality.get('status', ''))).upper()
        score = float(self.quality.get('quality_score', self.quality.get('score', 0.0)))
        minimum = float(self.get_parameter('min_quality_score').value)
        if label not in ('GOOD', 'ACCEPTABLE') or score < minimum:
            return self.reply(response, False, 'view quality rejected: %s %.2f' % (label, score))
        msg = String()
        msg.data = 'capture'
        self.cloud_request_pub.publish(msg)
        self.state = 'WAIT_CAPTURE'
        self.publish_status('full-resolution cloud capture requested')
        return self.reply(response, True, 'capture requested; hold the arm stationary')

    def finish_scan_cb(self, _request, response):
        if self.state != 'SCAN_READY':
            return self.reply(response, False, 'scan is not ready to finish')
        if self.modeled_views < self.accepted_views:
            return self.reply(response, False, 'waiting for the latest target model')
        minimum = int(self.get_parameter('min_views').value)
        if self.accepted_views < minimum:
            return self.reply(response, False, 'need at least %d accepted views' % minimum)
        convergence = float(self.get_parameter('center_convergence_m').value)
        converged = (len(self.centers) >= 2 and
                     distance(self.centers[-1], self.centers[-2]) <= convergence)
        if not converged and self.accepted_views < int(self.get_parameter('max_views').value):
            return self.reply(response, False, 'center has not converged; capture another view')
        msg = String()
        msg.data = 'save'
        self.cloud_request_pub.publish(msg)
        self.state = 'COMPLETE'
        self.publish_model()
        self.publish_status('scan complete; cloud save requested')
        return self.reply(response, True, 'scan complete')

    def abort_cb(self, _request, response):
        self.abort('operator requested abort')
        return self.reply(response, True, 'workflow aborted; no arm stop command was required')

    def tick(self):
        if self.target_center is not None:
            self.publish_target_frame(self.target_center)
        if self.state == 'VERIFY_ACTION' and self.fresh('obstacles') and self.fresh('landmark'):
            if self.verify_action():
                self.state = 'SURVEYING_SCENE'
                self.survey_views = 0
                self.survey_object_observations = {}
                self.publish_status(
                    'obstacle action verified; repeat scene survey before replanning')

    def assess_scene(self):
        movable = [item for item in self.obstacles.instances
                   if item.valid and int(item.classification) == item.CLASSIFICATION_MOVABLE
                   and self.canonical_label(item.semantic_label) != 'ground']
        unsafe = [item for item in self.obstacles.instances
                  if (not item.valid or int(item.classification) != item.CLASSIFICATION_MOVABLE)
                  and self.canonical_label(item.semantic_label) != 'ground']
        if not movable:
            if self.obstacles.scene_blocked or unsafe:
                self.abort('scene is blocked but no verified movable obstacle can be removed')
                return
            self.state = 'SCAN_READY'
            self.publish_status('scene is clear; ready for first scan view')
            return
        selected = min(
            movable,
            key=lambda item: distance(point(item.base_centroid), self.landmark))
        config = {name: self.get_parameter(name).value for name in (
            'movable_whitelist', 'target_clearance_m', 'drop_target_clearance_m',
            'drop_obstacle_clearance_m', 'drop_search_radius_m', 'max_grasp_width_m',
            'approach_height_m', 'pre_push_offset_m', 'push_distance_m',
            'workspace_x_min', 'workspace_x_max', 'workspace_y_min', 'workspace_y_max',
            'workspace_z_min', 'workspace_z_max')}
        self.plan = choose_removal_plan(selected, self.landmark, self.obstacles.instances, config)
        self.validate_plan_ik(self.plan)
        self.obstacles_at_plan = {int(item.object_id): point(item.base_centroid)
                                  for item in self.obstacles.instances if item.valid}
        self.publish_json(self.plan_pub, self.plan)
        self.publish_typed_plan(self.plan)
        self.publish_markers(self.plan)
        if not self.plan.get('valid'):
            self.abort('removal planning failed: %s' % self.plan.get('reason', 'unknown'))
            return
        self.state = 'PLAN_READY'
        self.publish_status('removal plan ready for operator review')

    def verify_action(self):
        object_id = int(self.plan['object_id'])
        current = next(
            (item for item in self.obstacles.instances
             if int(item.object_id) == object_id),
            None)
        if current and current.valid:
            moved = distance(point(current.base_centroid), self.obstacles_at_plan[object_id])
            if moved < float(self.get_parameter('obstacle_displacement_m').value):
                self.publish_status('planned obstacle has not moved far enough')
                return False
        remaining = [item for item in self.obstacles.instances
                     if not item.valid or int(item.classification) != item.CLASSIFICATION_MOVABLE]
        if remaining:
            self.abort('post-action scene contains unsafe or invalid obstacles')
            return False
        return True

    def publish_model(self):
        model = cloud_model(self.cloud_points, self.cloud_frame, self.accepted_views)
        if model.get('valid'):
            center = tuple(model['center'])
            if not self.centers or distance(center, self.centers[-1]) > 1e-9:
                self.centers.append(center)
            model['center_delta_m'] = (distance(self.centers[-1], self.centers[-2])
                                       if len(self.centers) >= 2 else None)
            model['confidence'] = min(1.0, self.accepted_views / float(
                max(1, int(self.get_parameter('min_views').value))))
            self.target_center = center
            self.publish_target_frame(center)
        self.publish_json(self.model_pub, model)

    def publish_target_frame(self, center):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'target_local'
        transform.transform.translation.x = float(center[0])
        transform.transform.translation.y = float(center[1])
        transform.transform.translation.z = float(center[2])
        transform.transform.rotation.w = 1.0
        self.target_tf.sendTransform(transform)

    def publish_status(self, reason):
        self.publish_json(self.status_pub, {
            'state': self.state, 'reason': reason, 'dry_run': True,
            'real_arm_motion': False, 'accepted_views': self.accepted_views,
            'min_views': int(self.get_parameter('min_views').value),
            'max_views': int(self.get_parameter('max_views').value),
        })
        typed = ScanStatus()
        typed.header.stamp = self.get_clock().now().to_msg()
        typed.header.frame_id = 'base_link'
        typed.state_name = self.state
        typed.state = self.status_code(self.state)
        typed.reason = str(reason)
        typed.current_viewpoint = int(self.current_viewpoint)
        typed.accepted_views = int(self.accepted_views)
        typed.total_viewpoints = int(self.get_parameter('max_views').value)
        typed.dry_run = True
        typed.motion_commands_enabled = False
        self.typed_status_pub.publish(typed)

    def publish_scene_map(self, obstacles):
        output = SceneObjectArray()
        output.header = obstacles.header
        output.header.frame_id = 'base_link'
        output.unseen_space_is_occupied = True
        ground_heights = []
        for item in obstacles.instances:
            scene = SceneObject()
            scene.header = item.header
            scene.header.frame_id = 'base_link'
            scene.object_id = int(item.object_id)
            scene.semantic_label = self.canonical_label(item.semantic_label)
            if scene.semantic_label == 'ground':
                scene.classification = SceneObject.CLASS_GROUND
                ground_heights.append(float(item.base_bounds_max.z))
            elif item.valid and int(item.classification) == item.CLASSIFICATION_MOVABLE:
                scene.classification = SceneObject.CLASS_MOVABLE
            elif not item.valid or not scene.semantic_label:
                scene.classification = SceneObject.CLASS_UNKNOWN
            else:
                scene.classification = SceneObject.CLASS_PROTECTED
            scene.pose.pose.position = item.base_centroid
            scene.pose.pose.orientation.w = 1.0
            scene.size.x = max(0.0, item.base_bounds_max.x - item.base_bounds_min.x)
            scene.size.y = max(0.0, item.base_bounds_max.y - item.base_bounds_min.y)
            scene.size.z = max(0.0, item.base_bounds_max.z - item.base_bounds_min.z)
            scene.confidence = float(item.confidence)
            scene.observation_count = len(
                self.survey_object_observations.get(int(item.object_id), []))
            scene.valid = bool(item.valid)
            scene.reason = str(item.validity_reason)
            output.objects.append(scene)
        output.ground_normal.z = 1.0
        output.ground_valid = bool(ground_heights)
        output.ground_offset = float(np.median(ground_heights)) if ground_heights else 0.0
        self.scene_map_pub.publish(output)

    def publish_typed_plan(self, plan):
        output = RemovalPlan()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = 'base_link'
        output.object_id = int(plan.get('object_id', 0))
        output.semantic_label = str(plan.get('label', ''))
        output.action = (RemovalPlan.ACTION_PICK_AND_PLACE
                         if plan.get('action') == 'pick_and_place'
                         else RemovalPlan.ACTION_PUSH if plan.get('action') == 'push'
                         else RemovalPlan.ACTION_NONE)
        self.set_pose_position(output.approach_pose, plan.get('approach'))
        self.set_pose_position(output.action_pose, plan.get('object_center'))
        self.set_pose_position(
            output.destination_pose, plan.get('drop_center', plan.get('push_end')))
        self.set_pose_position(output.retreat_pose, plan.get('retreat'))
        output.risk_score = float(plan.get('risk_score', 1.0))
        output.minimum_clearance_m = float(self.get_parameter('drop_obstacle_clearance_m').value)
        output.destination_observed = self.survey_views >= int(
            self.get_parameter('min_survey_views').value)
        output.destination_empty = bool(plan.get('valid', False))
        output.ik_valid = bool(plan.get('ik_valid', False))
        output.valid = bool(plan.get('valid', False))
        output.dry_run = True
        output.reason = str(plan.get('reason', ''))
        solutions = plan.get('joint_solutions', {})
        output.approach_joints = solutions.get('approach', [0.0] * 6)
        output.action_joints = solutions.get('object_center', [0.0] * 6)
        output.destination_joints = solutions.get('destination', [0.0] * 6)
        output.retreat_joints = solutions.get('retreat', [0.0] * 6)
        self.typed_plan_pub.publish(output)

    def validate_plan_ik(self, plan):
        plan['ik_valid'] = False
        if not plan.get('valid'):
            return
        if self.latest_joint_state is None or len(self.latest_joint_state.position) < 6:
            plan['valid'] = False
            plan['reason'] = 'fresh joint feedback is required for removal IK'
            return
        seed = np.asarray(self.latest_joint_state.position[:6], dtype=float)
        orientation = forward_matrix(seed)[:3, :3]
        points = {
            'approach': plan.get('approach'),
            'object_center': plan.get('object_center'),
            'destination': plan.get('drop_center', plan.get('push_end')),
            'retreat': plan.get('retreat'),
        }
        solutions = {}
        for name, xyz in points.items():
            if xyz is None:
                plan['valid'] = False
                plan['reason'] = 'removal plan has no %s pose' % name
                return
            desired = np.eye(4)
            desired[:3, :3] = orientation
            desired[:3, 3] = np.asarray(xyz, dtype=float)
            solution, converged, details = solve_link6_pose(
                desired, seed, self.joint_lower, self.joint_upper,
                position_tolerance=0.008, rotation_tolerance=np.deg2rad(5.0))
            if not converged:
                plan['valid'] = False
                plan['reason'] = 'IK rejected %s pose (position %.3fm, rotation %.1fdeg)' % (
                    name, details['position_error_m'],
                    np.rad2deg(details['rotation_error_rad']))
                return
            solutions[name] = [float(value) for value in solution]
            seed = solution
        plan['joint_solutions'] = solutions
        plan['ik_valid'] = True

    def load_joint_bounds(self):
        path = Path(str(self.get_parameter('joint_bounds_path').value)).expanduser()
        with path.open('r', encoding='utf-8') as stream:
            payload = json.load(stream)
        lower, upper = [], []
        for index in range(1, 7):
            item = payload['joints']['joint%d' % index]
            lower.append(float(item.get('command_min', item['min'])))
            upper.append(float(item.get('command_max', item['max'])))
        return np.asarray(lower), np.asarray(upper)

    def command_cb(self, request, response):
        accepted, message = False, 'command not valid in current state'
        if request.command == ScanCommand.Request.START_SURVEY:
            if self.state in ('IDLE', 'COMPLETE', 'ABORTED'):
                self.survey_views = 0
                self.survey_object_observations = {}
                self.state = 'SURVEYING_SCENE'
                accepted, message = True, 'survey started'
        elif request.command == ScanCommand.Request.CAPTURE_SURVEY:
            accepted, message = self.capture_survey()
        elif request.command == ScanCommand.Request.APPROVE_PLAN and self.state == 'PLAN_READY':
            self.state = 'WAIT_OPERATOR_ACTION'
            accepted, message = True, 'plan approved; no motion commanded'
        elif (request.command == ScanCommand.Request.CONFIRM_ACTION_COMPLETE and
              self.state == 'WAIT_OPERATOR_ACTION'):
            self.state = 'VERIFY_ACTION'
            accepted, message = True, 'post-action verification started'
        elif request.command == ScanCommand.Request.START_SCAN and self.state == 'SCAN_READY':
            self.current_viewpoint = 0
            accepted, message = True, 'dry-run scan sequence started'
        elif (request.command == ScanCommand.Request.ACKNOWLEDGE_VIEWPOINT and
              self.state == 'SCAN_READY'):
            self.current_viewpoint = int(request.viewpoint_index)
            accepted, message = True, 'viewpoint acknowledged; capture may be requested'
        elif request.command == ScanCommand.Request.CAPTURE_VIEW:
            legacy = self.capture_view_cb(None, SimpleNamespace())
            accepted, message = bool(legacy.success), str(legacy.message)
        elif request.command == ScanCommand.Request.SKIP_VIEW and self.state == 'SCAN_READY':
            self.current_viewpoint = max(
                self.current_viewpoint + 1, int(request.viewpoint_index) + 1)
            accepted, message = True, 'viewpoint skipped'
        elif request.command == ScanCommand.Request.FINALIZE:
            legacy = self.finish_scan_cb(None, SimpleNamespace())
            accepted, message = bool(legacy.success), str(legacy.message)
        elif request.command == ScanCommand.Request.CLEAR:
            clear = String()
            clear.data = 'clear'
            self.cloud_request_pub.publish(clear)
            self.accepted_views, self.modeled_views, self.centers = 0, 0, []
            self.state = 'IDLE'
            accepted, message = True, 'workflow and target cloud cleared'
        elif request.command == ScanCommand.Request.ABORT:
            self.abort(request.reason or 'operator requested abort')
            accepted, message = True, 'workflow aborted'
        response.accepted = bool(accepted)
        response.state = self.status_code(self.state)
        response.state_name = self.state
        response.message = message
        self.publish_status(message)
        return response

    @staticmethod
    def set_pose_position(pose, xyz):
        pose.orientation.w = 1.0
        if xyz is not None:
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]

    @staticmethod
    def canonical_label(label):
        words = set(str(label or '').lower().replace('_', ' ').split())
        if words.intersection({'pen', 'marker'}):
            return 'pen'
        if 'ground' in words:
            return 'ground'
        return ' '.join(sorted(words)) or 'unknown object'

    @staticmethod
    def status_code(state):
        mapping = {
            'IDLE': ScanStatus.IDLE, 'SURVEYING_SCENE': ScanStatus.SURVEYING_SCENE,
            'ASSESSING_SCENE': ScanStatus.ASSESSING_SCENE,
            'PLAN_READY': ScanStatus.PLAN_READY,
            'WAIT_OPERATOR_ACTION': ScanStatus.WAITING_FOR_OPERATOR,
            'VERIFY_ACTION': ScanStatus.VERIFYING_REMOVAL,
            'SCAN_READY': ScanStatus.PLANNING_SCAN,
            'WAIT_CAPTURE': ScanStatus.CAPTURING, 'WAIT_MODEL': ScanStatus.REGISTERING,
            'COMPLETE': ScanStatus.COMPLETE, 'ABORTED': ScanStatus.ABORTED,
        }
        return mapping.get(state, ScanStatus.PAUSED)

    def abort(self, reason):
        if self.state == 'ABORTED':
            return
        self.state = 'ABORTED'
        self.publish_status(reason)

    def publish_markers(self, plan):
        if not plan.get('valid'):
            return
        array = MarkerArray()
        points = [('object', plan.get('object_center'), (1.0, 0.2, 0.1)),
                  ('approach', plan.get('approach'), (1.0, 0.8, 0.0)),
                  ('destination', plan.get('drop_center', plan.get('push_end')), (0.1, 1.0, 0.2))]
        for index, (name, xyz, color) in enumerate(points):
            if xyz is None:
                continue
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'supervised_workflow'
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = xyz
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.025
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.9
            marker.text = name
            array.markers.append(marker)
        self.marker_pub.publish(array)

    @staticmethod
    def parse(msg):
        try:
            return json.loads(msg.data)
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        pub.publish(msg)

    @staticmethod
    def reply(response, success, message):
        response.success = bool(success)
        response.message = str(message)
        return response

    @staticmethod
    def read_xyz(msg):
        fields = {field.name: field for field in msg.fields}
        if not all(name in fields for name in ('x', 'y', 'z')):
            return []
        endian = '>' if msg.is_bigendian else '<'
        dtype = np.dtype({
            'names': ('x', 'y', 'z'),
            'formats': (endian + 'f4',) * 3,
            'offsets': tuple(fields[name].offset for name in ('x', 'y', 'z')),
            'itemsize': msg.point_step,
        })
        values = np.ndarray(
            shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data,
            strides=(msg.row_step, msg.point_step))
        points = np.column_stack(
            (values['x'].ravel(), values['y'].ravel(), values['z'].ravel()))
        return points[np.all(np.isfinite(points), axis=1)]


def main(args=None):
    rclpy.init(args=args)
    node = SupervisedCubeWorkflowNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
