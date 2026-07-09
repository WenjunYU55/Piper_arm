#!/usr/bin/env python3
"""Coordinate obstacle removal and adaptive cube scanning without moving the arm."""

import json
import math
import struct
import time

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

from piper_mobile_manipulation.msg import ObstacleInstance3DArray
from piper_mobile_manipulation.supervised_workflow import (
    choose_removal_plan, cloud_model, distance, point,
)


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
            'movable_whitelist': ['pen'],
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

        self.status_pub = self.create_publisher(String, defaults['status_topic'], 10)
        self.plan_pub = self.create_publisher(String, defaults['plan_topic'], 10)
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
        self.create_service(Trigger, '~/start', self.start_cb)
        self.create_service(Trigger, '~/approve_plan', self.approve_cb)
        self.create_service(Trigger, '~/confirm_action_complete', self.confirm_action_cb)
        self.create_service(Trigger, '~/capture_view', self.capture_view_cb)
        self.create_service(Trigger, '~/finish_scan', self.finish_scan_cb)
        self.create_service(Trigger, '~/abort', self.abort_cb)
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
            self.state = 'SCAN_READY'
            self.publish_status('full-resolution view accepted')

    def cloud_cb(self, msg):
        if msg.header.frame_id != 'base_link':
            self.abort('target cloud frame is not base_link')
            return
        self.cloud_points = self.read_xyz(msg)
        self.cloud_frame = msg.header.frame_id
        self.mark('cloud')
        if self.accepted_views > self.modeled_views:
            self.publish_model()
            self.modeled_views = self.accepted_views

    def start_cb(self, _request, response):
        if self.state not in ('IDLE', 'COMPLETE', 'ABORTED'):
            return self.reply(response, False, 'workflow already active')
        self.accepted_views, self.modeled_views, self.centers = 0, 0, []
        self.target_center = None
        self.plan = None
        self.initial_landmark = None
        self.state = 'INITIALIZING'
        self.publish_status('waiting for locked landmark and fresh obstacle geometry')
        return self.reply(response, True, 'workflow started')

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
        if self.state == 'INITIALIZING':
            if (self.landmark_locked and self.landmark and
                    self.fresh('landmark') and self.fresh('obstacles')):
                self.initial_landmark = self.landmark
                self.assess_scene()
        elif self.state == 'VERIFY_ACTION' and self.fresh('obstacles') and self.fresh('landmark'):
            if self.verify_action():
                self.state = 'INITIALIZING'
                self.publish_status('obstacle action verified; reassessing scene')

    def assess_scene(self):
        movable = [item for item in self.obstacles.instances
                   if item.valid and int(item.classification) == item.CLASSIFICATION_MOVABLE]
        unsafe = [item for item in self.obstacles.instances
                  if not item.valid or int(item.classification) != item.CLASSIFICATION_MOVABLE]
        if unsafe:
            self.abort('scene contains unsafe, blocked, or invalid obstacle geometry')
            return
        if not movable:
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
        self.obstacles_at_plan = {int(item.object_id): point(item.base_centroid)
                                  for item in self.obstacles.instances if item.valid}
        self.publish_json(self.plan_pub, self.plan)
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
        points = []
        for row in range(msg.height):
            for col in range(msg.width):
                offset = row * msg.row_step + col * msg.point_step
                xyz = tuple(
                    struct.unpack_from(
                        endian + 'f', msg.data, offset + fields[name].offset)[0]
                    for name in ('x', 'y', 'z'))
                if all(math.isfinite(value) for value in xyz):
                    points.append(xyz)
        return points


def main(args=None):
    rclpy.init(args=args)
    node = SupervisedCubeWorkflowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
