#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from piper_mobile_manipulation.msg import (
    HandoffTarget,
    ManipulationCommand,
    ManipulationState,
    TrackedTarget,
)
from piper_mobile_manipulation.utils import states
from piper_mobile_manipulation.utils.geometry import offset_pose_away_from_target
from piper_mobile_manipulation.utils.safety import point_in_workspace, safety_reason_for_command


class ManipulationStateMachineNode(Node):
    def __init__(self):
        super().__init__('manipulation_state_machine_node')
        self.declare_parameter('handoff_topic', '/piper/target_piper_base')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('state_topic', '/piper/manipulation_state')
        self.declare_parameter('command_topic', '/piper/manipulation_command')
        self.declare_parameter('task_command', 'inspect')
        self.declare_parameter('pre_grasp_offset_m', 0.12)
        self.declare_parameter('pre_push_offset_m', 0.10)
        self.declare_parameter('final_servo_step_m', 0.01)
        self.declare_parameter('grab_distance_threshold_m', 0.03)
        self.declare_parameter('push_distance_m', 0.05)
        self.declare_parameter('push_speed_mps', 0.02)
        self.declare_parameter('command_period_s', 0.5)
        self.declare_parameter('workspace_x_min', 0.10)
        self.declare_parameter('workspace_x_max', 0.70)
        self.declare_parameter('workspace_y_min', -0.40)
        self.declare_parameter('workspace_y_max', 0.40)
        self.declare_parameter('workspace_z_min', 0.02)
        self.declare_parameter('workspace_z_max', 0.60)
        self.declare_parameter('require_stable_before_grab', True)
        self.declare_parameter('require_valid_tf', True)
        self.declare_parameter('require_valid_depth', True)

        self.state = states.SEARCH
        self.task = self.get_parameter('task_command').value
        self.handoff = None
        self.tracked = None
        self.last_command_state = None
        self.transform_valid = False
        self.depth_valid = False

        self.safety_params = {
            'workspace_x_min': self.get_parameter('workspace_x_min').value,
            'workspace_x_max': self.get_parameter('workspace_x_max').value,
            'workspace_y_min': self.get_parameter('workspace_y_min').value,
            'workspace_y_max': self.get_parameter('workspace_y_max').value,
            'workspace_z_min': self.get_parameter('workspace_z_min').value,
            'workspace_z_max': self.get_parameter('workspace_z_max').value,
            'require_stable_before_grab': self.get_parameter('require_stable_before_grab').value,
            'require_valid_tf': self.get_parameter('require_valid_tf').value,
            'require_valid_depth': self.get_parameter('require_valid_depth').value,
        }

        self.state_pub = self.create_publisher(
            ManipulationState, self.get_parameter('state_topic').value, 10
        )
        self.cmd_pub = self.create_publisher(
            ManipulationCommand, self.get_parameter('command_topic').value, 10
        )
        self.handoff_sub = self.create_subscription(
            HandoffTarget, self.get_parameter('handoff_topic').value, self.handoff_cb, 10
        )
        self.tracked_sub = self.create_subscription(
            TrackedTarget, self.get_parameter('tracked_topic').value, self.tracked_cb, 10
        )
        self.timer = self.create_timer(
            float(self.get_parameter('command_period_s').value), self.tick
        )
        self.get_logger().info('Fake-safe manipulation state machine ready; task=%s' % self.task)

    def handoff_cb(self, msg):
        self.handoff = msg
        self.transform_valid = msg.valid
        if msg.valid and self.state == states.SEARCH:
            self.transition(states.BASE_TARGET_RECEIVED, 'target transformed and available')

    def tracked_cb(self, msg):
        self.tracked = msg
        self.depth_valid = msg.valid

    def tick(self):
        reason = 'waiting'
        visible = self.tracked is not None and self.tracked.valid
        stable = visible and self.tracked.stable
        reachable = visible and point_in_workspace(self.tracked.predicted_position, self.safety_params)

        if self.state == states.SEARCH:
            reason = 'waiting for target handoff'
        elif self.state == states.BASE_TARGET_RECEIVED:
            if self.transform_valid:
                self.publish_command('arm_camera_scan', self.handoff.pose, execute=False)
                self.transition(states.ARM_CAMERA_SCAN, 'arm camera scan requested')
            else:
                self.transition(states.ABORT, 'handoff transform invalid')
        elif self.state == states.ARM_CAMERA_SCAN:
            self.transition(states.TARGET_REFINE, 'waiting for L515 refined target')
        elif self.state == states.TARGET_REFINE:
            if visible:
                self.transition(states.TARGET_TRACK, 'L515 target available')
            else:
                reason = 'waiting for valid L515 depth target'
        elif self.state == states.TARGET_TRACK:
            if visible:
                self.transition(states.PRE_MANIPULATION_POSE, 'filtered target available')
            else:
                reason = 'tracker has no valid target'
        elif self.state == states.PRE_MANIPULATION_POSE:
            if not reachable:
                self.transition(states.ABORT, 'target outside configured workspace')
            else:
                pose = self.tracked_pose()
                offset = self.get_parameter('pre_push_offset_m').value if self.task == 'push' else self.get_parameter('pre_grasp_offset_m').value
                pre_pose = offset_pose_away_from_target(pose, offset)
                self.publish_command('pre_manipulation_pose', pre_pose, execute=False)
                self.transition(states.WAIT_STABLE, 'pre-grasp/pre-push pose requested')
        elif self.state == states.WAIT_STABLE:
            if self.task in ('inspect', 'push', 'sample') or stable:
                self.transition(states.VISUAL_SERVO, 'target stability gate passed')
            else:
                reason = 'waiting for stable target before contact'
        elif self.state == states.VISUAL_SERVO:
            ok, safety_reason = safety_reason_for_command(
                self.task, stable, visible, self.depth_valid, self.transform_valid, self.safety_params
            )
            if not ok:
                self.transition(states.ABORT, safety_reason)
            else:
                self.publish_command('visual_servo_small_correction', self.tracked_pose(), execute=False)
                if self.task == 'grab':
                    self.transition(states.GRAB, 'visual servo correction requested')
                elif self.task == 'push':
                    self.transition(states.PUSH, 'visual servo correction requested')
                elif self.task == 'sample':
                    self.transition(states.SAMPLE, 'visual servo correction requested')
                else:
                    self.transition(states.INSPECT, 'visual servo correction requested')
        elif self.state in (states.GRAB, states.PUSH, states.INSPECT, states.SAMPLE):
            self.publish_task_command(self.state.lower())
            self.transition(states.RETREAT, '%s command printed' % self.state.lower())
        elif self.state == states.RETREAT:
            self.publish_command('retreat', self.tracked_pose() if visible else PoseStamped(), execute=False)
            self.transition(states.DONE, 'retreat command printed')
        elif self.state == states.ABORT:
            self.publish_command('abort_hold', PoseStamped(), execute=False)
            reason = 'abort: holding fake arm command output'
        elif self.state == states.DONE:
            reason = 'done'

        self.publish_state(visible, stable, reachable, reason)

    def tracked_pose(self):
        pose = PoseStamped()
        pose.header = self.tracked.header
        pose.pose.position = self.tracked.predicted_position
        pose.pose.orientation.w = 1.0
        return pose

    def publish_task_command(self, command_type):
        pose = self.tracked_pose()
        cmd = self.publish_command(command_type, pose, execute=False)
        if command_type == 'push':
            cmd.push_direction.x = 1.0
            cmd.speed_limit = float(self.get_parameter('push_speed_mps').value)
            cmd.distance_limit = float(self.get_parameter('push_distance_m').value)
            self.cmd_pub.publish(cmd)

    def publish_command(self, command_type, target_pose, execute=False):
        cmd = ManipulationCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = target_pose.header.frame_id
        cmd.command_type = command_type
        cmd.target_pose = target_pose
        cmd.speed_limit = float(self.get_parameter('push_speed_mps').value)
        cmd.distance_limit = float(self.get_parameter('push_distance_m').value)
        cmd.execute = bool(execute)
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            'Command %s execute=%s frame=%s pos=(%.3f, %.3f, %.3f)'
            % (
                command_type,
                cmd.execute,
                target_pose.header.frame_id,
                target_pose.pose.position.x,
                target_pose.pose.position.y,
                target_pose.pose.position.z,
            )
        )
        return cmd

    def publish_state(self, visible, stable, reachable, reason):
        msg = ManipulationState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = self.state
        msg.target_visible = bool(visible)
        msg.target_stable = bool(stable)
        msg.target_reachable = bool(reachable)
        msg.transform_valid = bool(self.transform_valid)
        msg.depth_valid = bool(self.depth_valid)
        msg.reason = reason
        self.state_pub.publish(msg)

    def transition(self, new_state, reason):
        if self.state != new_state:
            self.get_logger().info('State %s -> %s: %s' % (self.state, new_state, reason))
            self.state = new_state


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationStateMachineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
