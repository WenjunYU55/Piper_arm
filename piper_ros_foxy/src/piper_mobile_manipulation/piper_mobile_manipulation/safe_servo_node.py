#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piper_mobile_manipulation.msg import ServoCommand, Target3D


class SafeServoNode(Node):
    def __init__(self):
        super().__init__('safe_servo_node')
        self.declare_parameter('manipulation_target_topic', '/piper/manipulation_target')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('servo_cmd_topic', '/piper/servo_cmd')
        self.declare_parameter('enable_real_arm_motion', False)
        self.declare_parameter('min_depth_m', 0.25)
        self.declare_parameter('max_target_jump_m', 0.03)
        self.declare_parameter('max_speed', 0.01)
        self.declare_parameter('gain_xy', 0.2)
        self.declare_parameter('gain_z', 0.2)

        self.target_status = 'SEARCHING'
        self.arm_status = ''
        self.last_target = None
        self.pub = self.create_publisher(ServoCommand, self.get_parameter('servo_cmd_topic').value, 10)
        self.target_sub = self.create_subscription(
            Target3D,
            self.get_parameter('manipulation_target_topic').value,
            self.target_cb,
            10,
        )
        self.status_sub = self.create_subscription(String, self.get_parameter('target_status_topic').value, self.status_cb, 10)
        self.arm_status_sub = self.create_subscription(String, '/arm_status', self.arm_status_cb, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states_single', self.joint_cb, 10)
        self.get_logger().warn('Safe servo started with enable_real_arm_motion=false by default')

    def status_cb(self, msg):
        self.target_status = msg.data

    def arm_status_cb(self, msg):
        self.arm_status = msg.data

    def joint_cb(self, _msg):
        pass

    def target_cb(self, msg):
        cmd = ServoCommand()
        cmd.header = msg.header
        cmd.command = 'hold'
        cmd.max_speed = float(self.get_parameter('max_speed').value)
        cmd.gain_xy = float(self.get_parameter('gain_xy').value)
        cmd.gain_z = float(self.get_parameter('gain_z').value)

        reason = self.stop_reason(msg)
        if reason:
            cmd.valid = False
            self.pub.publish(cmd)
            self.get_logger().warn('servo hold: %s' % reason)
            return

        cmd.linear.x = msg.point.x
        cmd.linear.y = msg.point.y
        cmd.linear.z = msg.point.z
        real_motion_enabled = self.param_bool('enable_real_arm_motion')
        cmd.command = 'track_target' if real_motion_enabled else 'dry_run_track_target'
        cmd.valid = real_motion_enabled
        self.pub.publish(cmd)
        self.last_target = (msg.point.x, msg.point.y, msg.point.z)
        if not cmd.valid:
            self.get_logger().info(
                'dry-run servo target=(%.3f, %.3f, %.3f); no real arm motion'
                % (msg.point.x, msg.point.y, msg.point.z)
            )

    def stop_reason(self, msg):
        if self.target_status in ('LOW_CONFIDENCE', 'LOST'):
            return 'target_status=%s' % self.target_status
        if not msg.valid or not math.isfinite(msg.point.z):
            return 'invalid target depth'
        if msg.point.z < float(self.get_parameter('min_depth_m').value):
            return 'target too close'
        if 'error' in self.arm_status.lower() or 'fault' in self.arm_status.lower():
            return 'arm status reports error'
        if self.last_target is not None:
            jump = math.sqrt(
                (msg.point.x - self.last_target[0]) ** 2
                + (msg.point.y - self.last_target[1]) ** 2
                + (msg.point.z - self.last_target[2]) ** 2
            )
            if jump > float(self.get_parameter('max_target_jump_m').value):
                return 'target jump %.3f too large' % jump
        return None

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = SafeServoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
