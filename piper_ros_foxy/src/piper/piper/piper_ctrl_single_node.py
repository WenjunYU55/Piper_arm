#!/usr/bin/env python3
# -*-coding:utf8-*-
# This file controls a single robotic arm node and handles the movement of the robotic arm with a gripper.
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import threading
import math
import hashlib
import json
import os
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperMotionLimits, PiperStatusMsg, PosCmd
from piper_msgs.srv import Enable
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R  # For Euler angle to quaternion conversion
from numpy import clip


DEFAULT_JOINT_BOUNDS = {
    'joint1': (-2.8, 2.8),
    'joint2': (-2.1, 2.1),
    'joint3': (-2.8, 2.8),
    'joint4': (-2.8, 2.8),
    'joint5': (-2.1, 2.1),
    'joint6': (-2.0944, 2.0944),
    'joint7': (0.0, 0.08),
}

ENABLE_RETRY_PERIOD_SEC = 0.01
MOTION_LIMIT_JOINT_NAMES = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
]
MAX_PROTOCOL_JOINT_SPEED_RAD_S = 3.0
MAX_PROTOCOL_JOINT_ACCELERATION_RAD_S2 = 5.0


def motion_limits_sha256(velocities, accelerations):
    """Hash the controller limit values used to time a trajectory."""
    payload = {
        'joint_names': list(MOTION_LIMIT_JOINT_NAMES),
        'max_velocity_rad_s': [round(float(value), 9) for value in velocities],
        'max_acceleration_rad_s2': [
            round(float(value), 9) for value in accelerations
        ],
        'source': 'piper_sdk_controller_feedback',
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def controller_motion_limits(piper, now_sec=None, maximum_age_sec=10.0):
    """Return validated read-only J1-J6 limits cached by the PiPER SDK."""
    now = time.time() if now_sec is None else float(now_sec)
    try:
        speed_feedback = piper.GetAllMotorAngleLimitMaxSpd()
        acceleration_feedback = piper.GetAllMotorMaxAccLimit()
        speed_stamp = float(speed_feedback.time_stamp)
        acceleration_stamp = float(acceleration_feedback.time_stamp)
        speed_motors = speed_feedback.all_motor_angle_limit_max_spd.motor
        acceleration_motors = \
            acceleration_feedback.all_motor_max_acc_limit.motor
        velocities = []
        accelerations = []
        for index in range(1, 7):
            speed = speed_motors[index]
            acceleration = acceleration_motors[index]
            if int(speed.motor_num) != index:
                raise ValueError('speed feedback is missing joint%d' % index)
            if int(acceleration.joint_motor_num) != index:
                raise ValueError(
                    'acceleration feedback is missing joint%d' % index)
            velocities.append(float(speed.max_joint_spd) * 0.001)
            accelerations.append(float(acceleration.max_joint_acc) * 0.001)
        if not all(
                math.isfinite(value)
                and 0.0 < value <= MAX_PROTOCOL_JOINT_SPEED_RAD_S
                for value in velocities):
            raise ValueError('controller velocity limits are invalid')
        if not all(
                math.isfinite(value)
                and 0.0 < value <= MAX_PROTOCOL_JOINT_ACCELERATION_RAD_S2
                for value in accelerations):
            raise ValueError('controller acceleration limits are invalid')
        source_stamp = min(speed_stamp, acceleration_stamp)
        if (
                not math.isfinite(source_stamp)
                or source_stamp <= 0.0
                or now - source_stamp > float(maximum_age_sec)):
            raise ValueError('controller motion-limit feedback is stale')
        return {
            'valid': True,
            'velocities': velocities,
            'accelerations': accelerations,
            'limits_sha256': motion_limits_sha256(
                velocities, accelerations),
            'reason': 'fresh controller limits',
        }
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        # Keep the serialized shape at its maximum even while invalid. Foxy's
        # Fast DDS can size a writer history from the first sample; publishing
        # empty arrays first makes the later valid six-joint payload too large
        # for that history and it is silently lost. The valid flag remains the
        # authoritative fail-closed gate.
        return {
            'valid': False,
            'velocities': [0.0] * 6,
            'accelerations': [0.0] * 6,
            'limits_sha256': '0' * 64,
            'reason': str(error),
        }


def request_piper_enable_state(piper, enabled, timeout_sec,
                               monotonic_fn=time.monotonic, sleep_fn=time.sleep):
    """Use the SDK's feedback-confirmed enable/disable handshake."""
    deadline = monotonic_fn() + max(0.0, float(timeout_sec))
    while True:
        if enabled:
            target_reached = bool(piper.EnablePiper())
        else:
            target_reached = not bool(piper.DisablePiper())

        if target_reached:
            return True
        if monotonic_fn() >= deadline:
            return False
        sleep_fn(ENABLE_RETRY_PERIOD_SEC)


class PiperRosNode(Node):
    """ROS2 node for the robotic arm"""

    def __init__(self) -> None:
        super().__init__('piper_ctrl_single_node')
        # ROS parameters
        self.declare_parameter('can_port', 'can0')
        self.declare_parameter('auto_enable', False)
        self.declare_parameter('gripper_exist', True)
        self.declare_parameter('enable_timeout', 15.0)
        self.declare_parameter('joint_bounds_path', '')
        self.declare_parameter('motion_limit_query_period_sec', 5.0)
        self.declare_parameter('motion_limit_max_age_sec', 10.0)

        self.can_port = self.get_parameter('can_port').get_parameter_value().string_value
        self.auto_enable = self.get_parameter('auto_enable').get_parameter_value().bool_value
        self.gripper_exist = self.get_parameter('gripper_exist').get_parameter_value().bool_value
        self.enable_timeout = self.get_parameter('enable_timeout').get_parameter_value().double_value
        self.joint_bounds_path = self.get_parameter('joint_bounds_path').get_parameter_value().string_value
        self.joint_bounds = self.load_joint_bounds(self.joint_bounds_path)

        self.get_logger().info(f"can_port is {self.can_port}")
        self.get_logger().info(f"auto_enable is {self.auto_enable}")
        self.get_logger().info(f"gripper_exist is {self.gripper_exist}")
        self.get_logger().info(f"enable_timeout is {self.enable_timeout}")
        self.get_logger().info(f"joint_bounds_path is {self.joint_bounds_path or 'default limits'}")
        self.feedback_callback_group = MutuallyExclusiveCallbackGroup()
        self.motion_limit_callback_group = MutuallyExclusiveCallbackGroup()
        self.motion_limit_query_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.command_callback_group = MutuallyExclusiveCallbackGroup()
        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states_single', 10)
        self.arm_status_pub = self.create_publisher(PiperStatusMsg, 'arm_status', 10)
        self.motion_limits_pub = self.create_publisher(
            PiperMotionLimits, '/piper/motion_limits', 10)
        self.end_pose_pub = self.create_publisher(Pose, 'end_pose', 10)
        # Service
        self.motor_srv = self.create_service(
            Enable,
            'enable_srv',
            self.handle_enable_service,
            callback_group=self.service_callback_group,
        )
        # Joint
        self.joint_states = JointState()
        self.joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        self.joint_states.position = [0.0] * 8
        self.joint_states.velocity = [0.0] * 8
        self.joint_states.effort = [0.0] * 8
        # Enable flag
        self.__enable_flag = False
        self._command_cache_lock = threading.Lock()
        self._motion_ctrl_2_signature = None
        self._gripper_command_signature = None
        # Create piper class and open CAN interface
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

        # Start subscription thread
        self.create_subscription(
            PosCmd, 'pos_cmd', self.pos_callback, 10,
            callback_group=self.command_callback_group)
        self.create_subscription(
            JointState, 'joint_ctrl_single', self.joint_callback, 10,
            callback_group=self.command_callback_group)
        self.create_subscription(
            Bool, 'enable_flag', self.enable_callback, 10,
            callback_group=self.command_callback_group)

        # Keep all ROS publication on the node executor. Publishing from an
        # unmanaged Python thread can silently stop delivering after prolonged
        # Foxy graph churn even while CAN reception and the process stay alive.
        self.feedback_timer = self.create_timer(
            1.0 / 200.0,
            self.publish_feedback,
            callback_group=self.feedback_callback_group,
        )
        self.motion_limits_timer = self.create_timer(
            1.0,
            self.publish_motion_limits,
            callback_group=self.motion_limit_callback_group,
        )
        self.motion_limit_query_timer = self.create_timer(
            float(self.get_parameter('motion_limit_query_period_sec').value),
            self.query_motion_limits,
            callback_group=self.motion_limit_query_callback_group,
        )
        self.feedback_timer_started = False
        self.motion_limits_timer_started = False
        if self.auto_enable:
            self.auto_enable_thread = threading.Thread(
                target=self.auto_enable_loop, daemon=True)
            self.auto_enable_thread.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def reset_command_cache(self):
        with self._command_cache_lock:
            self._motion_ctrl_2_signature = None
            self._gripper_command_signature = None

    def send_motion_ctrl_2_if_changed(self, mode_ctrl, move_mode, speed_percent):
        """Set the SDK MoveJ mode/speed only when its value actually changes."""
        signature = (
            int(mode_ctrl),
            int(move_mode),
            int(clip(round(float(speed_percent)), 0, 100)),
        )
        with self._command_cache_lock:
            if signature == self._motion_ctrl_2_signature:
                return False
            self.piper.MotionCtrl_2(*signature)
            self._motion_ctrl_2_signature = signature
        return True

    def send_gripper_if_changed(self, angle, effort, status_code, set_zero=0):
        """Avoid resending an unchanged gripper command with every arm sample."""
        signature = (
            int(angle), int(effort), int(status_code), int(set_zero))
        with self._command_cache_lock:
            if signature == self._gripper_command_signature:
                return False
            self.piper.GripperCtrl(*signature)
            self._gripper_command_signature = signature
        return True

    def load_joint_bounds(self, path):
        bounds = dict(DEFAULT_JOINT_BOUNDS)
        if not path:
            return bounds
        if not os.path.exists(path):
            self.get_logger().warn(f"Joint bounds file not found: {path}. Using default limits.")
            return bounds

        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            saved = data.get('joints', {})
            for joint_name in bounds:
                record = saved.get(joint_name)
                if record is None or record.get('valid', True) is False:
                    continue
                low = float(record.get('min', bounds[joint_name][0]))
                high = float(record.get('max', bounds[joint_name][1]))
                if low == high:
                    self.get_logger().warn(f"Ignoring zero-width bound for {joint_name}")
                    continue
                bounds[joint_name] = (min(low, high), max(low, high))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"Could not load joint bounds from {path}: {exc}. Using default limits.")
            return bounds

        self.get_logger().info(f"Loaded hard joint bounds from {path}: {bounds}")
        return bounds

    def enforce_joint_bound(self, joint_name, value):
        low, high = self.joint_bounds[joint_name]
        bounded = float(clip(value, low, high))
        if bounded != value:
            self.get_logger().warn(
                f"Hard bound clipped {joint_name}: requested {value}, using {bounded} "
                f"within [{low}, {high}]"
            )
        return bounded

    def get_joint_value(self, joint_data, joint_name, fallback_index, default=0.0):
        if joint_name in joint_data.name:
            name_index = joint_data.name.index(joint_name)
            if name_index < len(joint_data.position):
                return joint_data.position[name_index]
        if fallback_index < len(joint_data.position):
            return joint_data.position[fallback_index]
        return default

    def get_joint_effort(self, joint_data, joint_name, fallback_index, default=0.0):
        if joint_name in joint_data.name:
            name_index = joint_data.name.index(joint_name)
            if name_index < len(joint_data.effort):
                return joint_data.effort[name_index]
        if fallback_index < len(joint_data.effort):
            return joint_data.effort[fallback_index]
        return default

    def get_joint_velocity(self, joint_data, joint_name, fallback_index, default=0.0):
        if joint_name in joint_data.name:
            name_index = joint_data.name.index(joint_name)
            if name_index < len(joint_data.velocity):
                return joint_data.velocity[name_index]
        if fallback_index < len(joint_data.velocity):
            return joint_data.velocity[fallback_index]
        return default

    def auto_enable_loop(self):
        """Perform the optional startup enable handshake outside the executor."""
        enable_flag = False
        deadline = time.monotonic() + self.enable_timeout
        while rclpy.ok() and not enable_flag:
            enable_flag = (
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            if enable_flag:
                self.__enable_flag = True
                return
            self.piper.EnableArm(7)
            self.send_gripper_if_changed(0, 1000, 0x01, 0)
            if time.monotonic() >= deadline:
                self.get_logger().error(
                    'Automatic enable timed out; feedback publication remains active')
                return
            time.sleep(1.0)

    def publish_feedback(self):
        """Publish one executor-owned feedback sample from the robotic arm."""
        if not self.feedback_timer_started:
            self.feedback_timer_started = True
            self.get_logger().info(
                'Executor-owned 200 Hz feedback timer is active')
        self.PublishArmState()
        self.PublishArmJointAndGripper()
        self.PublishArmEndPose()

    def PublishArmState(self):
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        arm_status.communication_status_joint_2 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        arm_status.communication_status_joint_3 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        arm_status.communication_status_joint_4 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        arm_status.communication_status_joint_5 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        arm_status.communication_status_joint_6 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        self.arm_status_pub.publish(arm_status)

    def PublishArmJointAndGripper(self):
        # Assign timestamp
        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        # Here, you can set the joint positions to any value you want
        # The raw data obtained is in degrees multiplied by 1000. To convert to radians, divide by 1000, multiply by π/180, and limit to 5 decimal places
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
        self.joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, -joint_6]
        self.joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0, 0.0]
        self.joint_states.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_6, effort_6]
        self.joint_pub.publish(self.joint_states)

    def PublishArmEndPose(self):
        # End effector pose
        endpos = Pose()
        endpos.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        endpos.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        endpos.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        endpos.orientation.x = quaternion[0]
        endpos.orientation.y = quaternion[1]
        endpos.orientation.z = quaternion[2]
        endpos.orientation.w = quaternion[3]
        self.end_pose_pub.publish(endpos)

    def query_motion_limits(self):
        """Refresh SDK limit caches without blocking their 1 Hz publisher."""
        try:
            self.piper.SearchAllMotorMaxAngleSpd()
            self.piper.SearchAllMotorMaxAccLimit()
            if not getattr(self, 'motion_limits_first_query_logged', False):
                self.motion_limits_first_query_logged = True
                self.get_logger().info(
                    'Independent controller-limit CAN query timer is active')
        except Exception as error:
            self.get_logger().warn(
                'Controller motion-limit query failed: %s' % error)

    def publish_motion_limits(self):
        """Publish fresh controller-reported J1-J6 velocity/acceleration limits."""
        if not self.motion_limits_timer_started:
            self.motion_limits_timer_started = True
            self.get_logger().info(
                'Independent 1 Hz controller-limit publisher is active')
        maximum_age = (
            self.get_parameter('motion_limit_max_age_sec')
            .get_parameter_value().double_value
        )
        limits = controller_motion_limits(
            self.piper, maximum_age_sec=maximum_age)
        if not getattr(self, 'motion_limits_first_result_logged', False):
            self.motion_limits_first_result_logged = True
            self.get_logger().info(
                'First controller-limit cache result: valid=%s reason=%s'
                % (limits['valid'], limits['reason']))
        try:
            message = PiperMotionLimits()
            message.header.stamp = self.get_clock().now().to_msg()
            message.joint_names = list(MOTION_LIMIT_JOINT_NAMES)
            message.max_velocity_rad_s = limits['velocities']
            message.max_acceleration_rad_s2 = limits['accelerations']
            message.valid = bool(limits['valid'])
            message.limits_sha256 = limits['limits_sha256']
            message.source = 'piper_sdk_controller_feedback'
            message.reason = limits['reason']
            self.motion_limits_pub.publish(message)
            if not getattr(self, 'motion_limits_first_publish_logged', False):
                self.motion_limits_first_publish_logged = True
                self.get_logger().info(
                    'First controller-limit message published: valid=%s hash=%s'
                    % (message.valid, message.limits_sha256))
        except Exception as error:
            self.get_logger().error(
                'Controller-limit message publication failed: %r' % error)

    def pos_callback(self, pos_data):
        """Callback function for subscribing to the end effector pose

        Args:
            pos_data (): The position data
        """
        factor = 180 / 3.1415926
        self.get_logger().info(f"Received PosCmd:")
        self.get_logger().info(f"x: {pos_data.x}")
        self.get_logger().info(f"y: {pos_data.y}")
        self.get_logger().info(f"z: {pos_data.z}")
        self.get_logger().info(f"roll: {pos_data.roll}")
        self.get_logger().info(f"pitch: {pos_data.pitch}")
        self.get_logger().info(f"yaw: {pos_data.yaw}")
        self.get_logger().info(f"gripper: {pos_data.gripper}")
        self.get_logger().info(f"mode1: {pos_data.mode1}")
        self.get_logger().info(f"mode2: {pos_data.mode2}")
        x = round(pos_data.x*1000) * 1000
        y = round(pos_data.y*1000) * 1000
        z = round(pos_data.z*1000) * 1000
        rx = round(pos_data.roll*1000*factor)
        ry = round(pos_data.pitch*1000*factor)
        rz = round(pos_data.yaw*1000*factor)
        if(self.GetEnableFlag()):
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.send_motion_ctrl_2_if_changed(0x01, 0x00, 50)
            self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
            gripper = round(pos_data.gripper * 1000 * 1000)
            if pos_data.gripper > 80000:
                gripper = 80000
            if pos_data.gripper < 0:
                gripper = 0
            if self.gripper_exist:
                self.send_gripper_if_changed(abs(gripper), 1000, 0x01, 0)

    def joint_callback(self, joint_data):
        """Callback function for joint angles

        Args:
            joint_data (): The joint data
        """
        factor = 57324.840764  # 1000*180/3.14
        self.get_logger().debug("Received JointState command")
        if len(joint_data.position) < 6:
            self.get_logger().warn("Ignoring JointState command with fewer than 6 arm joints")
            return

        arm_joint_0 = self.enforce_joint_bound('joint1', self.get_joint_value(joint_data, 'joint1', 0))
        arm_joint_1 = self.enforce_joint_bound('joint2', self.get_joint_value(joint_data, 'joint2', 1))
        arm_joint_2 = self.enforce_joint_bound('joint3', self.get_joint_value(joint_data, 'joint3', 2))
        arm_joint_3 = self.enforce_joint_bound('joint4', self.get_joint_value(joint_data, 'joint4', 3))
        arm_joint_4 = self.enforce_joint_bound('joint5', self.get_joint_value(joint_data, 'joint5', 4))
        arm_joint_5 = self.enforce_joint_bound('joint6', self.get_joint_value(joint_data, 'joint6', 5))

        self.get_logger().debug(
            "arm joints: %.6f %.6f %.6f %.6f %.6f %.6f"
            % (
                arm_joint_0, arm_joint_1, arm_joint_2,
                arm_joint_3, arm_joint_4, arm_joint_5,
            ))

        joint_0 = round(arm_joint_0*factor)
        joint_1 = round(arm_joint_1*factor)
        joint_2 = round(arm_joint_2*factor)
        joint_3 = round(arm_joint_3*factor)
        joint_4 = round(arm_joint_4*factor)
        joint_5 = round(arm_joint_5*factor)
        if(len(joint_data.position) >= 7):
            gripper_joint = self.enforce_joint_bound('joint7', self.get_joint_value(joint_data, 'joint7', 6))
            self.get_logger().debug(f"gripper: {gripper_joint}")
            joint_6 = round(gripper_joint*1000*1000)
            joint_6 = int(clip(joint_6, 0, 80000))
        else: joint_6 = 0
        if(self.GetEnableFlag()):
            # 设定电机速度
            if(joint_data.velocity != []):
                all_zeros = all(v == 0 for v in joint_data.velocity)
            else:
                all_zeros = True
            if not all_zeros:
                lens = len(joint_data.velocity)
                if lens >= 7:
                    vel_all = int(clip(round(self.get_joint_velocity(joint_data, 'joint7', 6)), 0, 100))
                    self.get_logger().debug(f"vel_all: {vel_all}")
                    self.send_motion_ctrl_2_if_changed(0x01, 0x01, vel_all)
                else:
                    self.send_motion_ctrl_2_if_changed(0x01, 0x01, 30)
            else:
                self.send_motion_ctrl_2_if_changed(0x01, 0x01, 30)
            self.piper.JointCtrl(joint_0, joint_1, joint_2,
                                 joint_3, joint_4, joint_5)
            # A six-position JointState is an explicit arm-only MoveJ command.
            # Do not invent a zero gripper target: automation uses this form so
            # scan waypoints cannot clip, warn about, or actuate the gripper.
            if self.gripper_exist and len(joint_data.position) >= 7:
                if len(joint_data.effort) >= 7:
                    gripper_effort = float(clip(self.get_joint_effort(joint_data, 'joint7', 6), 0.5, 3))
                    self.get_logger().debug(f"gripper_effort: {gripper_effort}")
                    gripper_effort = round(gripper_effort * 1000)
                    self.send_gripper_if_changed(
                        abs(joint_6), gripper_effort, 0x01, 0)
                else:
                    self.send_gripper_if_changed(
                        abs(joint_6), 1000, 0x01, 0)

    def enable_callback(self, enable_flag: Bool):
        """Callback function for enabling the robotic arm

        Args:
            enable_flag (): Boolean flag
        """
        self.get_logger().info(f"Received enable flag:")
        self.get_logger().info(f"enable_flag: {enable_flag.data}")
        self.reset_command_cache()
        if enable_flag.data:
            self.__enable_flag = True
            self.piper.EnableArm(7)
            if self.gripper_exist:
                self.send_gripper_if_changed(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            if self.gripper_exist:
                self.send_gripper_if_changed(0, 1000, 0x00, 0)

    def handle_enable_service(self, req, resp):
        """Handle enable service for the robotic arm"""
        self.get_logger().info(f"Received request: {req.enable_request}")
        succeeded = request_piper_enable_state(
            self.piper,
            bool(req.enable_request),
            self.enable_timeout,
        )
        self.reset_command_cache()
        if succeeded:
            if req.enable_request:
                self.send_gripper_if_changed(0, 1000, 0x01, 0)
            else:
                self.send_gripper_if_changed(0, 1000, 0x02, 0)
        else:
            self.get_logger().error(
                f"Timed out waiting for motors to {'enable' if req.enable_request else 'disable'}"
            )

        self.__enable_flag = bool(succeeded and req.enable_request)
        resp.enable_response = bool(succeeded)
        self.get_logger().info(f"Returning response: {resp.enable_response}")
        return resp


def main(args=None):
    rclpy.init(args=args)
    piper_single_node = PiperRosNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(piper_single_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        piper_single_node.destroy_node()
        rclpy.shutdown()
