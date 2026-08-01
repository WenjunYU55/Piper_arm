from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from piper.piper_ctrl_single_node import (
    controller_motion_limits,
    motion_limits_sha256,
    PiperRosNode,
    request_piper_enable_state,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakePiper:
    def __init__(self, enable_results=(), disable_results=()):
        self.enable_results = iter(enable_results)
        self.disable_results = iter(disable_results)
        self.enable_calls = 0
        self.disable_calls = 0

    def EnablePiper(self):
        self.enable_calls += 1
        return next(self.enable_results, False)

    def DisablePiper(self):
        self.disable_calls += 1
        return next(self.disable_results, True)


class FakeCommandPiper:
    def __init__(self):
        self.motion_commands = []
        self.gripper_commands = []
        self.joint_commands = []

    def MotionCtrl_2(self, *command):
        self.motion_commands.append(command)

    def GripperCtrl(self, *command):
        self.gripper_commands.append(command)

    def JointCtrl(self, *command):
        self.joint_commands.append(command)


class FakeCommandNode:
    def __init__(self):
        self.piper = FakeCommandPiper()
        self._command_cache_lock = threading.Lock()
        self._motion_ctrl_2_signature = None
        self._gripper_command_signature = None


def test_removed_driver_compatibility_inputs_do_not_return():
    package_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[4]
    driver = (package_root / 'piper' / 'piper_ctrl_single_node.py').read_text()
    launch = (package_root / 'launch' / 'start_single_piper.launch.py').read_text()
    startup = (repo_root / 'start_piper.sh').read_text()

    combined = '\n'.join((driver, launch, startup))
    assert 'girpper_exist' not in combined
    assert 'rviz_ctrl_flag' not in combined
    assert "default_value='false'" in launch
    assert not (package_root / 'launch' / 'start_single_piper_rviz.launch.py').exists()


def test_feedback_publication_uses_executor_timer_not_unmanaged_thread():
    package_root = Path(__file__).resolve().parents[1]
    driver = (package_root / 'piper' / 'piper_ctrl_single_node.py').read_text()
    initialization = driver.split(
        'class PiperRosNode(Node):', 1)[1].split(
            '    def GetEnableFlag(self):', 1)[0]
    publish_feedback = driver.split(
        '    def publish_feedback(self):', 1)[1].split(
            '    def PublishArmState(self):', 1)[0]

    assert 'self.feedback_timer = self.create_timer(' in initialization
    assert 'target=self.publish_thread' not in initialization
    assert 'self.PublishArmState()' in publish_feedback
    assert 'self.PublishArmJointAndGripper()' in publish_feedback
    assert 'self.PublishArmEndPose()' in publish_feedback


def test_enable_retries_until_feedback_confirms_all_motors():
    clock = FakeClock()
    piper = FakePiper(enable_results=[False, False, True])

    assert request_piper_enable_state(
        piper, True, 1.0, clock.monotonic, clock.sleep
    )
    assert piper.enable_calls == 3
    assert piper.disable_calls == 0
    assert clock.now == 0.02


def test_disable_retries_until_feedback_confirms_no_enabled_motor():
    clock = FakeClock()
    piper = FakePiper(disable_results=[True, True, False])

    assert request_piper_enable_state(
        piper, False, 1.0, clock.monotonic, clock.sleep
    )
    assert piper.disable_calls == 3
    assert piper.enable_calls == 0
    assert clock.now == 0.02


def test_enable_times_out_without_positive_feedback():
    clock = FakeClock()
    piper = FakePiper(enable_results=[False])

    assert not request_piper_enable_state(
        piper, True, 0.025, clock.monotonic, clock.sleep
    )
    assert piper.enable_calls == 4


def test_repeated_movej_mode_and_gripper_commands_are_cached():
    node = FakeCommandNode()

    assert PiperRosNode.send_motion_ctrl_2_if_changed(node, 1, 1, 50)
    assert not PiperRosNode.send_motion_ctrl_2_if_changed(node, 1, 1, 50)
    assert PiperRosNode.send_motion_ctrl_2_if_changed(node, 1, 1, 75)
    assert node.piper.motion_commands == [(1, 1, 50), (1, 1, 75)]

    assert PiperRosNode.send_gripper_if_changed(node, 1000, 1000, 1, 0)
    assert not PiperRosNode.send_gripper_if_changed(node, 1000, 1000, 1, 0)
    assert PiperRosNode.send_gripper_if_changed(node, 2000, 1000, 1, 0)
    assert node.piper.gripper_commands == [
        (1000, 1000, 1, 0),
        (2000, 1000, 1, 0),
    ]


def test_command_cache_reset_forces_mode_and_gripper_reassertion():
    node = FakeCommandNode()
    PiperRosNode.send_motion_ctrl_2_if_changed(node, 1, 1, 20)
    PiperRosNode.send_gripper_if_changed(node, 0, 1000, 1, 0)

    PiperRosNode.reset_command_cache(node)

    assert PiperRosNode.send_motion_ctrl_2_if_changed(node, 1, 1, 20)
    assert PiperRosNode.send_gripper_if_changed(node, 0, 1000, 1, 0)


def test_arm_only_movej_commands_cannot_actuate_the_gripper():
    class ArmOnlyNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = {
                'joint%d' % index: (-3.0, 3.0) for index in range(1, 7)}
            self.joint_bounds['joint7'] = (0.00035, 0.08)
            self.gripper_exist = True
            self._joint_feedback_lock = threading.Lock()
            self.last_commanded_joint_positions = None
            self.last_command_feedback_best_error = None
            self.last_joint_commanded_at = 0.0
            self.last_joint_feedback_positions = None
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                warn=lambda *_args: None,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = ArmOnlyNode()
    command = SimpleNamespace(
        name=[
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'],
        position=[0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
        velocity=[0.0] * 6 + [5.0],
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)

    assert len(node.piper.joint_commands) == 1
    assert node.piper.motion_commands == [(1, 1, 5)]
    assert node.piper.gripper_commands == []


def test_controller_motion_limits_are_typed_converted_and_hash_bound():
    speeds = [None] + [
        SimpleNamespace(motor_num=index, max_joint_spd=1000 + index)
        for index in range(1, 7)
    ]
    accelerations = [None] + [
        SimpleNamespace(joint_motor_num=index, max_joint_acc=2000 + index)
        for index in range(1, 7)
    ]
    piper = SimpleNamespace(
        GetAllMotorAngleLimitMaxSpd=lambda: SimpleNamespace(
            time_stamp=99.5,
            all_motor_angle_limit_max_spd=SimpleNamespace(motor=speeds)),
        GetAllMotorMaxAccLimit=lambda: SimpleNamespace(
            time_stamp=99.6,
            all_motor_max_acc_limit=SimpleNamespace(motor=accelerations)),
    )
    result = controller_motion_limits(
        piper, now_sec=100.0, maximum_age_sec=1.0)
    assert result['valid']
    assert result['velocities'] == pytest.approx([
        1.001, 1.002, 1.003, 1.004, 1.005, 1.006]
    )
    assert result['accelerations'] == pytest.approx([
        2.001, 2.002, 2.003, 2.004, 2.005, 2.006]
    )
    assert result['limits_sha256'] == motion_limits_sha256(
        result['velocities'], result['accelerations'])


def test_controller_motion_limits_fail_closed_when_stale_or_over_protocol_cap():
    speed_motors = [None] + [
        SimpleNamespace(motor_num=index, max_joint_spd=1000)
        for index in range(1, 7)
    ]
    acceleration_motors = [None] + [
        SimpleNamespace(joint_motor_num=index, max_joint_acc=2000)
        for index in range(1, 7)
    ]
    piper = SimpleNamespace(
        GetAllMotorAngleLimitMaxSpd=lambda: SimpleNamespace(
            time_stamp=90.0,
            all_motor_angle_limit_max_spd=SimpleNamespace(
                motor=speed_motors)),
        GetAllMotorMaxAccLimit=lambda: SimpleNamespace(
            time_stamp=90.0,
            all_motor_max_acc_limit=SimpleNamespace(
                motor=acceleration_motors)),
    )
    stale = controller_motion_limits(
        piper, now_sec=100.0, maximum_age_sec=1.0)
    assert not stale['valid']
    assert stale['velocities'] == [0.0] * 6
    assert stale['accelerations'] == [0.0] * 6
    assert stale['limits_sha256'] == '0' * 64
    speed_motors[6].max_joint_spd = 3001
    assert not controller_motion_limits(
        piper, now_sec=90.1, maximum_age_sec=1.0)['valid']
