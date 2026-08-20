from pathlib import Path
import math
import threading
from types import SimpleNamespace

import pytest

from piper.piper_ctrl_single_node import (
    controller_motion_limits,
    DEFAULT_JOINT_BOUNDS,
    format_gripper_feedback_diagnostic,
    gripper_feedback_diagnostic,
    JOINT6_LIMIT_RAD,
    JOINT6_STARTUP_CONTROLLER_MAX_DEG,
    JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG,
    motor_driver_faults,
    motion_limits_sha256,
    PiperRosNode,
    motor_driver_enable_states,
    qualify_startup_joint6_controller_limit,
    reset_startup_joint6_transaction,
    request_piper_enable_state,
)


def test_joint6_fallback_bound_is_standard_signed_pi():
    assert math.isclose(JOINT6_LIMIT_RAD, math.pi)
    assert DEFAULT_JOINT_BOUNDS['joint6'] == (
        -JOINT6_LIMIT_RAD, JOINT6_LIMIT_RAD)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeControllerLimitPiper:
    def __init__(self, reported_max_tenths=3100, apply_setting=True):
        self.reported_max_tenths = reported_max_tenths
        self.apply_setting = apply_setting
        self.set_calls = []
        self.query_calls = []

    def MotorAngleLimitMaxSpdSet(
            self, motor_num, max_angle_limit, min_angle_limit, max_joint_spd):
        self.set_calls.append((
            motor_num, max_angle_limit, min_angle_limit, max_joint_spd))
        if self.apply_setting:
            self.reported_max_tenths = max_angle_limit

    def SearchMotorMaxAngleSpdAccLimit(self, motor_num, content):
        self.query_calls.append((motor_num, content))

    def GetAllMotorAngleLimitMaxSpd(self):
        motors = [None] * 7
        motors[6] = SimpleNamespace(
            motor_num=6,
            max_angle_limit=self.reported_max_tenths,
        )
        return SimpleNamespace(
            all_motor_angle_limit_max_spd=SimpleNamespace(motor=motors))


def test_positive_only_startup_controller_limit_is_set_and_proved():
    piper = FakeControllerLimitPiper()
    succeeded, reason = qualify_startup_joint6_controller_limit(piper)
    assert succeeded is True
    assert '545.0 deg' in reason
    assert piper.set_calls == [(6, 5450, 0x7FFF, 0x7FFF)]
    assert piper.query_calls == [(6, 0x01)]


def test_controller_limit_covers_full_logical_range_after_startup_turn():
    assert JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG == 540.0
    assert JOINT6_STARTUP_CONTROLLER_MAX_DEG > \
        JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG
    assert math.degrees(2.0 * math.pi + JOINT6_LIMIT_RAD) == \
        JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG


def test_positive_only_startup_controller_limit_fails_closed_at_310_deg():
    piper = FakeControllerLimitPiper(apply_setting=False)
    clock = FakeClock()
    succeeded, reason = qualify_startup_joint6_controller_limit(
        piper,
        timeout_sec=0.1,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )
    assert succeeded is False
    assert 'got 310.0 deg' in reason


class FakePiper:
    def __init__(self, enable_results=(), disable_results=()):
        self.enable_results = iter(enable_results)
        self.disable_results = iter(disable_results)
        self.enable_calls = 0
        self.disable_calls = 0
        self.states = [False] * 6
        self.faults = [set() for _ in range(6)]

    def EnablePiper(self):
        self.enable_calls += 1
        result = next(self.enable_results, False)
        self.states = [bool(result)] * 6
        return result

    def DisablePiper(self):
        self.disable_calls += 1
        result = next(self.disable_results, True)
        self.states = [bool(result)] * 6
        return result

    def GetArmLowSpdInfoMsgs(self):
        return SimpleNamespace(**{
            'motor_%d' % index: SimpleNamespace(
                foc_status=SimpleNamespace(
                    driver_enable_status=self.states[index - 1],
                    **{
                        field: field in self.faults[index - 1]
                        for field in (
                            'voltage_too_low', 'motor_overheating',
                            'driver_overcurrent', 'driver_overheating',
                            'collision_status', 'driver_error_status',
                            'stall_status')
                    }))
            for index in range(1, 7)
        })


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


class FakeGripperFeedbackPiper:
    def __init__(self, timestamp=100.0, position_raw=40000,
                 effort_raw=1000, enabled=True, homed=True, faults=()):
        self.wrapper = SimpleNamespace(
            time_stamp=timestamp,
            Hz=200.0,
            gripper_state=SimpleNamespace(
                grippers_angle=position_raw,
                grippers_effort=effort_raw,
                foc_status=SimpleNamespace(
                    driver_enable_status=enabled,
                    homing_status=homed,
                    **{
                        field: field in faults
                        for field in (
                            'voltage_too_low', 'motor_overheating',
                            'driver_overcurrent', 'driver_overheating',
                            'sensor_status', 'driver_error_status')
                    },
                ),
            ),
        )

    def GetArmGripperMsgs(self):
        return self.wrapper


class FakeCommandNode:
    def __init__(self):
        self.piper = FakeCommandPiper()
        self._command_cache_lock = threading.Lock()
        self._motion_ctrl_2_signature = None
        self._gripper_command_signature = None


def test_gripper_diagnostic_reports_feedback_state_faults_and_units():
    piper = FakeGripperFeedbackPiper(
        faults=('voltage_too_low', 'driver_error_status'))

    diagnostic = gripper_feedback_diagnostic(
        piper, wall_time_fn=lambda: 100.25)

    assert diagnostic == {
        'available': True,
        'timestamp': 100.0,
        'age_sec': 0.25,
        'hz': 200.0,
        'position_raw': 40000,
        'effort_raw': 1000,
        'enabled': True,
        'homed': True,
        'faults': ('voltage_too_low', 'driver_error_status'),
    }
    formatted = format_gripper_feedback_diagnostic(diagnostic)
    assert 'position=40000(40.000mm)' in formatted
    assert 'effort=1000(1.000Nm)' in formatted
    assert 'enabled=True' in formatted
    assert 'faults=voltage_too_low,driver_error_status' in formatted


def test_gripper_diagnostic_does_not_treat_sdk_default_as_live_feedback():
    piper = FakeGripperFeedbackPiper(
        timestamp=0.0, position_raw=0, effort_raw=0,
        enabled=False, homed=False)

    diagnostic = gripper_feedback_diagnostic(
        piper, wall_time_fn=lambda: 100.0)

    assert diagnostic['available'] is False
    assert math.isinf(diagnostic['age_sec'])
    assert format_gripper_feedback_diagnostic(
        diagnostic) == 'feedback=NO_FEEDBACK'


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


def test_partial_enable_feedback_never_counts_as_enabled():
    piper = FakePiper()
    piper.states = [True, True, True, True, False, True]

    assert motor_driver_enable_states(piper) == (
        True, True, True, True, False, True)


def test_motor_faults_identify_the_joint_and_fault_field():
    piper = FakePiper()
    piper.faults[4].update(('collision_status', 'stall_status'))

    assert motor_driver_faults(piper) == (
        'joint5:collision_status', 'joint5:stall_status')


def test_motor_watchdog_disables_every_axis_on_joint5_collision_fault():
    class WatchdogPiper(FakePiper):
        def __init__(self):
            super().__init__()
            self.disable_arm_calls = []

        def DisableArm(self, motor):
            self.disable_arm_calls.append(motor)
            self.states = [False] * 6

    piper = WatchdogPiper()
    piper.states = [True] * 6
    piper.faults[4].add('collision_status')
    errors = []
    node = SimpleNamespace(
        piper=piper,
        _PiperRosNode__enable_flag=True,
        _motor_watchdog_reason='',
        _motor_watchdog_disable_at=0.0,
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(error=errors.append),
    )

    PiperRosNode.fail_closed_motor_watchdog(node)

    assert piper.disable_arm_calls == [7]
    assert node._PiperRosNode__enable_flag is False
    assert node._latest_motor_states == (True, True, True, True, True, True)
    assert node._latest_motor_faults == ('joint5:collision_status',)
    assert errors and 'joint5:collision_status' in errors[0]


def test_motor_watchdog_allows_bounded_fault_free_enable_transition():
    class WatchdogPiper(FakePiper):
        def __init__(self):
            super().__init__()
            self.disable_arm_calls = []

        def DisableArm(self, motor):
            self.disable_arm_calls.append(motor)

    piper = WatchdogPiper()
    piper.states = [True, True, True, False, False, False]
    node = SimpleNamespace(
        piper=piper,
        _PiperRosNode__enable_flag=False,
        _enable_transition_active=True,
        _motor_watchdog_reason='',
        _motor_watchdog_disable_at=0.0,
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    PiperRosNode.fail_closed_motor_watchdog(node)

    assert piper.disable_arm_calls == []
    assert piper.states == [True, True, True, False, False, False]


def test_motor_watchdog_allows_initial_sdk_snapshot_to_cohere():
    class WatchdogPiper(FakePiper):
        def __init__(self):
            super().__init__()
            self.disable_arm_calls = []

        def DisableArm(self, motor):
            self.disable_arm_calls.append(motor)

    piper = WatchdogPiper()
    piper.states = [False, True, True, True, True, True]
    node = SimpleNamespace(
        piper=piper,
        _PiperRosNode__enable_flag=False,
        _disable_required=False,
        _enable_transition_active=False,
        _motor_watchdog_reason='',
        _motor_watchdog_disable_at=0.0,
        _motor_watchdog_started_at=__import__('time').monotonic(),
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    PiperRosNode.fail_closed_motor_watchdog(node)

    assert piper.disable_arm_calls == []
    assert node._latest_motor_states == (
        False, True, True, True, True, True)


def test_motor_watchdog_startup_grace_never_masks_motor_fault():
    class WatchdogPiper(FakePiper):
        def __init__(self):
            super().__init__()
            self.disable_arm_calls = []

        def DisableArm(self, motor):
            self.disable_arm_calls.append(motor)

    piper = WatchdogPiper()
    piper.states = [False, True, True, True, True, True]
    piper.faults[4].add('collision_status')
    node = SimpleNamespace(
        piper=piper,
        _PiperRosNode__enable_flag=False,
        _disable_required=False,
        _enable_transition_active=False,
        _motor_watchdog_reason='',
        _motor_watchdog_disable_at=0.0,
        _motor_watchdog_started_at=__import__('time').monotonic(),
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    PiperRosNode.fail_closed_motor_watchdog(node)

    assert piper.disable_arm_calls == [7]


def test_motor_watchdog_retries_latched_all_axis_disable():
    class WatchdogPiper(FakePiper):
        def __init__(self):
            super().__init__()
            self.disable_arm_calls = []

        def DisableArm(self, motor):
            self.disable_arm_calls.append(motor)

    piper = WatchdogPiper()
    piper.states = [True] * 6
    node = SimpleNamespace(
        piper=piper,
        _PiperRosNode__enable_flag=True,
        _disable_required=True,
        _enable_transition_active=False,
        _motor_watchdog_reason='',
        _motor_watchdog_disable_at=0.0,
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    PiperRosNode.fail_closed_motor_watchdog(node)

    assert piper.disable_arm_calls == [7]
    assert node._disable_required is True
    assert node._PiperRosNode__enable_flag is False


def test_failed_partial_enable_service_rolls_every_axis_back_to_disabled():
    class FaultedEnablePiper(FakePiper):
        def EnablePiper(self):
            self.enable_calls += 1
            self.states = [True, True, True, True, False, True]
            self.faults[4].add('collision_status')
            return False

        def DisablePiper(self):
            self.disable_calls += 1
            self.states = [False] * 6
            return False

    messages = []
    piper = FaultedEnablePiper()
    node = SimpleNamespace(
        piper=piper,
        enable_timeout=0.1,
        gripper_exist=False,
        _PiperRosNode__enable_flag=False,
        _disable_required=False,
        _enable_transition_active=False,
        _enable_transition_lock=threading.Lock(),
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(
            info=messages.append,
            error=messages.append,
            fatal=messages.append,
        ),
    )
    request = SimpleNamespace(enable_request=True)
    response = SimpleNamespace(enable_response=None)

    PiperRosNode.handle_enable_service(node, request, response)

    assert response.enable_response is False
    assert piper.states == [False] * 6
    assert piper.disable_calls >= 1
    assert any('rolled back' in message for message in messages)


def test_disable_requires_every_motor_feedback_flag_to_clear():
    class PartialDisablePiper(FakePiper):
        def DisablePiper(self):
            self.disable_calls += 1
            if self.disable_calls == 1:
                self.states = [False, False, False, False, True, False]
                return False
            self.states = [False] * 6
            return False

    clock = FakeClock()
    piper = PartialDisablePiper()
    piper.states = [True] * 6

    assert request_piper_enable_state(
        piper, False, 1.0, clock.monotonic, clock.sleep)
    assert piper.disable_calls == 2
    assert clock.now == 0.01


def test_proved_disable_clears_incomplete_j6_startup_watchdog_state():
    piper = FakePiper(disable_results=[False])
    piper.states = [True] * 6
    node = SimpleNamespace(
        piper=piper,
        enable_timeout=0.1,
        gripper_exist=False,
        _PiperRosNode__enable_flag=True,
        _disable_required=False,
        _enable_transition_active=False,
        _enable_transition_lock=threading.Lock(),
        _startup_joint6_active=True,
        _startup_joint6_armed=True,
        _startup_joint6_last_target=-3.08,
        _startup_joint6_last_controller_target=3.2,
        _startup_joint6_direction_previous_raw=3.64,
        _startup_joint6_direction_unwrapped=3.64,
        _startup_joint6_direction_high_water=3.64,
        _continuous_joint6_feedback=-2.64,
        reset_command_cache=lambda: None,
        send_gripper_if_changed=lambda *_args: None,
        get_logger=lambda: SimpleNamespace(
            info=lambda _message: None,
            error=lambda _message: None,
            fatal=lambda _message: None,
        ),
    )
    request = SimpleNamespace(enable_request=False)
    response = SimpleNamespace(enable_response=None)

    PiperRosNode.handle_enable_service(node, request, response)

    assert response.enable_response is True
    assert piper.states == [False] * 6
    assert node._startup_joint6_active is False
    assert node._startup_joint6_armed is False
    assert node._startup_joint6_direction_previous_raw is None
    assert node._startup_joint6_direction_high_water is None
    assert node._continuous_joint6_feedback is None


def test_startup_reset_allows_next_feedback_generation_to_rearm_cleanly():
    node = SimpleNamespace(
        _startup_joint6_active=True,
        _startup_joint6_armed=True,
        _startup_joint6_last_target=-3.08,
        _startup_joint6_last_controller_target=3.2,
        _startup_joint6_direction_previous_raw=3.64,
        _startup_joint6_direction_unwrapped=3.64,
        _startup_joint6_direction_high_water=3.64,
        _continuous_joint6_feedback=-2.64,
    )

    reset_startup_joint6_transaction(node)

    assert node._startup_joint6_active is False
    assert node._startup_joint6_armed is False
    assert node._startup_joint6_last_target is None
    assert node._startup_joint6_last_controller_target is None
    assert node._startup_joint6_direction_previous_raw is None
    assert node._startup_joint6_direction_unwrapped is None
    assert node._startup_joint6_direction_high_water is None
    assert node._continuous_joint6_feedback is None


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


def test_startup_joint6_callback_never_commands_negative_before_raw_wrap():
    class StartupNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._joint_feedback_lock = threading.Lock()
            self.last_commanded_joint_positions = None
            self.last_command_feedback_best_error = None
            self.last_joint_commanded_at = 0.0
            self.last_joint_feedback_positions = None
            self._startup_joint6_finished = False
            self._startup_joint6_armed = True
            self._startup_joint6_active = False
            self._startup_joint6_last_target = None
            self._raw_joint6_feedback = 3.011637
            self._published_joint6_feedback = 3.011637 - 2.0 * math.pi
            self._latest_raw_arm_positions = (
                0.07, -0.048633872, 0.025625236, 0.055, 0.348, 3.011637)
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=lambda *_args: None,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = StartupNode()
    command = SimpleNamespace(
        header=SimpleNamespace(
            frame_id='piper_scan_executor_startup_wrist'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.07, -0.048633872, 0.025625236, 0.055, 0.348, 0.0],
        velocity=[0.0] * 6,
        effort=[],
    )

    untagged = SimpleNamespace(
        header=SimpleNamespace(frame_id='piper_scan_executor_sdk_movej'),
        name=list(command.name),
        position=list(command.position),
        velocity=list(command.velocity),
        effort=[],
    )
    PiperRosNode.joint_callback(node, untagged)
    assert node.piper.joint_commands == []

    PiperRosNode.joint_callback(node, command)
    before_wrap = node.piper.joint_commands[-1][5]
    assert before_wrap > 180000
    factor = 57324.840764
    # Reproduce the live failure: J2/J3 are outside the controller's ordinary
    # powered command range after gravity relaxation.  STARTUP_WRIST must keep
    # their newest coherent feedback instead of clipping them to zero/bounds.
    assert node.piper.joint_commands[-1][1] == round(-0.048633872 * factor)
    assert node.piper.joint_commands[-1][2] == round(0.025625236 * factor)

    node._raw_joint6_feedback = -3.13
    node._published_joint6_feedback = -3.13
    PiperRosNode.joint_callback(node, command)
    after_wrap = node.piper.joint_commands[-1][5]
    assert after_wrap == 0
    assert after_wrap > round(-3.13 * 57324.840764)


def test_startup_wrap_bridge_continues_positive_to_raw_two_pi():
    class StartupNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._startup_joint6_finished = False
            self._startup_joint6_armed = True
            self._startup_joint6_active = True
            self._startup_joint6_last_target = 0.0
            self._raw_joint6_feedback = 3.200695
            self._published_joint6_feedback = 3.200695 - 2.0 * math.pi
            self._latest_raw_arm_positions = (
                0.070770308, 0.0, 0.0, 0.055977796, 0.348164796,
                3.200695)
            self.errors = []
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=self.errors.append,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = StartupNode()
    command = SimpleNamespace(
        header=SimpleNamespace(
            frame_id='piper_scan_executor_startup_wrist'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.070770308, 0.0, 0.0, 0.055977796, 0.348164796, 0.0],
        velocity=[0.0] * 6,
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)

    assert node.errors == []
    assert len(node.piper.joint_commands) == 1
    # Once +3.2 is measured, logical ready zero must remain on the same
    # increasing controller branch. A numeric zero command here rotates the
    # physical wrist anticlockwise and is forbidden.
    ready_command = node.piper.joint_commands[-1][5]
    measured_command = round(node._raw_joint6_feedback * 57324.840764)
    assert ready_command == round((2.0 * math.pi) * 57324.840764)
    assert ready_command > measured_command

    node.piper.joint_commands.clear()
    node._startup_joint6_last_target = 3.2 - 2.0 * math.pi
    command.position[-1] = 3.2 - 2.0 * math.pi
    PiperRosNode.joint_callback(node, command)

    assert len(node.piper.joint_commands) == 1
    # A repeated bridge endpoint retains measured raw J6 and never reverses
    # to correct the 0.000695-rad overshoot.
    assert node.piper.joint_commands[-1][5] == round(
        node._raw_joint6_feedback * 57324.840764)


def test_startup_joint6_explicit_hold_uses_measured_raw_coordinate():
    class StartupNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._startup_joint6_finished = False
            self._startup_joint6_armed = True
            self._startup_joint6_active = False
            self._startup_joint6_last_target = None
            self._raw_joint6_feedback = 3.071260
            self._published_joint6_feedback = 3.071260 - 2.0 * math.pi
            self._latest_raw_arm_positions = (
                0.0, -0.048337, 0.012455, 0.0, 0.401072, 3.071260)
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=lambda *_args: None,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = StartupNode()
    # Reproduce the failed run's asynchronous executor snapshot: its logical
    # J6 is 0.016 rad behind the driver's current sample. The explicit hold
    # must emit the exact measured raw coordinate, not rotate either way.
    command = SimpleNamespace(
        header=SimpleNamespace(frame_id='piper_scan_executor_hold'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.0, 0.0, 0.0, 0.0, 0.4, -3.228165],
        velocity=[0.0] * 6,
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)

    assert len(node.piper.joint_commands) == 1
    expected = round(3.071260 * 57324.840764)
    assert node.piper.joint_commands[-1][5] == expected
    assert node._startup_joint6_armed
    assert not node._startup_joint6_active
    assert node._startup_joint6_last_target is None


def test_commissioning_can_hold_startup_j6_while_moving_other_joints():
    class CommissioningNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._startup_joint6_finished = False
            self._startup_joint6_armed = True
            self._startup_joint6_active = False
            self._startup_joint6_last_target = None
            self._raw_joint6_feedback = 3.071260
            self._published_joint6_feedback = 3.071260 - 2.0 * math.pi
            self._latest_raw_arm_positions = (
                0.0, 0.0, 0.0, 0.0, 0.4, 3.071260)
            self.errors = []
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=self.errors.append,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = CommissioningNode()
    measured_logical_j6 = node._published_joint6_feedback
    command = SimpleNamespace(
        header=SimpleNamespace(frame_id='piper_native_gui'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.2, 0.1, -0.1, 0.15, 0.3, measured_logical_j6],
        velocity=[0.0] * 5 + [5.0],
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)

    factor = 57324.840764
    assert node.errors == []
    assert len(node.piper.joint_commands) == 1
    assert node.piper.joint_commands[-1][0] == round(0.2 * factor)
    assert node.piper.joint_commands[-1][3] == round(0.15 * factor)
    assert node.piper.joint_commands[-1][5] == round(3.071260 * factor)


def test_commissioning_startup_j6_is_positive_only_and_j6_only():
    class CommissioningNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._startup_joint6_finished = False
            self._startup_joint6_armed = True
            self._startup_joint6_active = False
            self._startup_joint6_last_target = None
            self._startup_joint6_last_controller_target = None
            self._raw_joint6_feedback = 3.071260
            self._published_joint6_feedback = 3.071260 - 2.0 * math.pi
            self._latest_raw_arm_positions = (
                0.01, 0.02, -0.03, 0.04, 0.4, 3.071260)
            self.errors = []
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=self.errors.append,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = CommissioningNode()
    command = SimpleNamespace(
        header=SimpleNamespace(frame_id='piper_native_gui'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.5, 0.5, -0.5, 0.5, 0.5, 0.0],
        velocity=[0.0] * 5 + [5.0],
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)

    factor = 57324.840764
    assert node.errors == []
    assert len(node.piper.joint_commands) == 1
    assert node.piper.joint_commands[-1][:5] == tuple(
        round(value * factor) for value in node._latest_raw_arm_positions[:5])
    assert node.piper.joint_commands[-1][5] == round(
        (2.0 * math.pi) * factor)

    node.piper.joint_commands.clear()
    command.position[5] = node._published_joint6_feedback - 0.1
    PiperRosNode.joint_callback(node, command)
    assert node.piper.joint_commands == []
    assert 'positive direction toward ready zero' in node.errors[-1]
    assert node._startup_joint6_armed
    assert node._startup_joint6_active
    assert node._startup_joint6_last_target == 0.0


def test_normal_joint6_command_retains_completed_startup_turn_offset():
    class ReadyNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._startup_joint6_finished = True
            self._startup_joint6_armed = False
            self._startup_joint6_active = False
            self._joint6_controller_turn_offset = 2.0 * math.pi
            self.startup_joint6_controller_max_deg = \
                JOINT6_STARTUP_CONTROLLER_MAX_DEG
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=lambda *_args: None,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = ReadyNode()
    command = SimpleNamespace(
        header=SimpleNamespace(frame_id='piper_scan_executor_sdk_movej'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        velocity=[0.0] * 6,
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)
    ready_raw = node.piper.joint_commands[-1][5]
    assert ready_raw == round((2.0 * math.pi) * 57324.840764)

    command.position[-1] = -3.139536232
    PiperRosNode.joint_callback(node, command)
    storage_raw = node.piper.joint_commands[-1][5]
    assert storage_raw == round(
        (2.0 * math.pi - 3.139536232) * 57324.840764)
    assert storage_raw < ready_raw

    command.position[-1] = 2.483481659203857
    PiperRosNode.joint_callback(node, command)
    positive_scan_raw = node.piper.joint_commands[-1][5]
    assert positive_scan_raw == round(
        (2.0 * math.pi + 2.483481659203857) * 57324.840764)
    assert positive_scan_raw < round(
        math.radians(JOINT6_STARTUP_CONTROLLER_MAX_DEG) * 57324.840764)

    command.position[-1] = math.pi
    PiperRosNode.joint_callback(node, command)
    positive_endpoint_raw = node.piper.joint_commands[-1][5]
    assert positive_endpoint_raw == round(3.0 * math.pi * 57324.840764)


def test_normal_joint6_command_rejects_unqualified_controller_coordinate():
    class ReadyNode(FakeCommandNode):
        enforce_joint_bound = PiperRosNode.enforce_joint_bound
        get_joint_value = PiperRosNode.get_joint_value
        get_joint_velocity = PiperRosNode.get_joint_velocity
        get_joint_effort = PiperRosNode.get_joint_effort
        send_motion_ctrl_2_if_changed = PiperRosNode.send_motion_ctrl_2_if_changed
        send_gripper_if_changed = PiperRosNode.send_gripper_if_changed

        def __init__(self):
            super().__init__()
            self.joint_bounds = dict(DEFAULT_JOINT_BOUNDS)
            self.gripper_exist = False
            self._startup_joint6_finished = True
            self._startup_joint6_armed = False
            self._startup_joint6_active = False
            self._joint6_controller_turn_offset = 2.0 * math.pi
            self.startup_joint6_controller_max_deg = 365.0
            self.errors = []
            self.logger = SimpleNamespace(
                debug=lambda *_args: None,
                info=lambda *_args: None,
                warn=lambda *_args: None,
                error=self.errors.append,
            )

        def GetEnableFlag(self):
            return True

        def get_logger(self):
            return self.logger

    node = ReadyNode()
    command = SimpleNamespace(
        header=SimpleNamespace(frame_id='piper_scan_executor_sdk_movej'),
        name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        position=[0.0, 0.0, 0.0, 0.0, 0.4, 2.483481659203857],
        velocity=[0.0] * 6,
        effort=[],
    )

    PiperRosNode.joint_callback(node, command)
    assert node.piper.joint_commands == []
    assert 'above the qualified 365.0-deg positive limit' in node.errors[-1]


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
