#!/usr/bin/env python3
# -*-coding:utf8-*-
# Controls one robotic arm node and its optional gripper.
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
import can
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperMotionLimits, PiperStatusMsg, PosCmd
from piper_msgs.srv import Enable
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R  # For Euler angle to quaternion conversion
from numpy import clip


JOINT6_LIMIT_RAD = math.pi
JOINT6_STARTUP_LIMIT_RAD = math.radians(240.0)
JOINT6_WRAP_RAD = 2.0 * math.pi
JOINT6_STARTUP_READY_TOLERANCE_RAD = 0.03
JOINT6_STARTUP_COMMAND_EPSILON_RAD = 1e-4
JOINT6_STARTUP_WRAP_TARGET_RAD = 3.2
JOINT6_STARTUP_WRAP_SETTLE_TOLERANCE_RAD = 0.005
MOTOR_WATCHDOG_STARTUP_GRACE_SEC = 0.5
JOINT6_STARTUP_DIRECTION_TRIP_RAD = 0.01
# Positive-only startup finishes on the controller's raw +360-degree turn.
# Normal ROS motion must still retain the complete logical [-180, +180]
# interval on that same turn, so the controller coordinate must extend through
# raw +540 degrees.  Keep a five-degree controller-side convergence margin;
# the driver's ordinary logical bound remains exactly [-pi, +pi].
JOINT6_STARTUP_CONTROLLER_MAX_DEG = 545.0
JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG = 540.0
JOINT6_STARTUP_CONTROLLER_LIMIT_TIMEOUT_SEC = 2.0
PIPER_SETTING_UNCHANGED = 0x7FFF
JOINT6_STARTUP_COMMAND_FRAME = 'piper_scan_executor_startup_wrist'
JOINT6_HOLD_COMMAND_FRAME = 'piper_scan_executor_hold'
JOINT6_COMMISSIONING_COMMAND_FRAME = 'piper_native_gui'


DEFAULT_JOINT_BOUNDS = {
    'joint1': (-2.8, 2.8),
    'joint2': (-2.1, 2.1),
    'joint3': (-2.8, 2.8),
    'joint4': (-2.8, 2.8),
    'joint5': (-2.1, 2.1),
    'joint6': (-JOINT6_LIMIT_RAD, JOINT6_LIMIT_RAD),
    'joint7': (0.0, 0.08),
}

ENABLE_RETRY_PERIOD_SEC = 0.01
MOTION_LIMIT_JOINT_NAMES = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
]
MOTOR_DRIVER_FAULT_FIELDS = (
    'voltage_too_low',
    'motor_overheating',
    'driver_overcurrent',
    'driver_overheating',
    'collision_status',
    'driver_error_status',
    'stall_status',
)
GRIPPER_FAULT_FIELDS = (
    'voltage_too_low',
    'motor_overheating',
    'driver_overcurrent',
    'driver_overheating',
    'sensor_status',
    'driver_error_status',
)
GRIPPER_COMMAND_MODES = {
    0x00: 'DISABLE',
    0x01: 'ENABLE',
    0x02: 'DISABLE_CLEAR_ERROR',
    0x03: 'ENABLE_CLEAR_ERROR',
}
MAX_PROTOCOL_JOINT_SPEED_RAD_S = 3.0
MAX_PROTOCOL_JOINT_ACCELERATION_RAD_S2 = 5.0
JOINT_FEEDBACK_CAN_IDS = (0x2A5, 0x2A6, 0x2A7)
JOINT_FEEDBACK_CAN_INDEX = {
    0x2A5: (0, 1),
    0x2A6: (2, 3),
    0x2A7: (4, 5),
}
JOINT_FEEDBACK_RAW_TO_RAD = 0.017444 / 1000.0
JOINT_FEEDBACK_CAN_MAX_AGE_SEC = 0.1
JOINT_FEEDBACK_CAN_MAX_SKEW_SEC = 0.03
JOINT_FEEDBACK_WARNING_GAP_SEC = 0.25
CONTROLLER_COMMAND_BOUNDS = {
    'joint2': (0.0, math.pi),
    'joint3': (-2.967, 0.0),
}


def controller_command_position(joint_name, value):
    """Clamp gravity-droop axes to the controller's powered range."""
    result = float(value)
    bounds = CONTROLLER_COMMAND_BOUNDS.get(str(joint_name))
    if bounds is None:
        return result
    return min(float(bounds[1]), max(float(bounds[0]), result))


def continuous_joint6_feedback(raw_value, previous_value=None):
    """
    Return the in-range J6 equivalent continuous with prior feedback.

    Controller feedback wraps at +/-pi.  The ordinary configured J6 range is
    exactly that interval, but the startup transaction may represent its saved
    storage pose down to -240 degrees while crossing the signed boundary in the
    positive physical direction.  The first ambiguous sample therefore
    selects its negative logical equivalent.  Later samples select the
    equivalent closest to the last publication so the startup observation is
    continuous through the wrap.
    """
    raw = float(raw_value)
    if not math.isfinite(raw):
        return raw
    candidates = [
        candidate
        for candidate in (
            raw - JOINT6_WRAP_RAD, raw, raw + JOINT6_WRAP_RAD)
        if (
            -JOINT6_STARTUP_LIMIT_RAD - 1e-12
            <= candidate
            <= JOINT6_STARTUP_LIMIT_RAD + 1e-12
        )
    ]
    if not candidates:
        return raw
    if previous_value is None or not math.isfinite(float(previous_value)):
        negative = [candidate for candidate in candidates if candidate < 0.0]
        if negative:
            return min(negative, key=lambda candidate: abs(candidate - raw))
        return min(candidates, key=lambda candidate: abs(candidate - raw))
    previous = float(previous_value)
    return min(
        candidates,
        key=lambda candidate: (abs(candidate - previous), candidate > 0.0),
    )


def standard_joint6_feedback(raw_value):
    """Return J6 in the ordinary closed [-pi, +pi] interval."""
    raw = float(raw_value)
    if not math.isfinite(raw):
        return raw
    wrapped = (raw + math.pi) % JOINT6_WRAP_RAD - math.pi
    if wrapped == -math.pi and raw > 0.0:
        return math.pi
    return wrapped


def startup_joint6_direction_update(
        raw_feedback, previous_raw_feedback, unwrapped_feedback,
        high_water_feedback,
        trip_tolerance=JOINT6_STARTUP_DIRECTION_TRIP_RAD):
    """Track physical J6 direction across controller-coordinate wraps."""
    raw = float(raw_feedback)
    if previous_raw_feedback is None:
        return raw, raw, False
    previous = float(previous_raw_feedback)
    continuous = float(unwrapped_feedback)
    high_water = float(high_water_feedback)
    delta = raw - previous
    if delta < -math.pi:
        delta += JOINT6_WRAP_RAD
    elif delta > math.pi:
        delta -= JOINT6_WRAP_RAD
    continuous += delta
    high_water = max(high_water, continuous)
    wrong_direction = bool(
        continuous < high_water - float(trip_tolerance))
    return continuous, high_water, wrong_direction


def startup_joint6_controller_target(
        raw_feedback, canonical_feedback, requested_target,
        previous_target=None):
    """
    Map one increasing startup-only J6 target to the controller wrap.

    A target below -pi is represented on the controller's positive side while
    raw feedback is positive.  The executor first requests the logical
    equivalent of raw +3.2 rad, then logical ready zero.  Ready zero must remain
    on the same increasing controller branch and therefore maps to raw +2*pi;
    sending numeric controller zero after +3.2 causes forbidden anticlockwise
    motion.  Every logical target must be nondecreasing, and the driver retains
    the resulting controller-turn offset after startup completes.
    """
    raw = float(raw_feedback)
    canonical = float(canonical_feedback)
    target = float(requested_target)
    if not all(math.isfinite(value) for value in (raw, canonical, target)):
        raise ValueError('startup J6 command contains non-finite feedback')
    if target < -JOINT6_STARTUP_LIMIT_RAD - 1e-9 or target > 1e-9:
        raise ValueError(
            'startup J6 target %.6f is outside [-240deg, 0]' % target)
    measured_hold = abs(target - canonical) <= \
        JOINT6_STARTUP_COMMAND_EPSILON_RAD
    if previous_target is not None:
        previous = float(previous_target)
        if (
                target < previous - 1e-9
                and not measured_hold):
            raise ValueError(
                'startup J6 target decreased from %.6f to %.6f'
                % (previous, target))

    waiting_for_feedback_wrap = False
    wrapped_positive_side = bool(
        raw >= 0.0
        and canonical < 0.0
        and abs(raw - canonical) > math.pi)
    if wrapped_positive_side:
        if target < -math.pi:
            desired_controller_target = target + JOINT6_WRAP_RAD
        elif (
                abs(target - (
                    JOINT6_STARTUP_WRAP_TARGET_RAD - JOINT6_WRAP_RAD))
                <= JOINT6_STARTUP_WRAP_SETTLE_TOLERANCE_RAD):
            # The executor presents the +3.2 controller bridge through its
            # equivalent negative logical endpoint. Command the bridge until
            # measured raw feedback is actually inside its settle band. Only
            # then retain harmless positive servo overshoot rather than
            # correcting it backwards. Holding every exact bridge request at
            # its pre-bridge feedback caused a zero-progress startup timeout.
            if raw < (
                    JOINT6_STARTUP_WRAP_TARGET_RAD -
                    JOINT6_STARTUP_WRAP_SETTLE_TOLERANCE_RAD):
                desired_controller_target = JOINT6_STARTUP_WRAP_TARGET_RAD
            else:
                desired_controller_target = raw
            waiting_for_feedback_wrap = True
        elif abs(target) <= JOINT6_STARTUP_COMMAND_EPSILON_RAD:
            # Logical ready zero is always raw +2*pi on this wrapped-positive
            # startup branch. Do not make this mapping depend on observing the
            # bridge in this callback: the executor publishes each MoveJ
            # endpoint once, and the driver's coherent CAN snapshot can lag
            # the ROS feedback that just proved +3.2 by a few milliseconds.
            # Re-emitting +3.2 for that one zero message leaves the endpoint
            # permanently stalled. The enable handshake has already proved a
            # controller positive limit of at least +360 degrees, so +2*pi is
            # both monotonic and controller-authorized from any such sample.
            desired_controller_target = target + JOINT6_WRAP_RAD
        else:
            desired_controller_target = JOINT6_STARTUP_WRAP_TARGET_RAD
            waiting_for_feedback_wrap = True
    else:
        desired_controller_target = target

    if (
            desired_controller_target < raw -
            JOINT6_STARTUP_COMMAND_EPSILON_RAD
            and not measured_hold):
        bridge_overshoot = raw - desired_controller_target
        if (
                waiting_for_feedback_wrap
                and bridge_overshoot <=
                JOINT6_STARTUP_WRAP_SETTLE_TOLERANCE_RAD):
            # MoveJ can settle a fraction beyond the +3.2-rad bridge.  Never
            # command the bridge backwards to correct that harmless servo
            # overshoot: retain the exact measured controller coordinate.
            # The executor already observes the equivalent negative logical
            # angle within its endpoint tolerance and can advance to zero.
            desired_controller_target = raw
        else:
            raise ValueError(
                'startup J6 controller target %.6f is behind raw feedback %.6f'
                % (desired_controller_target, raw))
    return desired_controller_target, waiting_for_feedback_wrap


def reset_startup_joint6_transaction(node):
    """
    Clear one incomplete startup transaction after fail-closed disable.

    The arm can relax after its motors are disabled.  That motion must not be
    compared with the high-water mark from the preceding powered J6 startup,
    otherwise the stale direction watchdog continually reissues DisableArm
    and defeats a later commissioning enable.  The next coherent feedback
    sample will independently re-arm the negative logical branch when startup
    has not yet completed.
    """
    node._startup_joint6_active = False
    node._startup_joint6_armed = False
    node._startup_joint6_last_target = None
    node._startup_joint6_last_controller_target = None
    node._startup_joint6_direction_previous_raw = None
    node._startup_joint6_direction_unwrapped = None
    node._startup_joint6_direction_high_water = None
    node._continuous_joint6_feedback = None


def decode_joint_feedback_pair(arbitration_id, data):
    """Decode one PiPER joint-pair CAN frame into joint indices and radians."""
    can_id = int(arbitration_id)
    if can_id not in JOINT_FEEDBACK_CAN_INDEX:
        raise ValueError('unsupported joint-feedback CAN id 0x%03X' % can_id)
    payload = bytes(data)
    if len(payload) != 8:
        raise ValueError(
            'joint-feedback CAN frame 0x%03X has %d bytes, expected 8'
            % (can_id, len(payload)))
    raw_first = int.from_bytes(payload[0:4], byteorder='big', signed=True)
    raw_second = int.from_bytes(payload[4:8], byteorder='big', signed=True)
    return (
        JOINT_FEEDBACK_CAN_INDEX[can_id],
        (
            raw_first * JOINT_FEEDBACK_RAW_TO_RAD,
            raw_second * JOINT_FEEDBACK_RAW_TO_RAD,
        ),
    )


def coherent_joint_feedback(pairs, last_sequences, now, max_age=None,
                            max_skew=None):
    """
    Return six joints only after every pair advanced in one fresh cycle.

    ``pairs`` maps each PiPER feedback CAN ID to ``(sequence, stamp, values)``.
    The sequence gate prevents a fast pair from being combined repeatedly with
    an older pair, while the skew gate prevents frames from separate cycles
    from appearing as one six-joint sample.
    """
    age_limit = (
        JOINT_FEEDBACK_CAN_MAX_AGE_SEC if max_age is None else float(max_age))
    skew_limit = (
        JOINT_FEEDBACK_CAN_MAX_SKEW_SEC if max_skew is None else float(max_skew))
    missing = [can_id for can_id in JOINT_FEEDBACK_CAN_IDS if can_id not in pairs]
    if missing:
        return None, None, (
            'missing joint-feedback CAN frames: '
            + ', '.join('0x%03X' % can_id for can_id in missing))
    records = [pairs[can_id] for can_id in JOINT_FEEDBACK_CAN_IDS]
    try:
        sequences = tuple(int(record[0]) for record in records)
        stamps = tuple(float(record[1]) for record in records)
        values = tuple(tuple(float(value) for value in record[2])
                       for record in records)
    except (TypeError, ValueError, IndexError):
        return None, None, 'joint-feedback CAN records are invalid'
    if not all(
            len(pair) == 2 and all(math.isfinite(value) for value in pair)
            for pair in values):
        return None, None, 'joint-feedback CAN pairs are not finite pairs'
    if not all(math.isfinite(stamp) for stamp in stamps):
        return None, None, 'joint-feedback CAN timestamps are invalid'
    stale = [
        can_id for can_id, stamp in zip(JOINT_FEEDBACK_CAN_IDS, stamps)
        if float(now) - stamp > age_limit or stamp > float(now) + skew_limit
    ]
    if stale:
        return None, None, (
            'stale joint-feedback CAN frames: '
            + ', '.join('0x%03X' % can_id for can_id in stale))
    if max(stamps) - min(stamps) > skew_limit:
        return None, None, (
            'joint-feedback CAN frame skew %.6f > %.6f sec'
            % (max(stamps) - min(stamps), skew_limit))
    if last_sequences is not None:
        try:
            previous = tuple(int(value) for value in last_sequences)
        except (TypeError, ValueError):
            return None, None, 'last joint-feedback CAN sequences are invalid'
        if len(previous) != 3:
            return None, None, 'last joint-feedback CAN sequences are invalid'
        waiting = [
            can_id for can_id, sequence, old in zip(
                JOINT_FEEDBACK_CAN_IDS, sequences, previous)
            if sequence <= old
        ]
        if waiting:
            return None, None, (
                'waiting for a complete new joint-feedback CAN cycle: '
                + ', '.join('0x%03X' % can_id for can_id in waiting))
    positions = [0.0] * 6
    for can_id, pair in zip(JOINT_FEEDBACK_CAN_IDS, values):
        first, second = JOINT_FEEDBACK_CAN_INDEX[can_id]
        positions[first], positions[second] = pair
    return positions, sequences, ''


def joint_feedback_warning_due(
        reason, now, last_valid_at, last_warning_at,
        gap_sec=JOINT_FEEDBACK_WARNING_GAP_SEC, repeat_sec=1.0):
    """Warn only when rejected CAN cycles cause a sustained feedback gap."""
    if not reason or str(reason).startswith('waiting for a complete new'):
        return False
    return (
        float(now) - float(last_valid_at) >= float(gap_sec)
        and float(now) - float(last_warning_at) >= float(repeat_sec)
    )


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


def motor_driver_enable_states(piper):
    """
    Return the six feedback enable flags, or ``None`` if unavailable.

    The aggregate SDK helpers are command conveniences, not authoritative
    state.  In particular, an enable attempt can leave only the healthy axes
    enabled when one motor is faulted.  Every arm-wide transition therefore
    has to be proved from all six low-speed FOC feedback records.
    """
    try:
        low_speed = piper.GetArmLowSpdInfoMsgs()
        states = tuple(
            bool(getattr(low_speed, 'motor_%d' % index)
                 .foc_status.driver_enable_status)
            for index in range(1, 7)
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return states


def motor_driver_faults(piper):
    """Return active per-motor FOC faults from low-speed feedback."""
    try:
        low_speed = piper.GetArmLowSpdInfoMsgs()
        faults = []
        for index in range(1, 7):
            foc_status = getattr(
                low_speed, 'motor_%d' % index).foc_status
            for field in MOTOR_DRIVER_FAULT_FIELDS:
                if bool(getattr(foc_status, field, False)):
                    faults.append('joint%d:%s' % (index, field))
    except (AttributeError, TypeError, ValueError):
        return None
    return tuple(faults)


def gripper_feedback_diagnostic(piper, wall_time_fn=time.time):
    """Return a read-only diagnostic view of the SDK's latest 0x2A8 state."""
    try:
        wrapper = piper.GetArmGripperMsgs()
        state = wrapper.gripper_state
        foc = state.foc_status
        timestamp = float(getattr(wrapper, 'time_stamp', 0.0) or 0.0)
        now = float(wall_time_fn())
        faults = tuple(
            field for field in GRIPPER_FAULT_FIELDS
            if bool(getattr(foc, field, False)))
        return {
            'available': timestamp > 0.0,
            'timestamp': timestamp,
            'age_sec': (
                max(0.0, now - timestamp)
                if timestamp > 0.0 else math.inf),
            'hz': float(getattr(wrapper, 'Hz', 0.0) or 0.0),
            'position_raw': int(getattr(state, 'grippers_angle', 0)),
            'effort_raw': int(getattr(state, 'grippers_effort', 0)),
            'enabled': bool(getattr(foc, 'driver_enable_status', False)),
            'homed': bool(getattr(foc, 'homing_status', False)),
            'faults': faults,
        }
    except (AttributeError, TypeError, ValueError):
        return {
            'available': False,
            'timestamp': 0.0,
            'age_sec': math.inf,
            'hz': 0.0,
            'position_raw': 0,
            'effort_raw': 0,
            'enabled': False,
            'homed': False,
            'faults': (),
        }


def format_gripper_feedback_diagnostic(record):
    """Format feedback without treating an SDK default object as live data."""
    if not bool(record.get('available', False)):
        return 'feedback=NO_FEEDBACK'
    faults = ','.join(record.get('faults', ())) or 'none'
    return (
        'feedback=LIVE age=%.3fs hz=%.2f position=%d(%.3fmm) '
        'effort=%d(%.3fNm) enabled=%s homed=%s faults=%s'
        % (
            float(record['age_sec']), float(record['hz']),
            int(record['position_raw']),
            int(record['position_raw']) * 0.001,
            int(record['effort_raw']), int(record['effort_raw']) * 0.001,
            bool(record['enabled']), bool(record['homed']), faults))


def diagnostic_log(node, level, message):
    """Log diagnostics while preserving lightweight unbound-method fixtures."""
    try:
        logger = node.get_logger()
        method = getattr(logger, str(level), None)
        if callable(method):
            method(str(message))
    except AttributeError:
        pass


def log_enable_transition_state(node, label):
    states = motor_driver_enable_states(node.piper)
    faults = motor_driver_faults(node.piper)
    diagnostic_log(
        node,
        'info',
        '%s main_joint_enable_states=%s main_joint_faults=%s %s'
        % (
            str(label), states if states is not None else 'NO_FEEDBACK',
            faults if faults is not None else 'NO_FEEDBACK',
            format_gripper_feedback_diagnostic(
                gripper_feedback_diagnostic(node.piper))))


def request_piper_enable_state(piper, enabled, timeout_sec,
                               monotonic_fn=time.monotonic, sleep_fn=time.sleep):
    """Command and prove one all-six-motor enable/disable transition."""
    deadline = monotonic_fn() + max(0.0, float(timeout_sec))
    while True:
        if enabled:
            piper.EnablePiper()
        else:
            # DisableArm(7) is the explicit all-axis command.  It remains
            # effective when DisablePiper's aggregate return value is
            # ambiguous because one axis was already fault-disabled.
            disable_arm = getattr(piper, 'DisableArm', None)
            if callable(disable_arm):
                disable_arm(7)
            else:
                piper.DisablePiper()

        states = motor_driver_enable_states(piper)
        faults = motor_driver_faults(piper)
        if enabled and faults:
            return False
        target_reached = (
            states is not None
            and (all(states) if enabled else not any(states))
        )

        if target_reached:
            return True
        if monotonic_fn() >= deadline:
            return False
        sleep_fn(ENABLE_RETRY_PERIOD_SEC)


def qualify_startup_joint6_controller_limit(
        piper,
        configured_max_deg=JOINT6_STARTUP_CONTROLLER_MAX_DEG,
        required_max_deg=JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG,
        timeout_sec=JOINT6_STARTUP_CONTROLLER_LIMIT_TIMEOUT_SEC,
        monotonic_fn=time.monotonic,
        sleep_fn=time.sleep):
    """
    Set and prove the controller range required by positive-only startup.

    A logical startup position as low as -240 degrees is represented on the
    controller's positive multi-turn branch. Returning to logical zero while
    moving only in the positive direction therefore reaches raw +360 degrees.
    The retained turn offset maps the ordinary logical +pi endpoint to raw
    +540 degrees, so qualification must cover that complete powered-session
    interval rather than startup zero alone. The controller gets a small
    convergence margin; its negative limit and maximum speed remain unchanged.
    Ordinary logical J6 commands remain bounded to [-pi, +pi] by this driver.
    """
    configured = float(configured_max_deg)
    required = float(required_max_deg)
    if (
            not math.isfinite(configured)
            or not math.isfinite(required)
            or configured < required
            or required < 360.0):
        return False, 'invalid startup J6 controller-limit configuration'
    setter = getattr(piper, 'MotorAngleLimitMaxSpdSet', None)
    query = getattr(piper, 'SearchMotorMaxAngleSpdAccLimit', None)
    getter = getattr(piper, 'GetAllMotorAngleLimitMaxSpd', None)
    if not all(callable(method) for method in (setter, query, getter)):
        return False, 'PiPER SDK lacks controller-limit qualification APIs'
    setter(
        6,
        int(round(configured * 10.0)),
        PIPER_SETTING_UNCHANGED,
        PIPER_SETTING_UNCHANGED,
    )
    deadline = monotonic_fn() + max(0.0, float(timeout_sec))
    last_value = None
    while True:
        query(6, 0x01)
        feedback = getter()
        try:
            motor = feedback.all_motor_angle_limit_max_spd.motor[6]
            if int(motor.motor_num) == 6:
                last_value = float(motor.max_angle_limit) * 0.1
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            last_value = None
        if (
                last_value is not None
                and math.isfinite(last_value)
                and last_value >= required - 1e-9):
            return True, 'controller J6 positive limit %.1f deg' % last_value
        remaining = deadline - monotonic_fn()
        if remaining <= 0.0:
            shown = (
                'unavailable' if last_value is None
                else '%.1f deg' % last_value)
            return False, (
                'controller J6 positive limit did not reach %.1f deg; got %s'
                % (required, shown))
        sleep_fn(min(0.05, remaining))


class PiperRosNode(Node):
    """Run the ROS 2 node for the robotic arm."""

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
        self.declare_parameter(
            'startup_joint6_controller_max_deg',
            JOINT6_STARTUP_CONTROLLER_MAX_DEG,
        )

        self.can_port = (
            self.get_parameter('can_port').get_parameter_value().string_value)
        self.auto_enable = (
            self.get_parameter('auto_enable').get_parameter_value().bool_value)
        self.gripper_exist = self.get_parameter(
            'gripper_exist').get_parameter_value().bool_value
        self.enable_timeout = self.get_parameter(
            'enable_timeout').get_parameter_value().double_value
        self.joint_bounds_path = self.get_parameter(
            'joint_bounds_path').get_parameter_value().string_value
        self.startup_joint6_controller_max_deg = self.get_parameter(
            'startup_joint6_controller_max_deg'
        ).get_parameter_value().double_value
        self.joint_bounds = self.load_joint_bounds(self.joint_bounds_path)

        self.get_logger().info(f"can_port is {self.can_port}")
        self.get_logger().info(f"auto_enable is {self.auto_enable}")
        self.get_logger().info(f"gripper_exist is {self.gripper_exist}")
        self.get_logger().info(f"enable_timeout is {self.enable_timeout}")
        self.get_logger().info(
            f"joint_bounds_path is "
            f"{self.joint_bounds_path or 'default limits'}")
        self.get_logger().info(
            'startup_joint6_controller_max_deg is %.1f'
            % self.startup_joint6_controller_max_deg)
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
        self.joint_states.name = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8',
        ]
        self.joint_states.position = [0.0] * 8
        self.joint_states.velocity = [0.0] * 8
        self.joint_states.effort = [0.0] * 8
        self._raw_joint_lock = threading.Lock()
        self._raw_joint_pairs = {}
        self._raw_joint_sequence = 0
        self._raw_joint_last_emitted_sequences = None
        self._raw_joint_last_valid_at = time.monotonic()
        self._raw_joint_warning_at = 0.0
        self._raw_joint_stop = threading.Event()
        self._continuous_joint6_feedback = None
        self._published_joint6_feedback = None
        self._raw_joint6_feedback = None
        self._latest_raw_arm_positions = None
        self._startup_joint6_armed = False
        self._startup_joint6_active = False
        self._startup_joint6_finished = False
        self._startup_joint6_last_target = None
        self._startup_joint6_last_controller_target = None
        self._joint6_controller_turn_offset = 0.0
        self._startup_joint6_direction_previous_raw = None
        self._startup_joint6_direction_unwrapped = None
        self._startup_joint6_direction_high_water = None
        self._raw_joint_bus = None
        self._raw_joint_thread = None
        # Enable flag
        self.__enable_flag = False
        self._command_cache_lock = threading.Lock()
        self._motion_ctrl_2_signature = None
        self._gripper_command_signature = None
        self._motor_watchdog_reason = ''
        self._motor_watchdog_disable_at = 0.0
        self._motor_watchdog_started_at = time.monotonic()
        self._latest_motor_states = None
        self._latest_motor_faults = None
        self._disable_required = False
        self._enable_transition_active = False
        self._enable_transition_lock = threading.Lock()
        # Create piper class and open CAN interface
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        self.start_raw_joint_feedback_receiver()

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

    def start_raw_joint_feedback_receiver(self):
        """Start a storage-only CAN reader for coherent joint-pair cycles."""
        filters = [
            {'can_id': can_id, 'can_mask': 0x7FF, 'extended': False}
            for can_id in JOINT_FEEDBACK_CAN_IDS
        ]
        try:
            self._raw_joint_bus = can.interface.Bus(
                interface='socketcan',
                channel=self.can_port,
                can_filters=filters,
                receive_own_messages=False,
            )
        except (can.CanError, OSError) as exc:
            raise RuntimeError(
                'could not open passive joint-feedback CAN receiver on %s: %s'
                % (self.can_port, exc)) from exc
        self._raw_joint_thread = threading.Thread(
            target=self._raw_joint_feedback_loop,
            name='piper-coherent-joint-feedback',
            daemon=True,
        )
        self._raw_joint_thread.start()
        self.get_logger().info(
            'Passive coherent joint-feedback receiver is active on %s'
            % self.can_port)

    def _raw_joint_feedback_loop(self):
        """Store raw pair frames; ROS publication remains executor-owned."""
        while not self._raw_joint_stop.is_set():
            try:
                message = self._raw_joint_bus.recv(timeout=0.1)
            except (can.CanError, OSError) as exc:
                now = time.monotonic()
                if now - self._raw_joint_warning_at >= 1.0:
                    self.get_logger().error(
                        'Passive joint-feedback CAN receive failed: %s' % exc)
                    self._raw_joint_warning_at = now
                continue
            if message is None:
                continue
            if int(message.arbitration_id) not in JOINT_FEEDBACK_CAN_INDEX:
                # Kernel filters normally prevent this. A frame already queued
                # while filters are installed is harmless and not a fault.
                continue
            try:
                _, values = decode_joint_feedback_pair(
                    message.arbitration_id, message.data)
            except (TypeError, ValueError) as exc:
                now = time.monotonic()
                if now - self._raw_joint_warning_at >= 1.0:
                    self.get_logger().warn(
                        'Ignored invalid joint-feedback CAN frame: %s' % exc)
                    self._raw_joint_warning_at = now
                continue
            with self._raw_joint_lock:
                self._raw_joint_sequence += 1
                self._raw_joint_pairs[int(message.arbitration_id)] = (
                    self._raw_joint_sequence,
                    time.monotonic(),
                    values,
                )

    def coherent_raw_joint_positions(self):
        """Consume one fresh complete raw joint-feedback cycle."""
        now = time.monotonic()
        with self._raw_joint_lock:
            positions, sequences, reason = coherent_joint_feedback(
                dict(self._raw_joint_pairs),
                self._raw_joint_last_emitted_sequences,
                now,
            )
            if not reason:
                self._raw_joint_last_emitted_sequences = sequences
                self._raw_joint_last_valid_at = now
        return positions, reason

    def stop_raw_joint_feedback_receiver(self):
        """Stop and close the passive reader without affecting arm commands."""
        self._raw_joint_stop.set()
        if self._raw_joint_thread is not None:
            self._raw_joint_thread.join(timeout=1.0)
        if self._raw_joint_bus is not None:
            try:
                self._raw_joint_bus.shutdown()
            except (can.CanError, OSError):
                pass

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
        sent = False
        with self._command_cache_lock:
            if signature == self._gripper_command_signature:
                sent = False
            else:
                self.piper.GripperCtrl(*signature)
                self._gripper_command_signature = signature
                sent = True
        diagnostic_log(
            self,
            'info' if sent else 'debug',
            'Gripper command %s mode=%s(0x%02X) position=%d(%.3fmm) '
            'effort=%d(%.3fNm) set_zero=0x%02X'
            % (
                'SENT' if sent else 'NOT_SENT_UNCHANGED',
                GRIPPER_COMMAND_MODES.get(signature[2], 'UNKNOWN'),
                signature[2], signature[0], signature[0] * 0.001,
                signature[1], signature[1] * 0.001, signature[3]))
        return sent

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
            self.get_logger().warn(
                f"Could not load joint bounds from {path}: {exc}. "
                "Using default limits.")
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
        with self._enable_transition_lock:
            self._disable_required = False
            self._enable_transition_active = True
            try:
                succeeded = request_piper_enable_state(
                    self.piper, True, self.enable_timeout)
                if succeeded:
                    succeeded = self.qualify_startup_joint6_limit()
                if not succeeded:
                    self._disable_required = True
                    rollback_succeeded = request_piper_enable_state(
                        self.piper, False, self.enable_timeout)
            finally:
                self._enable_transition_active = False
        if succeeded:
            self.__enable_flag = True
            self.send_gripper_if_changed(0, 1000, 0x01, 0)
            return
        self.__enable_flag = False
        if rollback_succeeded:
            self._disable_required = False
            self.get_logger().error(
                'Automatic enable failed and all six motors were rolled back to disabled')
        else:
            self.get_logger().fatal(
                'Automatic enable failed and rollback could not prove all six motors disabled')

    def qualify_startup_joint6_limit(self):
        """Prove the controller has room for positive-only startup to zero."""
        succeeded, reason = qualify_startup_joint6_controller_limit(
            self.piper,
            configured_max_deg=self.startup_joint6_controller_max_deg,
        )
        if succeeded:
            self.get_logger().info(
                'Qualified positive-only startup range: %s' % reason)
        else:
            self.get_logger().error(
                'Could not qualify positive-only startup range: %s' % reason)
        return succeeded

    def publish_feedback(self):
        """Publish one executor-owned feedback sample from the robotic arm."""
        if not self.feedback_timer_started:
            self.feedback_timer_started = True
            self.get_logger().info(
                'Executor-owned 200 Hz feedback timer is active')
        self.fail_closed_motor_watchdog()
        self.PublishArmState()
        self.PublishArmJointAndGripper()
        self.PublishArmEndPose()

    def fail_closed_motor_watchdog(self):
        """Disable every axis when feedback shows a powered unsafe state."""
        states = motor_driver_enable_states(self.piper)
        faults = motor_driver_faults(self.piper)
        self._latest_motor_states = states
        self._latest_motor_faults = faults
        if states is None or faults is None:
            return
        partial_enable = any(states) and not all(states)
        active_faults = tuple(faults)
        transition_active = bool(getattr(
            self, '_enable_transition_active', False))
        disable_required = bool(getattr(self, '_disable_required', False))
        # The SDK exposes six low-speed motor records that are refreshed one
        # at a time.  Immediately after attaching to an already-powered arm,
        # the first aggregate read can therefore look partially enabled even
        # though no motor changed state.  Allow only that initial, fault-free,
        # non-commanded snapshot time to cohere.  A real fault or an explicit
        # disable requirement still acts immediately, and a persistent partial
        # state is fail-closed as soon as the bounded grace expires.
        watchdog_age = time.monotonic() - float(getattr(
            self, '_motor_watchdog_started_at', 0.0))
        if (
                partial_enable
                and not active_faults
                and not disable_required
                and not transition_active
                and watchdog_age < MOTOR_WATCHDOG_STARTUP_GRACE_SEC):
            self._motor_watchdog_reason = ''
            return
        unsafe = any(states) and (
            disable_required
            or
            bool(active_faults)
            or (partial_enable and not transition_active)
        )
        if not unsafe:
            if not any(states):
                self.__enable_flag = False
                self._disable_required = False
            self._motor_watchdog_reason = ''
            return
        reason_parts = []
        if disable_required:
            reason_parts.append('all-axis disable remains unproved')
        if partial_enable:
            reason_parts.append('partial enable flags=%s' % (states,))
        if active_faults:
            reason_parts.append('motor faults=%s' % ','.join(active_faults))
        reason = '; '.join(reason_parts)
        now = time.monotonic()
        if now - self._motor_watchdog_disable_at >= 0.1:
            self.piper.DisableArm(7)
            self._motor_watchdog_disable_at = now
        self.__enable_flag = False
        self._disable_required = True
        self.reset_command_cache()
        if reason != self._motor_watchdog_reason:
            self.get_logger().error(
                'Fail-closed motor watchdog disabled all axes: ' + reason)
            self._motor_watchdog_reason = reason

    def PublishArmState(self):
        arm_status = PiperStatusMsg()
        sdk_status = self.piper.GetArmStatus().arm_status
        sdk_errors = sdk_status.err_status
        arm_status.ctrl_mode = sdk_status.ctrl_mode
        arm_status.arm_status = sdk_status.arm_status
        arm_status.mode_feedback = sdk_status.mode_feed
        arm_status.teach_status = sdk_status.teach_status
        arm_status.motion_status = sdk_status.motion_status
        arm_status.trajectory_num = sdk_status.trajectory_num
        arm_status.err_code = sdk_status.err_code
        arm_status.joint_1_angle_limit = sdk_errors.joint_1_angle_limit
        arm_status.joint_2_angle_limit = sdk_errors.joint_2_angle_limit
        arm_status.joint_3_angle_limit = sdk_errors.joint_3_angle_limit
        arm_status.joint_4_angle_limit = sdk_errors.joint_4_angle_limit
        arm_status.joint_5_angle_limit = sdk_errors.joint_5_angle_limit
        arm_status.joint_6_angle_limit = sdk_errors.joint_6_angle_limit
        arm_status.communication_status_joint_1 = (
            sdk_errors.communication_status_joint_1)
        arm_status.communication_status_joint_2 = (
            sdk_errors.communication_status_joint_2)
        arm_status.communication_status_joint_3 = (
            sdk_errors.communication_status_joint_3)
        arm_status.communication_status_joint_4 = (
            sdk_errors.communication_status_joint_4)
        arm_status.communication_status_joint_5 = (
            sdk_errors.communication_status_joint_5)
        arm_status.communication_status_joint_6 = (
            sdk_errors.communication_status_joint_6)
        motor_states = self._latest_motor_states
        motor_faults = self._latest_motor_faults
        arm_status.motor_feedback_valid = bool(
            motor_states is not None and motor_faults is not None)
        normalized_states = (
            tuple(bool(value) for value in motor_states)
            if motor_states is not None and len(motor_states) == 6
            else (False,) * 6)
        arm_status.motor_1_driver_enabled = normalized_states[0]
        arm_status.motor_2_driver_enabled = normalized_states[1]
        arm_status.motor_3_driver_enabled = normalized_states[2]
        arm_status.motor_4_driver_enabled = normalized_states[3]
        arm_status.motor_5_driver_enabled = normalized_states[4]
        arm_status.motor_6_driver_enabled = normalized_states[5]
        arm_status.motor_faults = (
            list(motor_faults) if motor_faults is not None else [])
        arm_status.motor_watchdog_reason = str(self._motor_watchdog_reason)
        self.arm_status_pub.publish(arm_status)

    def PublishArmJointAndGripper(self):
        positions, raw_reason = self.coherent_raw_joint_positions()
        if raw_reason:
            now = time.monotonic()
            # A 200 Hz timer normally waits several ticks for the next complete
            # CAN cycle. Keep that expected condition quiet, but report actual
            # missing, stale, or skewed feedback once per second.
            if joint_feedback_warning_due(
                    raw_reason,
                    now,
                    self._raw_joint_last_valid_at,
                    self._raw_joint_warning_at):
                self.get_logger().warn(
                    'Joint feedback unavailable for at least %.2f sec: %s'
                    % (JOINT_FEEDBACK_WARNING_GAP_SEC, raw_reason))
                self._raw_joint_warning_at = now
            return
        # Assign a ROS timestamp only to a complete newly assembled cycle.
        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        speed_feedback = self.piper.GetArmHighSpdInfoMsgs()
        gripper_feedback = self.piper.GetArmGripperMsgs().gripper_state
        joint_0, joint_1, joint_2, joint_3, joint_4, raw_joint_5 = positions
        # Keep one immutable coherent controller-coordinate snapshot for
        # STARTUP_WRIST.  That transaction must rotate J6 only.  In
        # particular, powered gravity relaxation can leave J2/J3 just beyond
        # their ordinary command limits; clipping those axes while issuing the
        # J6 endpoint creates unintended motion and makes the executor wait on
        # a pose that the driver never commanded.
        self._latest_raw_arm_positions = tuple(float(value) for value in positions)
        self._raw_joint6_feedback = raw_joint_5
        if self._startup_joint6_active:
            if self._startup_joint6_direction_previous_raw is None:
                continuous_direction, high_water, wrong_direction = \
                    startup_joint6_direction_update(
                        raw_joint_5, None, raw_joint_5, raw_joint_5)
            else:
                continuous_direction, high_water, wrong_direction = \
                    startup_joint6_direction_update(
                        raw_joint_5,
                        self._startup_joint6_direction_previous_raw,
                        self._startup_joint6_direction_unwrapped,
                        self._startup_joint6_direction_high_water,
                    )
            self._startup_joint6_direction_previous_raw = raw_joint_5
            self._startup_joint6_direction_unwrapped = continuous_direction
            self._startup_joint6_direction_high_water = high_water
            if wrong_direction:
                reason = (
                    'startup J6 moved in the forbidden negative direction: '
                    'unwrapped feedback %.6f fell behind high-water %.6f rad'
                    % (continuous_direction, high_water))
                self.piper.DisableArm(7)
                self._PiperRosNode__enable_flag = False
                self._disable_required = True
                self._motor_watchdog_reason = reason
                self.reset_command_cache()
                reset_startup_joint6_transaction(self)
                self.get_logger().error(
                    'Fail-closed startup direction watchdog disabled all '
                    'axes: ' + reason)
        previous_joint_5 = self._continuous_joint6_feedback
        if self._startup_joint6_finished:
            joint_5 = standard_joint6_feedback(raw_joint_5)
        else:
            joint_5 = continuous_joint6_feedback(
                raw_joint_5, previous_joint_5)
            if previous_joint_5 is None:
                # A fresh driver may begin at storage or part-way through an
                # interrupted startup. Any nonpositive logical J6 state can
                # safely resume toward zero. A genuinely positive state is
                # deliberately left unarmed because reaching zero from there
                # would require forbidden negative startup motion.
                self._startup_joint6_armed = bool(joint_5 <= 1e-9)
                if self._startup_joint6_armed:
                    self.get_logger().info(
                        'Armed startup-only negative J6 branch from raw '
                        'feedback %.6f rad as %.6f rad'
                        % (raw_joint_5, joint_5))
            self._continuous_joint6_feedback = joint_5
            if (
                    self._startup_joint6_active
                    and self._startup_joint6_last_target is not None
                    and abs(self._startup_joint6_last_target) <=
                    JOINT6_STARTUP_READY_TOLERANCE_RAD
                    and abs(joint_5) <=
                    JOINT6_STARTUP_READY_TOLERANCE_RAD):
                self._startup_joint6_active = False
                self._startup_joint6_armed = False
                self._startup_joint6_finished = True
                self._startup_joint6_last_target = None
                self._startup_joint6_last_controller_target = None
                self._startup_joint6_direction_previous_raw = None
                self._startup_joint6_direction_unwrapped = None
                self._startup_joint6_direction_high_water = None
                self._joint6_controller_turn_offset = (
                    round(
                        (raw_joint_5 - standard_joint6_feedback(raw_joint_5))
                        / JOINT6_WRAP_RAD)
                    * JOINT6_WRAP_RAD)
                self._continuous_joint6_feedback = None
                joint_5 = standard_joint6_feedback(raw_joint_5)
                self.get_logger().info(
                    'Startup-only positive J6 rotation reached ready zero; '
                    'restored standard [-pi,+pi] logical feedback with '
                    'controller turn offset %.6f rad'
                    % self._joint6_controller_turn_offset)
        self._published_joint6_feedback = joint_5
        joint_6: float = gripper_feedback.grippers_angle / 1000000
        vel_0: float = speed_feedback.motor_1.motor_speed / 1000
        vel_1: float = speed_feedback.motor_2.motor_speed / 1000
        vel_2: float = speed_feedback.motor_3.motor_speed / 1000
        vel_3: float = speed_feedback.motor_4.motor_speed / 1000
        vel_4: float = speed_feedback.motor_5.motor_speed / 1000
        vel_5: float = speed_feedback.motor_6.motor_speed / 1000
        effort_6: float = gripper_feedback.grippers_effort / 1000
        velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
        self.joint_states.position = [
            joint_0, joint_1, joint_2, joint_3,
            joint_4, joint_5, joint_6, -joint_6,
        ]
        self.joint_states.velocity = velocities + [0.0, 0.0]
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
        """Handle an end-effector pose command."""
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
        """Handle a joint-angle command."""
        factor = 57324.840764  # 1000*180/3.14
        self.get_logger().debug("Received JointState command")
        if len(joint_data.position) < 6:
            self.get_logger().warn("Ignoring JointState command with fewer than 6 arm joints")
            return

        arm_joint_0 = self.enforce_joint_bound(
            'joint1', self.get_joint_value(joint_data, 'joint1', 0))
        requested_joint_2 = self.enforce_joint_bound(
            'joint2', self.get_joint_value(joint_data, 'joint2', 1))
        requested_joint_3 = self.enforce_joint_bound(
            'joint3', self.get_joint_value(joint_data, 'joint3', 2))
        arm_joint_1 = controller_command_position('joint2', requested_joint_2)
        arm_joint_2 = controller_command_position('joint3', requested_joint_3)
        arm_joint_3 = self.enforce_joint_bound(
            'joint4', self.get_joint_value(joint_data, 'joint4', 3))
        arm_joint_4 = self.enforce_joint_bound(
            'joint5', self.get_joint_value(joint_data, 'joint5', 4))
        requested_joint_6 = self.get_joint_value(joint_data, 'joint6', 5)
        command_frame = str(getattr(
            getattr(joint_data, 'header', None), 'frame_id', ''))
        startup_stage_command = command_frame == JOINT6_STARTUP_COMMAND_FRAME
        explicit_hold_command = command_frame == JOINT6_HOLD_COMMAND_FRAME
        commissioning_command = (
            command_frame == JOINT6_COMMISSIONING_COMMAND_FRAME)
        startup_transaction_available = bool(
            getattr(self, '_startup_joint6_active', False)
            or getattr(self, '_startup_joint6_armed', False))
        startup_stage_candidate = bool(
            not bool(getattr(self, '_startup_joint6_finished', True))
            and getattr(self, '_raw_joint6_feedback', None) is not None
            and getattr(self, '_published_joint6_feedback', None) is not None
            and requested_joint_6 <= 1e-9
            and startup_transaction_available
            and startup_stage_command
        )
        startup_hold_candidate = bool(
            not bool(getattr(self, '_startup_joint6_finished', True))
            and getattr(self, '_raw_joint6_feedback', None) is not None
            and getattr(self, '_published_joint6_feedback', None) is not None
            and startup_transaction_available
            and explicit_hold_command
        )
        commissioning_hold_candidate = bool(
            not bool(getattr(self, '_startup_joint6_finished', True))
            and startup_transaction_available
            and commissioning_command
            and getattr(self, '_raw_joint6_feedback', None) is not None
            and getattr(self, '_published_joint6_feedback', None) is not None
            and abs(
                requested_joint_6
                - float(self._published_joint6_feedback))
            <= JOINT6_STARTUP_READY_TOLERANCE_RAD
        )
        commissioning_startup_candidate = bool(
            not bool(getattr(self, '_startup_joint6_finished', True))
            and startup_transaction_available
            and commissioning_command
            and getattr(self, '_published_joint6_feedback', None) is not None
            and requested_joint_6 <= 1e-9
            and requested_joint_6 > (
                float(self._published_joint6_feedback)
                + JOINT6_STARTUP_READY_TOLERANCE_RAD)
        )
        startup_motion_candidate = bool(
            startup_stage_candidate or commissioning_startup_candidate)
        if startup_stage_command and not startup_stage_candidate:
            self.get_logger().error(
                'Rejected STARTUP_WRIST J6 command outside the armed '
                'startup-only positive-direction transaction')
            return
        if (
                startup_transaction_available
                and not startup_motion_candidate
                and not startup_hold_candidate
                and not commissioning_hold_candidate):
            if commissioning_command:
                self.get_logger().error(
                    'Rejected commissioning J6 command while STARTUP_WRIST '
                    'is armed; hold measured J6 or command only the positive '
                    'direction toward ready zero')
                return
            self.get_logger().error(
                'Rejected non-startup J6 motion while STARTUP_WRIST is armed; '
                'only the explicit measured hold or tagged positive-direction '
                'startup transaction is allowed')
            return
        if startup_hold_candidate or commissioning_hold_candidate:
            # The executor's hold snapshot and this callback's newest driver
            # feedback are asynchronous. Never use their small discrepancy as
            # a J6 move while startup is armed: command the exact measured J6
            # controller coordinate and leave the transaction state unchanged.
            arm_joint_5, _waiting_for_wrap = \
                startup_joint6_controller_target(
                    self._raw_joint6_feedback,
                    self._published_joint6_feedback,
                    self._published_joint6_feedback,
                    None,
                )
        elif startup_motion_candidate:
            try:
                arm_joint_5, waiting_for_wrap = \
                    startup_joint6_controller_target(
                        self._raw_joint6_feedback,
                        self._published_joint6_feedback,
                        requested_joint_6,
                        self._startup_joint6_last_target,
                    )
            except ValueError as exc:
                self.get_logger().error(
                    'Rejected unsafe startup J6 command: %s' % exc)
                return
            if not self._startup_joint6_active:
                self.get_logger().info(
                    'Activated startup-only positive J6 command mode%s'
                    % (
                        ' from explicit commissioning control'
                        if commissioning_startup_candidate else ''))
            self._startup_joint6_active = True
            self._startup_joint6_last_target = requested_joint_6
            previous_controller_target = getattr(
                self, '_startup_joint6_last_controller_target', None)
            if (
                    previous_controller_target is None
                    or abs(float(arm_joint_5) - float(
                        previous_controller_target)) > 1e-4):
                self.get_logger().info(
                    'STARTUP_WRIST map logical %.6f raw_feedback %.6f '
                    'logical_feedback %.6f -> raw_target %.6f'
                    % (
                        requested_joint_6,
                        self._raw_joint6_feedback,
                        self._published_joint6_feedback,
                        arm_joint_5,
                    ))
                self._startup_joint6_last_controller_target = arm_joint_5
            if waiting_for_wrap:
                self.get_logger().debug(
                    'Commanding startup-only raw +3.2-rad wrap bridge before '
                    'the final positive raw +2*pi move to logical ready zero')
        else:
            logical_joint_6 = self.enforce_joint_bound(
                'joint6', requested_joint_6)
            # A completed positive-only startup can leave the controller on
            # its +2*pi multi-turn branch while ROS correctly publishes
            # logical ready zero.  Keep every later MoveJ target on that same
            # controller branch for the rest of the powered session.  Dropping
            # the offset here numerically commands a large anticlockwise move.
            arm_joint_5 = (
                logical_joint_6
                + float(getattr(
                    self, '_joint6_controller_turn_offset', 0.0)))
            controller_max_rad = math.radians(float(getattr(
                self,
                'startup_joint6_controller_max_deg',
                JOINT6_STARTUP_CONTROLLER_MAX_DEG,
            )))
            if arm_joint_5 > controller_max_rad + 1e-9:
                self.get_logger().error(
                    'Rejected J6 controller target %.6f rad above the '
                    'qualified %.1f-deg positive limit'
                    % (
                        arm_joint_5,
                        math.degrees(controller_max_rad),
                    ))
                return
        if startup_motion_candidate or startup_hold_candidate:
            measured = getattr(self, '_latest_raw_arm_positions', None)
            if (
                    measured is None
                    or len(measured) < 5
                    or not all(math.isfinite(float(value))
                               for value in measured[:5])):
                self.get_logger().error(
                    'Rejected STARTUP_WRIST command without a fresh coherent '
                    'J1-J5 controller-coordinate snapshot')
                return
            # PiPER JointCtrl always transports all six axes.  Preserve the
            # newest measured J1-J5 exactly for this tagged transaction and do
            # not pass them through ordinary command-limit normalization.
            # STARTUP_WRIST authority is J6-only; ROUGH_HOME immediately after
            # it restores the configured all-joint pose normally.
            arm_joint_0, arm_joint_1, arm_joint_2, arm_joint_3, arm_joint_4 = (
                float(value) for value in measured[:5])
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
            gripper_joint = self.enforce_joint_bound(
                'joint7', self.get_joint_value(joint_data, 'joint7', 6))
            self.get_logger().debug(f"gripper: {gripper_joint}")
            joint_6 = round(gripper_joint*1000*1000)
            joint_6 = int(clip(joint_6, 0, 80000))
        else:
            joint_6 = 0
        if(self.GetEnableFlag()):
            # 设定电机速度
            if(joint_data.velocity != []):
                all_zeros = all(v == 0 for v in joint_data.velocity)
            else:
                all_zeros = True
            if not all_zeros:
                lens = len(joint_data.velocity)
                if lens >= 7:
                    vel_all = int(clip(round(self.get_joint_velocity(
                        joint_data, 'joint7', 6)), 0, 100))
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
                    gripper_effort = float(clip(self.get_joint_effort(
                        joint_data, 'joint7', 6), 0.5, 3))
                    self.get_logger().debug(f"gripper_effort: {gripper_effort}")
                    gripper_effort = round(gripper_effort * 1000)
                    self.send_gripper_if_changed(
                        abs(joint_6), gripper_effort, 0x01, 0)
                else:
                    self.send_gripper_if_changed(
                        abs(joint_6), 1000, 0x01, 0)

    def enable_callback(self, enable_flag: Bool):
        """Handle a motor enable or disable topic command."""
        self.get_logger().info(
            'ROS enable/disable topic request received: requested_enable=%s'
            % bool(enable_flag.data))
        log_enable_transition_state(self, 'BEFORE topic request')
        self.reset_command_cache()
        requested_enable = bool(enable_flag.data)
        with self._enable_transition_lock:
            self._disable_required = not requested_enable
            self._enable_transition_active = requested_enable
            try:
                succeeded = request_piper_enable_state(
                    self.piper, requested_enable, self.enable_timeout)
                rollback_succeeded = True
                if succeeded and requested_enable:
                    succeeded = self.qualify_startup_joint6_limit()
                if not succeeded and requested_enable:
                    self._disable_required = True
                    rollback_succeeded = request_piper_enable_state(
                        self.piper, False, self.enable_timeout)
            finally:
                self._enable_transition_active = False
        if not succeeded and requested_enable:
            if not rollback_succeeded:
                self.get_logger().fatal(
                    'Topic enable failed and rollback could not prove all six motors disabled')
        if succeeded or (requested_enable and rollback_succeeded):
            self._disable_required = False
        if not requested_enable and succeeded:
            reset_startup_joint6_transaction(self)
        self.__enable_flag = bool(succeeded and requested_enable)
        log_enable_transition_state(
            self, 'AFTER main-joint topic transition before gripper command')
        gripper_sent = False
        if succeeded and self.gripper_exist:
            gripper_sent = self.send_gripper_if_changed(
                0, 1000, 0x01 if requested_enable else 0x00, 0)
        else:
            diagnostic_log(
                self, 'info',
                'Gripper command NOT_SENT transition_succeeded=%s '
                'gripper_exist=%s' % (succeeded, self.gripper_exist))
        if succeeded and self.gripper_exist:
            diagnostic_log(
                self, 'info',
                'Topic transition gripper command sent=%s requested_enable=%s'
                % (gripper_sent, requested_enable))
        log_enable_transition_state(self, 'AFTER topic request')
        self.get_logger().info(
            'ROS enable/disable topic request completed: '
            'requested_enable=%s succeeded=%s internal_enable_flag=%s'
            % (requested_enable, succeeded, self.__enable_flag))

    def handle_enable_service(self, req, resp):
        """Handle the enable service for the robotic arm."""
        self.get_logger().info(
            'ROS enable/disable service request received: requested_enable=%s'
            % bool(req.enable_request))
        requested_enable = bool(req.enable_request)
        log_enable_transition_state(self, 'BEFORE service request')
        with self._enable_transition_lock:
            self._disable_required = not requested_enable
            self._enable_transition_active = requested_enable
            try:
                succeeded = request_piper_enable_state(
                    self.piper,
                    requested_enable,
                    self.enable_timeout,
                )
                rollback_succeeded = True
                if succeeded and requested_enable:
                    succeeded = self.qualify_startup_joint6_limit()
                if not succeeded and requested_enable:
                    # Keep the transition lock held until the failed enable is
                    # proved fully disabled; another caller must not re-enable
                    # between the failure and its fail-closed rollback.
                    self._disable_required = True
                    rollback_succeeded = request_piper_enable_state(
                        self.piper,
                        False,
                        self.enable_timeout,
                    )
            finally:
                self._enable_transition_active = False
        self.reset_command_cache()
        log_enable_transition_state(
            self, 'AFTER main-joint service transition before gripper command')
        gripper_sent = False
        if succeeded:
            if requested_enable:
                gripper_sent = self.send_gripper_if_changed(
                    0, 1000, 0x01, 0)
            else:
                gripper_sent = self.send_gripper_if_changed(
                    0, 1000, 0x02, 0)
            diagnostic_log(
                self, 'info',
                'Service transition gripper command sent=%s '
                'requested_enable=%s' % (gripper_sent, requested_enable))
        else:
            diagnostic_log(
                self, 'info',
                'Gripper command NOT_SENT because main-joint transition '
                'did not succeed')
            self.get_logger().error(
                f"Timed out waiting for motors to {'enable' if requested_enable else 'disable'}"
            )
            if requested_enable:
                # A faulted axis can make an all-arm enable time out after the
                # other five axes have enabled.  Never return from a failed
                # enable with that partial powered state still active.
                if rollback_succeeded:
                    self.get_logger().error(
                        'Failed enable rolled back; all six motor feedback flags are disabled'
                    )
                else:
                    self.get_logger().fatal(
                        'Failed enable rollback could not prove all six motors disabled'
                    )

        self.__enable_flag = bool(succeeded and requested_enable)
        if succeeded or (requested_enable and rollback_succeeded):
            self._disable_required = False
        if not requested_enable and succeeded:
            reset_startup_joint6_transaction(self)
        resp.enable_response = bool(succeeded)
        log_enable_transition_state(self, 'AFTER service request')
        self.get_logger().info(
            'ROS enable/disable service response: requested_enable=%s '
            'enable_response=%s internal_enable_flag=%s'
            % (requested_enable, resp.enable_response, self.__enable_flag))
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
        piper_single_node.stop_raw_joint_feedback_receiver()
        piper_single_node.destroy_node()
        rclpy.shutdown()
