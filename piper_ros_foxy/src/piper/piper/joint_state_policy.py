"""Joint command mapping and coherent raw-feedback policy."""

import math


JOINT6_LIMIT_RAD = math.pi
JOINT6_STARTUP_LIMIT_RAD = math.radians(240.0)
JOINT6_WRAP_RAD = 2.0 * math.pi
JOINT6_STARTUP_READY_TOLERANCE_RAD = 0.03
JOINT6_STARTUP_COMMAND_EPSILON_RAD = 1e-4
JOINT6_STARTUP_WRAP_TARGET_RAD = 3.2
JOINT6_STARTUP_WRAP_SETTLE_TOLERANCE_RAD = 0.005
JOINT6_STARTUP_DIRECTION_TRIP_RAD = 0.01
JOINT6_STARTUP_CONTROLLER_MAX_DEG = 545.0
JOINT6_STARTUP_CONTROLLER_REQUIRED_DEG = 540.0
JOINT6_STARTUP_CONTROLLER_LIMIT_TIMEOUT_SEC = 2.0
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
