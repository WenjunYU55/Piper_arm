import math

from piper.piper_ctrl_single_node import (
    JOINT_FEEDBACK_RAW_TO_RAD,
    JOINT6_STARTUP_WRAP_TARGET_RAD,
    JOINT6_STARTUP_LIMIT_RAD,
    coherent_joint_feedback,
    continuous_joint6_feedback,
    controller_command_position,
    decode_joint_feedback_pair,
    joint_feedback_warning_due,
    standard_joint6_feedback,
    startup_joint6_controller_target,
    startup_joint6_direction_update,
)


def test_joint6_startup_limit_is_exactly_240_degrees():
    assert abs(JOINT6_STARTUP_LIMIT_RAD - math.radians(240.0)) < 1e-12


def test_joint6_first_feedback_at_240_degree_storage_selects_negative_branch():
    wrapped_positive = math.radians(120.0)
    logical = continuous_joint6_feedback(wrapped_positive)
    assert abs(logical + JOINT6_STARTUP_LIMIT_RAD) < 1e-12


def test_joint6_first_ambiguous_feedback_selects_negative_storage_branch():
    wrapped_positive = 3.126426
    assert continuous_joint6_feedback(wrapped_positive) < 0.0
    assert abs(
        continuous_joint6_feedback(wrapped_positive)
        - (wrapped_positive - 2.0 * math.pi)) < 1e-12


def test_joint6_feedback_unwraps_continuously_across_signed_pi():
    first = continuous_joint6_feedback(3.126426)
    before_wrap = continuous_joint6_feedback(3.140000, first)
    after_wrap = continuous_joint6_feedback(-3.130000, before_wrap)
    assert first < before_wrap < after_wrap < 0.0
    assert after_wrap - before_wrap < 0.02


def test_joint6_standard_feedback_restores_signed_pi_after_startup():
    assert standard_joint6_feedback(0.0) == 0.0
    assert standard_joint6_feedback(math.pi) == math.pi
    assert standard_joint6_feedback(-3.20) > 3.0


def test_startup_joint6_maps_extended_negative_target_ahead_of_positive_raw():
    raw = 3.011637
    canonical = raw - 2.0 * math.pi
    mapped, waiting = startup_joint6_controller_target(
        raw, canonical, canonical)
    assert abs(mapped - raw) < 1e-12
    assert waiting is False

    mapped, waiting = startup_joint6_controller_target(
        3.05, -3.233185307, -3.20, canonical)
    assert abs(mapped - (2.0 * math.pi - 3.20)) < 1e-12
    assert mapped > 3.05
    assert waiting is False


def test_startup_joint6_uses_3_2_rad_bridge_for_measured_wrap():
    mapped, waiting = startup_joint6_controller_target(
        3.12, -3.163185307, -3.10, -3.20)
    assert mapped == JOINT6_STARTUP_WRAP_TARGET_RAD
    assert waiting is True

    mapped, waiting = startup_joint6_controller_target(
        -3.14, -3.14, -3.10, -3.20)
    assert mapped == -3.10
    assert waiting is False


def test_exact_bridge_endpoint_advances_from_storage_raw_feedback():
    raw_storage = 3.143130
    logical_storage = raw_storage - 2.0 * math.pi
    logical_bridge = (
        JOINT6_STARTUP_WRAP_TARGET_RAD - 2.0 * math.pi)
    mapped, waiting = startup_joint6_controller_target(
        raw_storage, logical_storage, logical_bridge, logical_storage)
    assert mapped == JOINT6_STARTUP_WRAP_TARGET_RAD
    assert mapped > raw_storage
    assert waiting is True

    raw_overshoot = JOINT6_STARTUP_WRAP_TARGET_RAD + 0.0007
    mapped, waiting = startup_joint6_controller_target(
        raw_overshoot,
        raw_overshoot - 2.0 * math.pi,
        logical_bridge,
        logical_bridge,
    )
    assert mapped == raw_overshoot
    assert waiting is True


def test_startup_joint6_zero_goal_never_emits_ambiguous_half_turn():
    raw_before_wrap = math.radians(120.0)
    logical_before_wrap = math.radians(-240.0)
    mapped, waiting = startup_joint6_controller_target(
        raw_before_wrap, logical_before_wrap, 0.0)
    assert mapped == 2.0 * math.pi
    assert mapped > raw_before_wrap
    assert waiting is False

    raw_after_wrap = JOINT6_STARTUP_WRAP_TARGET_RAD - 2.0 * math.pi
    mapped, waiting = startup_joint6_controller_target(
        raw_after_wrap, raw_after_wrap, 0.0, previous_target=0.0)
    assert mapped == 0.0
    assert 0.0 < mapped - raw_after_wrap < math.pi
    assert waiting is False


def test_startup_zero_does_not_depend_on_fresh_bridge_raw_snapshot():
    stale_raw = JOINT6_STARTUP_WRAP_TARGET_RAD - 0.012
    logical = stale_raw - 2.0 * math.pi
    mapped, waiting = startup_joint6_controller_target(
        stale_raw,
        logical,
        0.0,
        previous_target=(
            JOINT6_STARTUP_WRAP_TARGET_RAD - 2.0 * math.pi),
    )
    assert mapped == 2.0 * math.pi
    assert mapped > stale_raw
    assert waiting is False


def test_startup_joint6_zero_is_noop_not_positive_pi_wrap():
    mapped, waiting = startup_joint6_controller_target(0.0, 0.0, 0.0)
    assert mapped == 0.0
    assert waiting is False


def test_startup_joint6_accepts_240_degree_boundary_and_rejects_beyond_it():
    raw = math.radians(120.0)
    target = -JOINT6_STARTUP_LIMIT_RAD
    mapped, waiting = startup_joint6_controller_target(raw, target, target)
    assert abs(mapped - raw) < 1e-12
    assert waiting is False

    try:
        startup_joint6_controller_target(
            raw, target, target - math.radians(0.01))
    except ValueError as exc:
        assert 'outside [-240deg, 0]' in str(exc)
    else:
        raise AssertionError('startup target below -240 degrees was accepted')


def test_startup_joint6_can_resume_positive_motion_from_negative_midpoint():
    mapped, waiting = startup_joint6_controller_target(
        -1.25, -1.25, -0.75)
    assert mapped == -0.75
    assert mapped > -1.25
    assert waiting is False


def test_startup_joint6_rejects_negative_direction_command():
    try:
        startup_joint6_controller_target(
            -3.0, -3.0, -3.10, previous_target=-2.90)
    except ValueError as exc:
        assert 'decreased' in str(exc)
    else:
        raise AssertionError('decreasing startup J6 target was accepted')

    try:
        startup_joint6_controller_target(
            -3.0, -3.0, -3.005, previous_target=None)
    except ValueError as exc:
        assert 'behind raw feedback' in str(exc)
    else:
        raise AssertionError('small negative startup J6 command was accepted')


def test_startup_joint6_full_wrap_trace_can_only_advance_positive():
    raw_trace = [3.011637, 3.05, 3.10, 3.14, -3.13, -2.8, -1.5, -0.2]
    target_trace = [-3.271548, -3.23, -3.18, -3.10,
                    -3.00, -2.70, -1.30, 0.0]
    previous_logical = None
    previous_target = None
    for raw, target in zip(raw_trace, target_trace):
        logical = continuous_joint6_feedback(raw, previous_logical)
        mapped, _waiting = startup_joint6_controller_target(
            raw, logical, target, previous_target)
        # Before the measured signed wrap, the driver must never expose a
        # negative absolute target to the controller.
        if raw >= 0.0:
            assert mapped >= raw - 1e-4
            assert mapped >= 0.0
        else:
            # After wrapping, an increasing logical target is also an
            # increasing raw target and therefore remains positive motion.
            assert mapped >= raw - 1e-4
        if previous_target is not None:
            assert target >= previous_target
        previous_logical = logical
        previous_target = target


def test_startup_joint6_240_degree_trace_can_only_advance_positive():
    raw_trace = [
        math.radians(120.0),
        math.radians(150.0),
        math.radians(179.9),
        math.radians(-179.9),
        math.radians(-120.0),
        math.radians(-30.0),
        0.0,
    ]
    target_trace = [
        math.radians(-240.0),
        math.radians(-210.0),
        math.radians(-180.1),
        math.radians(-170.0),
        math.radians(-110.0),
        math.radians(-20.0),
        0.0,
    ]
    previous_logical = None
    previous_target = None
    for raw, target in zip(raw_trace, target_trace):
        logical = continuous_joint6_feedback(raw, previous_logical)
        mapped, _waiting = startup_joint6_controller_target(
            raw, logical, target, previous_target)
        assert mapped >= raw - 1e-4
        if previous_target is not None:
            assert target >= previous_target
        previous_logical = logical
        previous_target = target


def test_startup_direction_watchdog_accepts_positive_wrap_and_trips_reverse():
    previous = None
    continuous = 0.0
    high_water = 0.0
    for raw in [3.18, 3.20, 3.24, 4.0, 6.27, 0.01]:
        continuous, high_water, wrong = startup_joint6_direction_update(
            raw, previous, continuous, high_water)
        assert not wrong
        previous = raw

    continuous, high_water, wrong = startup_joint6_direction_update(
        -0.02, previous, continuous, high_water)
    assert wrong


def _signed_pair(first, second):
    return (
        int(first).to_bytes(4, byteorder='big', signed=True)
        + int(second).to_bytes(4, byteorder='big', signed=True))


def _complete_pairs(stamps=(9.990, 9.995, 10.000), sequences=(1, 2, 3)):
    return {
        0x2A5: (sequences[0], stamps[0], (-0.01, -0.02)),
        0x2A6: (sequences[1], stamps[1], (0.03, -0.04)),
        0x2A7: (sequences[2], stamps[2], (0.05, -0.06)),
    }


def test_decodes_signed_joint_pair_from_big_endian_can_payload():
    indices, values = decode_joint_feedback_pair(
        0x2A6, _signed_pair(-1895, 1918))
    assert indices == (2, 3)
    assert values == (
        -1895 * JOINT_FEEDBACK_RAW_TO_RAD,
        1918 * JOINT_FEEDBACK_RAW_TO_RAD,
    )


def test_joint_pair_decoder_rejects_wrong_id_and_payload_length():
    for can_id, payload in ((0x2A4, bytes(8)), (0x2A5, bytes(7))):
        try:
            decode_joint_feedback_pair(can_id, payload)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid joint CAN frame was accepted')


def test_powered_command_clamps_only_gravity_droop_axes():
    assert controller_command_position('joint2', -0.041) == 0.0
    assert controller_command_position('joint3', 0.033) == 0.0
    assert controller_command_position('joint2', 0.8) == 0.8
    assert controller_command_position('joint3', -0.7) == -0.7
    assert controller_command_position('joint1', -0.2) == -0.2


def test_coherent_can_cycle_assembles_all_six_joints_including_zero():
    pairs = _complete_pairs()
    pairs[0x2A6] = (2, 9.995, (0.0, 0.0))
    positions, sequences, reason = coherent_joint_feedback(
        pairs, None, 10.001)
    assert reason == ''
    assert sequences == (1, 2, 3)
    assert positions == [-0.01, -0.02, 0.0, 0.0, 0.05, -0.06]


def test_coherent_can_cycle_requires_all_three_pairs():
    pairs = _complete_pairs()
    del pairs[0x2A6]
    positions, sequences, reason = coherent_joint_feedback(
        pairs, None, 10.001)
    assert positions is None and sequences is None
    assert reason == 'missing joint-feedback CAN frames: 0x2A6'


def test_coherent_can_cycle_rejects_stale_and_skewed_pairs():
    _, _, stale = coherent_joint_feedback(
        _complete_pairs(stamps=(9.8, 9.995, 10.0)), None, 10.001)
    assert stale == 'stale joint-feedback CAN frames: 0x2A5'
    _, _, skewed = coherent_joint_feedback(
        _complete_pairs(stamps=(9.960, 9.995, 10.0)), None, 10.001,
        max_age=0.1, max_skew=0.03)
    assert skewed.startswith('joint-feedback CAN frame skew')


def test_coherent_can_cycle_requires_every_pair_to_advance():
    _, _, reason = coherent_joint_feedback(
        _complete_pairs(sequences=(4, 5, 3)), (1, 2, 3), 10.001)
    assert reason == (
        'waiting for a complete new joint-feedback CAN cycle: 0x2A7')


def test_coherent_can_cycle_accepts_next_complete_generation():
    positions, sequences, reason = coherent_joint_feedback(
        _complete_pairs(sequences=(4, 5, 6)), (1, 2, 3), 10.001)
    assert reason == ''
    assert positions is not None and sequences == (4, 5, 6)


def test_isolated_skew_drop_does_not_emit_false_unavailable_warning():
    reason = 'joint-feedback CAN frame skew 0.045000 > 0.030000 sec'
    assert not joint_feedback_warning_due(reason, 10.10, 10.00, 0.0)
    assert joint_feedback_warning_due(reason, 10.30, 10.00, 0.0)
    assert not joint_feedback_warning_due(reason, 10.30, 10.00, 10.0)
    assert not joint_feedback_warning_due(
        'waiting for a complete new joint-feedback CAN cycle: 0x2A7',
        10.30, 10.00, 0.0)
