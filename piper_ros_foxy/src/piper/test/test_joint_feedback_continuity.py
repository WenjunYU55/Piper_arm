from piper.piper_ctrl_single_node import (
    JOINT_FEEDBACK_RAW_TO_RAD,
    coherent_joint_feedback,
    controller_command_position,
    decode_joint_feedback_pair,
    joint_feedback_warning_due,
)


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
