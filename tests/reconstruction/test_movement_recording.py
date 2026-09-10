"""Command-free movement-window selection and bounded pre-roll."""

from piper_mobile_manipulation.movement_recording import (
    MovementWindow, PreRoll, movement_intervals)


def test_one_second_shoulders_merge_and_repeat_status_does_not_extend_motion():
    assert movement_intervals([(3, 'MOVING'), (4, 'MOVING'), (5, 'SETTLING'),
                               (6, 'MOVING'), (7, 'CAPTURING')], 10, 1) == [(2, 8)]
    assert movement_intervals([(3, 'MOVING')], 10, 1) == [(2, 10)]
    window = MovementWindow()
    assert window.update('MOVING', 3)
    assert not window.update('MOVING', 4)
    window.update('SETTLING', 5)
    assert window.active(6)
    assert not window.active(6.01)


def test_preroll_expires_caps_bytes_and_does_not_duplicate_saved_frames():
    buffer = PreRoll(maximum_bytes=4)
    buffer.add(0., ('rgb', b'aa', 0))
    buffer.add(.5, ('depth', b'bb', 1))
    buffer.add(.9, ('rgb', b'cc', 2))
    assert buffer.overflow == 1
    assert buffer.take(1.6) == [('rgb', b'cc', 2)]
    assert buffer.take(1.6) == []
    assert buffer.bytes == 0
