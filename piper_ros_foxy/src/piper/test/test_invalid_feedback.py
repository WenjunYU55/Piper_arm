"""Incident regression tests: no hardware or CAN socket is opened."""

import json
import math
import threading
from types import SimpleNamespace

import can
import pytest

from piper.can_fault_trace import CanFaultTrace, TRACE_CAN_IDS
from piper.joint_state_policy import decode_joint_feedback_pair
from piper.piper_ctrl_single_node import PiperRosNode, request_piper_enable_state


@pytest.mark.parametrize('can_id', [0x2A5, 0x2A6, 0x2A7])
def test_limit_reply_payload_cannot_become_any_joint_pair(can_id):
    with pytest.raises(ValueError, match='implausible'):
        decode_joint_feedback_pair(can_id, bytes.fromhex('0105dcfa24012c00'))


@pytest.mark.parametrize('j6', [-540, -240, -180, 180, 360, 540])
def test_qualified_multiturn_feedback_remains_accepted(j6):
    raw = round(math.radians(j6) / 0.000017444)
    payload = bytes(4) + raw.to_bytes(4, 'big', signed=True)
    _, values = decode_joint_feedback_pair(0x2A7, payload)
    assert abs(values[1] - math.radians(j6)) < 0.000017444


def test_fault_latch_blocks_commands_and_survives_good_frames(tmp_path):
    trace = CanFaultTrace(tmp_path, lambda _: None)
    node = SimpleNamespace(
        _raw_joint_lock=threading.Lock(), _joint_feedback_fault='',
        _raw_joint_pairs={0x2A5: (1, 1.0, (0.0, 0.0))},
        _can_fault_trace=trace, _PiperRosNode__enable_flag=True)
    PiperRosNode.latch_joint_feedback_fault(node, 'bad payload')
    assert not node._raw_joint_pairs
    assert not PiperRosNode.GetEnableFlag(node)
    # New good samples cannot restore authority or cause a feedback publication.
    node._raw_joint_pairs[0x2A5] = (2, 2.0, (0.0, 0.0))
    assert PiperRosNode.coherent_raw_joint_positions(node)[0] is None
    PiperRosNode.joint_callback(node, object())
    PiperRosNode.pos_callback(node, object())
    node.get_logger = lambda: SimpleNamespace(error=lambda _: None)
    assert not PiperRosNode.prepare_enable_transition(node)


def test_enable_loop_stops_retrying_after_feedback_fault():
    calls = []
    piper = SimpleNamespace(EnablePiper=lambda: calls.append('enable'))
    assert not request_piper_enable_state(
        piper, True, 1.0, enable_blocked=lambda: True)
    assert calls == []


@pytest.mark.parametrize('states', [None, (True,) * 6, (False,) * 6])
def test_feedback_fault_stop_does_not_depend_on_motor_fault(monkeypatch, states):
    import piper.piper_ctrl_single_node as driver
    monkeypatch.setattr(driver, 'motor_driver_enable_states', lambda _: states)
    monkeypatch.setattr(driver, 'motor_driver_faults', lambda _: ())
    calls = []
    node = SimpleNamespace(
        piper=SimpleNamespace(DisableArm=calls.append),
        _joint_feedback_fault='INVALID_JOINT_FEEDBACK: incident',
        _motor_watchdog_disable_at=0.0,
        reset_command_cache=lambda: None,
        get_logger=lambda: SimpleNamespace(error=lambda _: None))
    PiperRosNode.fail_closed_motor_watchdog(node)
    assert calls == ([] if states == (False,) * 6 else [7])
    assert node._motor_watchdog_reason == node._joint_feedback_fault
    assert node._PiperRosNode__enable_flag is False


def test_trace_keeps_bad_id_payload_and_local_command_with_bounded_windows(tmp_path):
    trace = CanFaultTrace(tmp_path, lambda _: None, capacity=3)
    for i in range(10):
        trace.observe(can.Message(arbitration_id=0x473, data=bytes(8)), now=i)
    bad = can.Message(arbitration_id=0x2A5,
                      data=bytes.fromhex('0105dcfa24012c00'), is_extended_id=False)
    trace.observe(bad, now=10)
    trace.trigger('bad joint data', now=10)
    trace.trigger('later error must not replace first', now=11)
    trace.observe(can.Message(arbitration_id=0x151, data=bytes(8), is_rx=False), now=11)
    trace.flush(now=11)
    trace.writer.join(timeout=5)
    payload = json.loads(next(tmp_path.glob('fault-*.json')).read_text())
    assert len(payload['pre_fault']) == 3
    assert payload['pre_fault'][-1]['can_id'] == 0x2A5
    assert payload['pre_fault'][-1]['data'] == '0105dcfa24012c00'
    assert payload['post_fault'][0]['is_rx'] is False
    assert payload['fault']['reason'] == 'bad joint data'


def test_trace_disk_failure_does_not_escape_into_control(tmp_path):
    destination = tmp_path / 'file'
    destination.write_text('not a directory')
    errors = []
    trace = CanFaultTrace(destination, errors.append)
    trace.trigger('fault')
    trace.flush(force=True)
    trace.writer.join(timeout=5)
    assert len(errors) == 1


@pytest.mark.parametrize('bad', [
    can.Message(arbitration_id=0x2A5, is_extended_id=False,
                data=bytes.fromhex('0105dcfa24012c00')),
    can.Message(arbitration_id=0x2A5, is_extended_id=False, data=bytes(7)),
    can.Message(arbitration_id=0x2A5, is_extended_id=False,
                data=bytes(8), is_error_frame=True),
    can.Message(arbitration_id=0x2A5, is_extended_id=False,
                data=bytes(8), is_fd=True),
])
def test_receive_loop_latches_bad_frame_but_never_decodes_limit_reply(tmp_path, bad):
    stop = threading.Event()
    messages = iter([
        can.Message(arbitration_id=0x473, is_extended_id=False,
                    data=bytes.fromhex('0105dcfa24012c00')),
        bad,
        can.Message(arbitration_id=0x2A5, is_extended_id=False, data=bytes(8)),
    ])

    def receive(timeout):
        message = next(messages, None)
        if message is None:
            stop.set()
        return message

    node = SimpleNamespace(
        _raw_joint_stop=stop, _raw_joint_bus=SimpleNamespace(recv=receive),
        _raw_joint_lock=threading.Lock(), _joint_feedback_fault='',
        _raw_joint_pairs={}, _raw_joint_sequence=0, _raw_joint_warning_at=0,
        _can_fault_trace=CanFaultTrace(tmp_path, lambda _: None),
        get_logger=lambda: SimpleNamespace(warn=lambda _: None))
    node.latch_joint_feedback_fault = lambda reason: PiperRosNode.latch_joint_feedback_fault(
        node, reason)
    PiperRosNode._raw_joint_feedback_loop(node)
    assert node._joint_feedback_fault.startswith('INVALID_JOINT_FEEDBACK:')
    assert node._raw_joint_sequence == 1  # Only the final valid joint frame decoded.
    assert PiperRosNode.coherent_raw_joint_positions(node)[0] is None
    assert len(node._can_fault_trace.before) == 2


def test_trace_retention_keeps_only_five_fault_files(tmp_path):
    for i in range(7):
        trace = CanFaultTrace(tmp_path, lambda _: None)
        trace.trigger(str(i))
        trace.flush(force=True)
        trace.writer.join(timeout=5)
    assert len(list(tmp_path.glob('fault-*.json'))) == 5


def test_trace_includes_all_movej_position_frames_and_disable():
    assert {0x155, 0x156, 0x157, 0x471, 0x473, 0x2A5} <= set(TRACE_CAN_IDS)
