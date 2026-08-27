import importlib.util
import pathlib
from types import SimpleNamespace

import pytest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tools"
    / "debug_piper_gripper.py"
)
SPEC = importlib.util.spec_from_file_location("debug_piper_gripper", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def feedback_wrapper(timestamp=100.0, hz=20.0, position=25000, effort=500,
                     enabled=False, homed=True, **flags):
    foc = SimpleNamespace(
        driver_enable_status=enabled,
        homing_status=homed,
        voltage_too_low=flags.get("voltage_too_low", False),
        motor_overheating=flags.get("motor_overheating", False),
        driver_overcurrent=flags.get("driver_overcurrent", False),
        driver_overheating=flags.get("driver_overheating", False),
        sensor_status=flags.get("sensor_status", False),
        driver_error_status=flags.get("driver_error_status", False),
    )
    return SimpleNamespace(
        time_stamp=timestamp,
        Hz=hz,
        gripper_state=SimpleNamespace(
            grippers_angle=position,
            grippers_effort=effort,
            foc_status=foc,
        ),
    )


class FakePiper:
    def __init__(self, wrappers):
        self.wrappers = list(wrappers)
        self.index = 0
        self.commands = []

    def GetArmGripperMsgs(self):
        value = self.wrappers[min(self.index, len(self.wrappers) - 1)]
        self.index += 1
        return value

    def GripperCtrl(self, *args):
        self.commands.append(args)


def test_snapshot_converts_installed_sdk_units_and_status():
    piper = FakePiper([feedback_wrapper(
        timestamp=100.0, position=40000, effort=1000, enabled=True,
        driver_overcurrent=True)])

    snapshot = MODULE.snapshot_from_sdk(piper, wall_time=lambda: 100.1)

    assert snapshot.position_mm == 40.0
    assert snapshot.effort_nm == 1.0
    assert snapshot.enabled is True
    assert snapshot.is_stale is False
    assert snapshot.fault_names == ("driver_overcurrent",)


def test_no_feedback_and_stale_feedback_are_distinct():
    no_feedback = MODULE.snapshot_from_sdk(
        FakePiper([feedback_wrapper(timestamp=0.0)]),
        wall_time=lambda: 100.0)
    stale = MODULE.snapshot_from_sdk(
        FakePiper([feedback_wrapper(timestamp=90.0)]),
        wall_time=lambda: 100.0)

    assert no_feedback.has_feedback is False
    assert no_feedback.is_stale is False
    assert stale.has_feedback is True
    assert stale.is_stale is True


def test_disable_uses_documented_disable_clear_without_zeroing():
    piper = FakePiper([feedback_wrapper()])
    snapshot = MODULE.snapshot_from_sdk(piper, wall_time=lambda: 100.1)

    MODULE.disable_command(piper, snapshot)

    assert piper.commands == [(25000, 1000, 0x02, 0x00)]


def test_tiny_negative_closed_feedback_is_normalized_without_setting_zero():
    piper = FakePiper([feedback_wrapper(position=-70)])
    snapshot = MODULE.snapshot_from_sdk(piper, wall_time=lambda: 100.1)

    MODULE.disable_command(piper, snapshot)

    assert piper.commands == [(0, 1000, 0x02, 0x00)]


def test_clear_enable_preserves_measured_position_and_never_sets_zero():
    piper = FakePiper([
        feedback_wrapper(timestamp=100.0, position=32000),
        feedback_wrapper(timestamp=100.1, position=31500),
    ])
    snapshot = MODULE.snapshot_from_sdk(piper, wall_time=lambda: 100.1)

    MODULE.clear_enable_commands(
        piper, snapshot, sleeper=lambda _value: None,
        wall_time=lambda: 100.2)

    assert piper.commands == [
        (32000, 1000, 0x02, 0x00),
        (31500, 1000, 0x01, 0x00),
    ]


def test_clear_enable_refuses_unhomed_feedback_without_sending_command():
    piper = FakePiper([feedback_wrapper(homed=False)])
    snapshot = MODULE.snapshot_from_sdk(piper, wall_time=lambda: 100.1)

    with pytest.raises(RuntimeError, match="homing/zero status is false"):
        MODULE.clear_enable_commands(
            piper, snapshot, sleeper=lambda _value: None,
            wall_time=lambda: 100.2)

    assert piper.commands == []


def test_can_ids_come_from_installed_sdk_protocol_definition():
    assert MODULE.GRIPPER_COMMAND_CAN_ID == 0x159
    assert MODULE.GRIPPER_FEEDBACK_CAN_ID == 0x2A8
    assert MODULE.GRIPPER_PARAMETER_QUERY_CAN_ID == 0x477
    assert MODULE.GRIPPER_PARAMETER_FEEDBACK_CAN_ID == 0x47E
