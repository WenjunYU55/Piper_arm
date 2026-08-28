#!/usr/bin/env python3

"""Joint-six zeroing and driver convention regressions."""

import math
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tools.calibration import piper_joint6_zero as joint6


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


class FakePiper:
    def GetArmJointMsgs(self):
        return ns(Hz=200.0, joint_state=ns(joint_6=90000))

    def GetArmHighSpdInfoMsgs(self):
        return ns(
            Hz=200.0,
            motor_6=ns(motor_speed=15, current=1250, effort=420, pos=123),
        )

    def GetArmLowSpdInfoMsgs(self):
        foc = ns(
            voltage_too_low=False,
            motor_overheating=False,
            driver_overcurrent=False,
            driver_overheating=False,
            collision_status=False,
            driver_error_status=False,
            driver_enable_status=True,
            stall_status=False,
        )
        return ns(
            Hz=50.0,
            motor_6=ns(
                foc_status=foc,
                bus_current=900,
                vol=240,
                foc_temp=35,
                motor_temp=37,
            ),
        )

    def GetArmStatus(self):
        err = ns(joint_6_angle_limit=False, communication_status_joint_6=False)
        return ns(
            Hz=100.0,
            arm_status=ns(
                err_status=err,
                ctrl_mode=1,
                arm_status=0,
                motion_status=0,
                err_code=0,
            ),
        )


class Joint6ZeroTest(unittest.TestCase):
    def test_snapshot_converts_documented_units(self):
        snapshot = joint6.read_snapshot(FakePiper())
        data = snapshot["joint6"]
        self.assertAlmostEqual(data["angle_rad"], math.pi / 2)
        self.assertAlmostEqual(data["velocity_rad_s"], 0.015)
        self.assertAlmostEqual(data["current_a"], 1.25)
        self.assertAlmostEqual(data["torque_nm"], 0.42)
        self.assertAlmostEqual(data["driver_voltage_v"], 24.0)
        self.assertTrue(data["enabled"])
        self.assertEqual(joint6.active_faults(snapshot), [])
        self.assertTrue(joint6.feedback_is_live(snapshot))

    def test_active_faults_includes_driver_and_joint_faults(self):
        piper = FakePiper()
        original_low = piper.GetArmLowSpdInfoMsgs

        def faulted_low():
            result = original_low()
            result.motor_6.foc_status.stall_status = True
            return result

        piper.GetArmLowSpdInfoMsgs = faulted_low
        snapshot = joint6.read_snapshot(piper)
        self.assertEqual(joint6.active_faults(snapshot), ["stall_status"])

    def test_conflicting_process_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "100").mkdir()
            (root / "100" / "cmdline").write_bytes(
                b"python3\0/opt/ros/piper_ctrl_single_node\0"
            )
            (root / "101").mkdir()
            (root / "101" / "cmdline").write_bytes(b"python3\0unrelated.py\0")
            matches = joint6.find_conflicting_processes(root, own_pid=999)
        self.assertEqual(matches, ["python3 /opt/ros/piper_ctrl_single_node"])

    def test_validate_args_rejects_nonpositive_values(self):
        args = joint6.build_parser().parse_args(["--duration", "0"])
        with self.assertRaisesRegex(ValueError, "duration"):
            joint6.validate_args(args)

    def test_invalidate_joint6_bounds_preserves_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounds.json"
            path.write_text(
                '{"joints":{"joint6":{"min":-3.1,"max":5.4,'
                '"samples":{"x":1}}}}',
                encoding="utf-8",
            )
            result = joint6.invalidate_joint6_bounds(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("marked", result)
        self.assertFalse(data["joints"]["joint6"]["valid"])
        self.assertEqual(data["joints"]["joint6"]["samples"], {"x": 1})


if __name__ == "__main__":
    unittest.main()
