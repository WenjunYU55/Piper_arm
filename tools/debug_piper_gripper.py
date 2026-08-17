#!/usr/bin/env python3
"""
Single-owner PiPER gripper diagnostic utility.

The read-only modes never call PiperInit and never transmit a CAN command.
Command modes act only through GripperCtrl; they never enable, disable, reset,
zero, or move J1-J6.  The utility refuses to start while a known PiPER command
process is present so that CAN evidence cannot be confused by two controllers.
"""

import argparse
import importlib.metadata
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from piper_sdk import C_PiperInterface
from piper_sdk.piper_msgs.msg_v2.can_id import CanIDPiper


GRIPPER_COMMAND_CAN_ID = CanIDPiper.ARM_GRIPPER_CTRL.value
GRIPPER_FEEDBACK_CAN_ID = CanIDPiper.ARM_GRIPPER_FEEDBACK.value
GRIPPER_PARAMETER_QUERY_CAN_ID = CanIDPiper.ARM_PARAM_ENQUIRY_AND_CONFIG.value
GRIPPER_PARAMETER_FEEDBACK_CAN_ID = (
    CanIDPiper.ARM_GRIPPER_TEACHING_PENDANT_PARAM_FEEDBACK.value)

GRIPPER_DISABLE_CLEAR = 0x02
GRIPPER_ENABLE = 0x01
NO_ZERO_COMMAND = 0x00
MODERATE_EFFORT_RAW = 1000  # 0.001 N m units -> 1.0 N m.
FEEDBACK_STALE_SEC = 0.50
OBSERVE_INTERVAL_SEC = 0.05
COMMAND_OBSERVE_SEC = 1.25
MOTION_OBSERVE_SEC = 3.0
POSITION_FROZEN_EPSILON_MM = 0.20
POSITION_PROGRESS_EPSILON_MM = 0.50

FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
)

KNOWN_CONTROLLER_MARKERS = (
    "piper_ctrl_single_node",
    "piper_single_ctrl",
    "start_single_piper.launch",
    "start_piper.sh",
    "target_scan_mission_node",
    "scan_viewpoint_executor_node",
    "piper_gui_native.py",
    "reset_piper.py",
    "piper_joint6_zero.py",
    "piper_calibrate_bounds.py",
    "debug_piper_gripper.py",
)


@dataclass(frozen=True)
class GripperSnapshot:
    sampled_at: float
    sdk_timestamp: float
    sdk_hz: float
    position_raw: int
    effort_raw: int
    enabled: bool
    homed: bool
    voltage_too_low: bool
    motor_overheating: bool
    driver_overcurrent: bool
    driver_overheating: bool
    sensor_status: bool
    driver_error_status: bool

    @property
    def position_mm(self) -> float:
        return self.position_raw * 0.001

    @property
    def effort_nm(self) -> float:
        return self.effort_raw * 0.001

    @property
    def age_sec(self) -> float:
        if self.sdk_timestamp <= 0.0:
            return math.inf
        return max(0.0, self.sampled_at - self.sdk_timestamp)

    @property
    def has_feedback(self) -> bool:
        return self.sdk_timestamp > 0.0

    @property
    def is_stale(self) -> bool:
        return self.has_feedback and self.age_sec > FEEDBACK_STALE_SEC

    @property
    def fault_names(self) -> Tuple[str, ...]:
        return tuple(
            field for field in FAULT_FIELDS if bool(getattr(self, field)))


@dataclass(frozen=True)
class FeedbackObservation:
    samples: Tuple[GripperSnapshot, ...]
    distinct_timestamps: int
    observed_hz: float

    @property
    def latest(self) -> Optional[GripperSnapshot]:
        return self.samples[-1] if self.samples else None

    @property
    def state(self) -> str:
        latest = self.latest
        if latest is None or not latest.has_feedback:
            return "NO FEEDBACK"
        if latest.is_stale or self.distinct_timestamps < 2:
            return "STALE FEEDBACK"
        return "TRUE"


def bool_text(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def snapshot_from_sdk(piper, wall_time: Callable[[], float] = time.time) -> GripperSnapshot:
    wrapper = piper.GetArmGripperMsgs()
    state = wrapper.gripper_state
    foc = state.foc_status
    return GripperSnapshot(
        sampled_at=float(wall_time()),
        sdk_timestamp=float(getattr(wrapper, "time_stamp", 0.0) or 0.0),
        sdk_hz=float(getattr(wrapper, "Hz", 0.0) or 0.0),
        position_raw=int(getattr(state, "grippers_angle", 0)),
        effort_raw=int(getattr(state, "grippers_effort", 0)),
        enabled=bool(getattr(foc, "driver_enable_status", False)),
        homed=bool(getattr(foc, "homing_status", False)),
        voltage_too_low=bool(getattr(foc, "voltage_too_low", False)),
        motor_overheating=bool(getattr(foc, "motor_overheating", False)),
        driver_overcurrent=bool(getattr(foc, "driver_overcurrent", False)),
        driver_overheating=bool(getattr(foc, "driver_overheating", False)),
        sensor_status=bool(getattr(foc, "sensor_status", False)),
        driver_error_status=bool(
            getattr(foc, "driver_error_status", False)),
    )


def observe_feedback(
        piper,
        duration_sec: float,
        interval_sec: float = OBSERVE_INTERVAL_SEC,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time) -> FeedbackObservation:
    started = monotonic()
    samples: List[GripperSnapshot] = []
    timestamps = set()
    while True:
        sample = snapshot_from_sdk(piper, wall_time=wall_time)
        samples.append(sample)
        if sample.sdk_timestamp > 0.0:
            timestamps.add(sample.sdk_timestamp)
        elapsed = monotonic() - started
        if elapsed >= max(0.0, float(duration_sec)):
            break
        sleeper(min(float(interval_sec), max(0.0, duration_sec - elapsed)))
    elapsed = max(1e-9, monotonic() - started)
    observed_hz = max(0, len(timestamps) - 1) / elapsed
    return FeedbackObservation(tuple(samples), len(timestamps), observed_hz)


def print_snapshot(snapshot: Optional[GripperSnapshot], feedback_state: str,
                   observed_hz: float = 0.0) -> None:
    print("feedback state: %s" % feedback_state)
    if snapshot is None or not snapshot.has_feedback:
        for label in (
                "gripper position", "gripper effort", "driver enable state",
                "low-voltage fault", "overcurrent fault",
                "driver overheating", "motor overheating", "sensor fault",
                "driver fault", "homing/zero status", "feedback timestamp",
                "feedback update rate"):
            print("%s: NO FEEDBACK" % label)
        return
    print("gripper position: %.3f mm (raw %d)" % (
        snapshot.position_mm, snapshot.position_raw))
    print("gripper effort: %.3f N m (raw %d)" % (
        snapshot.effort_nm, snapshot.effort_raw))
    print("driver enable state: %s" % bool_text(snapshot.enabled))
    print("low-voltage fault: %s" % bool_text(snapshot.voltage_too_low))
    print("overcurrent fault: %s" % bool_text(snapshot.driver_overcurrent))
    print("driver overheating: %s" % bool_text(snapshot.driver_overheating))
    print("motor overheating: %s" % bool_text(snapshot.motor_overheating))
    print("sensor fault: %s" % bool_text(snapshot.sensor_status))
    print("driver fault: %s" % bool_text(snapshot.driver_error_status))
    print("homing/zero status: %s" % bool_text(snapshot.homed))
    print("feedback timestamp: %.6f" % snapshot.sdk_timestamp)
    print("feedback age: %.3f sec%s" % (
        snapshot.age_sec, " (STALE FEEDBACK)" if snapshot.is_stale else ""))
    print("feedback update rate: SDK %.2f Hz; observed %.2f Hz" % (
        snapshot.sdk_hz, observed_hz))


def _proc_cmdline(pid: int) -> str:
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def find_competing_controllers() -> Tuple[Tuple[int, str], ...]:
    ignored_pids = set()
    current_pid = os.getpid()
    while current_pid > 1 and current_pid not in ignored_pids:
        ignored_pids.add(current_pid)
        try:
            with open("/proc/%d/stat" % current_pid, "r",
                      encoding="utf-8") as handle:
                fields = handle.read().split()
            current_pid = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                IndexError, ValueError):
            break
    matches = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in ignored_pids:
            continue
        command = _proc_cmdline(pid)
        if not command:
            continue
        if any(marker in command for marker in KNOWN_CONTROLLER_MARKERS):
            matches.append((pid, command))
    return tuple(sorted(matches))


def require_single_owner() -> None:
    matches = find_competing_controllers()
    if not matches:
        return
    print("REFUSED: another possible PiPER command process is running:",
          file=sys.stderr)
    for pid, command in matches:
        print("  PID %d: %s" % (pid, command), file=sys.stderr)
    print("Stop the production driver, mission, GUI, and calibration tools "
          "before using this direct SDK diagnostic.", file=sys.stderr)
    raise SystemExit(2)


def confirm_exact(prompt: str, required: str) -> None:
    print(prompt)
    entered = input("Type %s to continue: " % required).strip()
    if entered != required:
        raise SystemExit("Cancelled; no command was sent.")


def send_gripper_command(piper, position_raw: int, effort_raw: int,
                         status_code: int) -> None:
    print("TX CAN 0x%03X: position=%d (%.3f mm), effort=%d (%.3f N m), "
          "mode=0x%02X, set_zero=0x00" % (
              GRIPPER_COMMAND_CAN_ID,
              int(position_raw), int(position_raw) * 0.001,
              int(effort_raw), int(effort_raw) * 0.001,
              int(status_code)))
    piper.GripperCtrl(
        int(position_raw), int(effort_raw), int(status_code), NO_ZERO_COMMAND)


def safe_measured_target_raw(position_raw: int) -> int:
    """Normalize only sub-millimetre closed-position noise for a command."""
    value = int(position_raw)
    if value < -1000 or value > 100000:
        raise RuntimeError(
            "gripper feedback position is outside the documented 0-100 mm "
            "range by more than the 1 mm closed-position readback allowance")
    return min(100000, max(0, value))


def disable_command(piper, snapshot: Optional[GripperSnapshot]) -> None:
    position_raw = (
        safe_measured_target_raw(snapshot.position_raw)
        if snapshot is not None and snapshot.has_feedback else 0)
    send_gripper_command(
        piper, position_raw, MODERATE_EFFORT_RAW, GRIPPER_DISABLE_CLEAR)


def clear_enable_commands(piper, snapshot: GripperSnapshot,
                          sleeper: Callable[[float], None] = time.sleep,
                          wall_time: Callable[[], float] = time.time) -> None:
    if not snapshot.has_feedback or snapshot.is_stale:
        raise RuntimeError(
            "fresh gripper feedback is required before clear-enable")
    if not snapshot.homed:
        raise RuntimeError(
            "gripper homing/zero status is false; enabling at an assumed "
            "aperture could move the jaws")
    initial_target = safe_measured_target_raw(snapshot.position_raw)
    # Clear while disabled, then enable at the measured aperture.  No set-zero
    # code is ever emitted and no zero/closed target is invented.
    send_gripper_command(
        piper, initial_target, MODERATE_EFFORT_RAW,
        GRIPPER_DISABLE_CLEAR)
    sleeper(0.20)
    refreshed = snapshot_from_sdk(piper, wall_time=wall_time)
    if not refreshed.has_feedback or refreshed.is_stale:
        raise RuntimeError(
            "feedback became unavailable after fault-clear; gripper remains "
            "disabled")
    refreshed_target = safe_measured_target_raw(refreshed.position_raw)
    send_gripper_command(
        piper, refreshed_target, MODERATE_EFFORT_RAW, GRIPPER_ENABLE)


def run_status(piper) -> int:
    observation = observe_feedback(piper, 1.25)
    print_snapshot(
        observation.latest, observation.state, observation.observed_hz)
    return 0 if observation.state == "TRUE" else 1


def run_disable(piper) -> int:
    before = observe_feedback(piper, 0.75)
    print("\nBEFORE")
    print_snapshot(before.latest, before.state, before.observed_hz)
    confirm_exact(
        "This sends only the documented gripper disable-and-clear command. "
        "It does not disable J1-J6, but it can release a held object.",
        "DISABLE GRIPPER")
    disable_command(piper, before.latest)
    after = observe_feedback(piper, COMMAND_OBSERVE_SEC)
    print("\nAFTER")
    print_snapshot(after.latest, after.state, after.observed_hz)
    before_enabled = bool(before.latest.enabled) if before.latest else False
    after_disabled = bool(after.latest and not after.latest.enabled)
    print("enable flag changed enabled -> disabled: %s" % bool_text(
        before_enabled and after_disabled))
    return 0 if after.state == "TRUE" and after_disabled else 1


def run_clear_enable(piper) -> int:
    before = observe_feedback(piper, 0.75)
    print("\nBEFORE")
    print_snapshot(before.latest, before.state, before.observed_hz)
    if before.state != "TRUE" or before.latest is None:
        print("REFUSED: fresh updating feedback is required; no command sent.")
        return 1
    if not before.latest.homed:
        print("REFUSED: gripper feedback says homing/zero status is FALSE. "
              "The diagnostic cannot safely hold an untrusted aperture and "
              "will not set zero automatically; no command sent.")
        return 1
    confirm_exact(
        "This clears gripper faults and enables position control at the "
        "currently measured aperture. It never sets zero and never requests "
        "fully closed.",
        "CLEAR ENABLE GRIPPER")
    clear_enable_commands(piper, before.latest)
    after = observe_feedback(piper, COMMAND_OBSERVE_SEC)
    print("\nAFTER")
    print_snapshot(after.latest, after.state, after.observed_hz)
    held_current = bool(
        after.latest
        and abs(after.latest.position_mm - before.latest.position_mm) <= 2.0)
    print("driver reports enabled: %s" % bool_text(
        bool(after.latest and after.latest.enabled)))
    print("position remained within 2 mm of measured aperture: %s" %
          bool_text(held_current))
    return 0 if (
        after.state == "TRUE"
        and after.latest is not None
        and after.latest.enabled
        and held_current
        and not after.latest.fault_names) else 1


def _print_motion_sample(sample: GripperSnapshot, target_mm: float) -> None:
    faults = ",".join(sample.fault_names) or "none"
    print("%.6f target=%5.1f mm actual=%7.3f mm effort=%6.3f N m "
          "enabled=%s age=%5.3f sec faults=%s" % (
              sample.sampled_at, target_mm, sample.position_mm,
              sample.effort_nm, bool_text(sample.enabled), sample.age_sec,
              faults))


def execute_motion_target(piper, target_mm: float) -> bool:
    before = snapshot_from_sdk(piper)
    if not before.has_feedback or before.is_stale or not before.enabled:
        print("REFUSED: motion target requires fresh enabled feedback.")
        return False
    target_raw = int(round(float(target_mm) * 1000.0))
    print("\nREQUEST %.1f mm" % target_mm)
    send_gripper_command(
        piper, target_raw, MODERATE_EFFORT_RAW, GRIPPER_ENABLE)
    started = time.monotonic()
    samples = []
    while time.monotonic() - started < MOTION_OBSERVE_SEC:
        sample = snapshot_from_sdk(piper)
        samples.append(sample)
        _print_motion_sample(sample, target_mm)
        if sample.fault_names or not sample.enabled or sample.is_stale:
            print("ABORTING target observation because authority/freshness/fault "
                  "evidence was lost.")
            return False
        time.sleep(0.10)
    if not samples:
        return False
    initial_error = abs(before.position_mm - target_mm)
    final_error = abs(samples[-1].position_mm - target_mm)
    span = max(item.position_mm for item in samples) - min(
        item.position_mm for item in samples)
    progressed = final_error <= initial_error - POSITION_PROGRESS_EPSILON_MM
    frozen = initial_error > 2.0 and span < POSITION_FROZEN_EPSILON_MM
    print("moved toward target: %s" % bool_text(progressed))
    print("feedback frozen while target differed: %s" % bool_text(frozen))
    print("final error: %.3f mm" % final_error)
    return progressed and not frozen


def run_motion_test(piper) -> int:
    before = observe_feedback(piper, 0.75)
    print("\nBEFORE")
    print_snapshot(before.latest, before.state, before.observed_hz)
    if before.state != "TRUE" or before.latest is None:
        print("REFUSED: fresh updating feedback is required; no command sent.")
        return 1
    if not before.latest.homed:
        print("REFUSED: gripper feedback says homing/zero status is FALSE. "
              "The diagnostic will not set zero automatically, and position "
              "motion is not safe until zeroing is separately supervised.")
        return 1
    confirm_exact(
        "This test will enable only the gripper at its current aperture, then "
        "request 40 mm and 15 mm at 1.0 N m. Keep fingers and objects clear. "
        "J1-J6 are never commanded.",
        "RUN GRIPPER MOTION TEST")
    commands_started = False
    success = False
    try:
        clear_enable_commands(piper, before.latest)
        commands_started = True
        enabled = observe_feedback(piper, COMMAND_OBSERVE_SEC)
        if (enabled.state != "TRUE" or enabled.latest is None
                or not enabled.latest.enabled or enabled.latest.fault_names):
            print("REFUSED: clear-enable did not produce healthy enabled feedback.")
            return 1
        opened = execute_motion_target(piper, 40.0)
        closed = execute_motion_target(piper, 15.0)
        success = opened and closed
        return 0 if success else 1
    finally:
        if commands_started:
            try:
                latest = snapshot_from_sdk(piper)
                disable_command(piper, latest)
                print("Final gripper disable-and-clear command sent. J1-J6 were "
                      "not disabled or commanded.")
            except Exception as error:
                print("WARNING: final gripper disable command failed: %s" % error,
                      file=sys.stderr)


def run_monitor(piper) -> int:
    print("Receive-only monitor active. No CAN command will be sent. Ctrl+C exits.")
    previous = None
    try:
        while True:
            sample = snapshot_from_sdk(piper)
            state = (
                "NO FEEDBACK" if not sample.has_feedback
                else "STALE FEEDBACK" if sample.is_stale else "TRUE")
            note = ""
            if previous is not None and sample.has_feedback:
                delta = sample.position_mm - previous.position_mm
                if (abs(delta) >= POSITION_FROZEN_EPSILON_MM
                        and not sample.enabled
                        and abs(sample.effort_nm) <= 0.05):
                    note = " PASSIVE-MOVEMENT-LIKELY delta=%+.3fmm" % delta
            faults = ",".join(sample.fault_names) or "none"
            print("%.6f state=%s pos=%7.3fmm effort=%6.3fNm enabled=%s "
                  "homed=%s age=%5.3fs faults=%s%s" % (
                      time.time(), state, sample.position_mm, sample.effort_nm,
                      bool_text(sample.enabled), bool_text(sample.homed),
                      sample.age_sec, faults, note), flush=True)
            previous = sample
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nMonitor stopped; no disable, enable, reset, zero, or motion "
              "command was sent.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-owner, no-zero PiPER gripper diagnostic")
    parser.add_argument(
        "mode", choices=(
            "status", "disable", "clear-enable", "motion-test", "monitor"))
    parser.add_argument("--can", default="can0", help="SocketCAN interface")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    require_single_owner()
    try:
        sdk_version = importlib.metadata.version("piper-sdk")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "unknown installed version"
    print("PiPER SDK: %s" % sdk_version)
    print("CAN interface: %s" % args.can)
    print("verified SDK CAN IDs: command=0x%03X feedback=0x%03X "
          "parameter-query=0x%03X parameter-feedback=0x%03X" % (
              GRIPPER_COMMAND_CAN_ID, GRIPPER_FEEDBACK_CAN_ID,
              GRIPPER_PARAMETER_QUERY_CAN_ID,
              GRIPPER_PARAMETER_FEEDBACK_CAN_ID))
    piper = C_PiperInterface(can_name=args.can)
    try:
        # piper_init=False is essential: status and monitor must not transmit
        # SDK initialization/query commands, and command modes should emit only
        # the GripperCtrl frames explicitly printed above.
        piper.ConnectPort(piper_init=False)
        runners = {
            "status": run_status,
            "disable": run_disable,
            "clear-enable": run_clear_enable,
            "motion-test": run_motion_test,
            "monitor": run_monitor,
        }
        return int(runners[args.mode](piper))
    finally:
        piper.DisconnectPort(thread_timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
