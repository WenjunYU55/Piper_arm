#!/usr/bin/env python3
"""Diagnose and, with explicit confirmation, calibrate PiPER joint 6 zero."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


JOINT_INDEX = 6
SET_ZERO_MAGIC = 0xAE
CONFIRMATION = "SET JOINT 6 ZERO"
DEFAULT_BOUNDS_PATH = (
    Path(__file__).resolve().parents[2] / "piper_joint_bounds.json"
)
CONFLICT_MARKERS = (
    "piper_ctrl_single_node",
    "start_single_piper.launch.py",
)
FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)


def _value(obj: Any, name: str, default: Any = 0) -> Any:
    return getattr(obj, name, default)


def find_conflicting_processes(
    proc_root: Path = Path("/proc"), own_pid: Optional[int] = None
) -> List[str]:
    """Return command lines that appear to own the PiPER CAN interface."""
    own_pid = os.getpid() if own_pid is None else own_pid
    matches = []
    try:
        entries: Iterable[Path] = proc_root.iterdir()
    except OSError:
        return matches

    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command and any(marker in command for marker in CONFLICT_MARKERS):
            matches.append(command)
    return sorted(set(matches))


def read_snapshot(piper: Any) -> Dict[str, Any]:
    """Read one J6 snapshot and convert documented SDK units to SI units."""
    joints = piper.GetArmJointMsgs()
    high = piper.GetArmHighSpdInfoMsgs()
    low = piper.GetArmLowSpdInfoMsgs()
    status = piper.GetArmStatus()

    joint_state = joints.joint_state
    motor_high = high.motor_6
    motor_low = low.motor_6
    foc = motor_low.foc_status
    arm_status = status.arm_status
    err = arm_status.err_status

    raw_millidegrees = int(joint_state.joint_6)
    faults = {field: bool(_value(foc, field, False)) for field in FAULT_FIELDS}
    faults["joint_6_angle_limit"] = bool(_value(err, "joint_6_angle_limit", False))
    faults["joint_6_communication"] = bool(
        _value(err, "communication_status_joint_6", False)
    )

    return {
        "timestamp_unix": time.time(),
        "feedback_hz": {
            "joints": float(_value(joints, "Hz", 0.0)),
            "high_speed": float(_value(high, "Hz", 0.0)),
            "low_speed": float(_value(low, "Hz", 0.0)),
            "status": float(_value(status, "Hz", 0.0)),
        },
        "joint6": {
            "angle_raw_millidegrees": raw_millidegrees,
            "angle_rad": math.radians(raw_millidegrees / 1000.0),
            "angle_deg": raw_millidegrees / 1000.0,
            "velocity_rad_s": float(_value(motor_high, "motor_speed", 0)) / 1000.0,
            "current_a": float(_value(motor_high, "current", 0)) / 1000.0,
            "torque_nm": float(_value(motor_high, "effort", 0)) / 1000.0,
            "motor_position_raw": int(_value(motor_high, "pos", 0)),
            "bus_current_a": float(_value(motor_low, "bus_current", 0)) / 1000.0,
            "driver_voltage_v": float(_value(motor_low, "vol", 0)) / 10.0,
            "driver_temperature_c": int(_value(motor_low, "foc_temp", 0)),
            "motor_temperature_c": int(_value(motor_low, "motor_temp", 0)),
            "enabled": bool(_value(foc, "driver_enable_status", False)),
            "faults": faults,
        },
        "arm": {
            "control_mode": int(_value(arm_status, "ctrl_mode", 0)),
            "status": int(_value(arm_status, "arm_status", 0)),
            "motion_status": int(_value(arm_status, "motion_status", 0)),
            "error_code": int(_value(arm_status, "err_code", 0)),
        },
    }


def feedback_is_live(snapshot: Dict[str, Any]) -> bool:
    rates = snapshot["feedback_hz"]
    return rates["joints"] > 0 and rates["high_speed"] > 0 and rates["low_speed"] > 0


def active_faults(snapshot: Dict[str, Any]) -> List[str]:
    return [name for name, active in snapshot["joint6"]["faults"].items() if active]


def format_snapshot(snapshot: Dict[str, Any]) -> str:
    joint = snapshot["joint6"]
    faults = active_faults(snapshot)
    return (
        f"J6 angle {joint['angle_rad']:+.5f} rad ({joint['angle_deg']:+.3f} deg) | "
        f"velocity {joint['velocity_rad_s']:+.4f} rad/s | "
        f"current {joint['current_a']:+.3f} A | torque {joint['torque_nm']:+.3f} Nm | "
        f"enabled {joint['enabled']} | faults {', '.join(faults) if faults else 'none'}"
    )


def wait_for_live_feedback(piper: Any, timeout: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = read_snapshot(piper)
        if feedback_is_live(latest):
            return latest
        time.sleep(0.1)
    if latest is None:
        raise RuntimeError("No PiPER feedback could be read")
    raise RuntimeError(
        "PiPER feedback streams are not live; check CAN power, bitrate, and interface"
    )


def wait_for_enabled_state(piper: Any, enabled: bool, timeout: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = read_snapshot(piper)
        if latest["joint6"]["enabled"] is enabled:
            return latest
        time.sleep(0.1)
    actual = None if latest is None else latest["joint6"]["enabled"]
    raise RuntimeError(f"J6 enable state did not become {enabled}; last state was {actual}")


def stationary_samples(
    piper: Any, duration: float, interval: float, max_velocity: float
) -> List[Dict[str, Any]]:
    samples = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        snapshot = read_snapshot(piper)
        samples.append(snapshot)
        if abs(snapshot["joint6"]["velocity_rad_s"]) > max_velocity:
            raise RuntimeError(
                "J6 is still moving: "
                f"{snapshot['joint6']['velocity_rad_s']:+.4f} rad/s exceeds "
                f"the {max_velocity:.4f} rad/s calibration threshold"
            )
        time.sleep(interval)
    return samples


def query_firmware(piper: Any, timeout: float = 3.0) -> str:
    piper.SearchPiperFirmwareVersion()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        firmware = piper.GetPiperFirmwareVersion()
        if isinstance(firmware, str) and firmware and firmware != str(-0x4AF):
            return firmware
        time.sleep(0.1)
    return "unavailable"


def save_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def invalidate_joint6_bounds(path: Path) -> str:
    """Mark pre-calibration J6 samples stale without deleting any measurements."""
    if not path.exists():
        return f"no bounds file found at {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        joint6 = data.get("joints", {}).get("joint6")
        if not isinstance(joint6, dict):
            return f"no joint6 record found in {path}"
        joint6["valid"] = False
        joint6["invalid_reason"] = "controller zero recalibrated; record new J6 bounds"
        joint6["invalidated_at_unix"] = time.time()
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not invalidate stale J6 bounds in {path}: {exc}") from exc
    return f"marked the old J6 bounds invalid in {path}"


def run_diagnostics(piper: Any, duration: float, interval: float) -> List[Dict[str, Any]]:
    samples = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        snapshot = read_snapshot(piper)
        samples.append(snapshot)
        print(format_snapshot(snapshot))
        time.sleep(interval)
    return samples


def calibrate_joint6(piper: Any, args: argparse.Namespace) -> Dict[str, Any]:
    if not sys.stdin.isatty():
        raise RuntimeError("Calibration requires an interactive terminal")

    print("\nCALIBRATION MODE")
    print("This permanently makes J6's current physical position its controller zero.")
    print(f"A verified write will also mark old J6 bounds invalid in {args.bounds_path}.")
    print("Support the wrist/camera, clear the workspace, and keep the emergency stop ready.")
    typed = input(f"Type {CONFIRMATION!r} to disable J6 and continue: ").strip()
    if typed != CONFIRMATION:
        raise RuntimeError("Calibration cancelled; confirmation did not match")

    piper.DisableArm(JOINT_INDEX)
    disabled = wait_for_enabled_state(piper, False, args.state_timeout)
    print("J6 is disabled. Other joints are not enabled or commanded by this tool.")
    print(format_snapshot(disabled))
    input(
        "Manually align the physical J6 neutral marks/fixture, support the payload, "
        "then press Enter: "
    )

    before_samples = stationary_samples(
        piper, args.stability_duration, args.sample_interval, args.max_stationary_velocity
    )
    before = before_samples[-1]
    faults = active_faults(before)
    if faults:
        raise RuntimeError("Refusing calibration because J6 faults are active: " + ", ".join(faults))
    if before["joint6"]["enabled"]:
        raise RuntimeError("Refusing calibration because J6 became enabled")

    print("Stable aligned position:")
    print(format_snapshot(before))
    typed = input(f"Type {CONFIRMATION!r} again to WRITE the persistent zero: ").strip()
    if typed != CONFIRMATION:
        raise RuntimeError("Calibration cancelled before persistent write")

    piper.JointConfig(joint_num=JOINT_INDEX, set_zero=SET_ZERO_MAGIC)
    time.sleep(args.write_settle_time)
    after = read_snapshot(piper)
    print("Controller feedback after zero write:")
    print(format_snapshot(after))
    if abs(after["joint6"]["angle_rad"]) > args.zero_tolerance:
        raise RuntimeError(
            "The zero command was sent, but J6 feedback is not near zero "
            f"({after['joint6']['angle_rad']:+.5f} rad). Do not enable the arm."
        )

    bounds_result = invalidate_joint6_bounds(args.bounds_path)
    print("Calibration write verified. J6 remains DISABLED.")
    print(bounds_result.capitalize() + ".")
    print("Re-record J6 bounds only after controlled motion testing.")
    return {
        "before": before,
        "after": after,
        "verified": True,
        "bounds_result": bounds_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only J6 diagnostics, with an optional guarded persistent zero calibration."
    )
    parser.add_argument("--can-port", default=os.environ.get("PIPER_CAN_PORT", "can0"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--feedback-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=3.0)
    parser.add_argument("--stability-duration", type=float, default=1.5)
    parser.add_argument("--max-stationary-velocity", type=float, default=0.02)
    parser.add_argument("--write-settle-time", type=float, default=1.0)
    parser.add_argument("--zero-tolerance", type=float, default=0.03)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--bounds-path",
        type=Path,
        default=DEFAULT_BOUNDS_PATH,
        help="Bounds JSON whose old J6 record is invalidated after verified calibration.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path (no report is written by default).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "duration",
        "sample_interval",
        "feedback_timeout",
        "state_timeout",
        "stability_duration",
        "max_stationary_velocity",
        "write_settle_time",
        "zero_tolerance",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    conflicts = find_conflicting_processes()
    if conflicts:
        print("ERROR: A PiPER ROS driver appears to be running:", file=sys.stderr)
        for command in conflicts:
            print(f"  {command}", file=sys.stderr)
        print("Stop its terminal before opening a second SDK connection.", file=sys.stderr)
        return 2

    try:
        from piper_sdk import C_PiperInterface
    except ImportError as exc:
        print(f"ERROR: piper_sdk is unavailable: {exc}", file=sys.stderr)
        return 2

    report: Dict[str, Any] = {
        "tool": "piper_joint6_zero.py",
        "can_port": args.can_port,
        "started_at_unix": time.time(),
        "calibration_requested": args.calibrate,
    }
    piper = C_PiperInterface(can_name=args.can_port)
    connected = False
    exit_code = 0
    try:
        piper.ConnectPort(piper_init=False)
        connected = True
        baseline = wait_for_live_feedback(piper, args.feedback_timeout)
        firmware = query_firmware(piper)
        report["firmware"] = firmware
        report["baseline"] = baseline
        print(f"Connected on {args.can_port}; firmware: {firmware}")
        print(format_snapshot(baseline))

        if args.calibrate:
            report["calibration"] = calibrate_joint6(piper, args)
        else:
            print("Read-only diagnostic mode; no motor or calibration commands will be sent.")
            report["samples"] = run_diagnostics(piper, args.duration, args.sample_interval)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. No automatic enable command will be sent.", file=sys.stderr)
        report["error"] = "cancelled"
        exit_code = 130
    except Exception as exc:  # Hardware/API failures should produce a clean operator message.
        print(f"ERROR: {exc}", file=sys.stderr)
        report["error"] = str(exc)
        exit_code = 1
    finally:
        report["finished_at_unix"] = time.time()
        if connected:
            try:
                piper.DisconnectPort()
            except Exception as exc:
                print(f"WARNING: SDK disconnect failed: {exc}", file=sys.stderr)
        if args.report:
            try:
                save_report(args.report, report)
                print(f"Saved report to {args.report}")
            except OSError as exc:
                print(f"ERROR: could not save report: {exc}", file=sys.stderr)
                exit_code = exit_code or 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
