"""Characterize public entry points after the responsibility-based root cleanup."""

from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_ENTRY_POINTS = (
    "run_target_scan_mission.sh",
    "run_target_scan_gateway.sh",
    "start_gui.sh",
    "start_piper.sh",
    "piper_gui_native.py",
)


RELOCATED_SCRIPT_ENTRY_POINTS = (
    "scripts/setup/install_host_dependencies.sh",
    "scripts/setup/install_piper_can_service.sh",
    "scripts/robot/check_piper_can.sh",
    "scripts/robot/enable_piper.sh",
    "scripts/robot/disable_piper.sh",
    "scripts/calibration/calibrate_bounds.sh",
    "scripts/calibration/calibrate_joint6_zero.sh",
)


def test_public_entry_points_exist_and_are_executable():
    for relative_path in PUBLIC_ENTRY_POINTS + RELOCATED_SCRIPT_ENTRY_POINTS:
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert path.stat().st_mode & stat.S_IXUSR, relative_path


def test_relocated_scripts_resolve_the_repository_root():
    for relative_path in RELOCATED_SCRIPT_ENTRY_POINTS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        if relative_path == "scripts/robot/check_piper_can.sh":
            continue
        assert "/../.." in source, relative_path


def test_gui_root_module_is_a_compatibility_facade():
    facade = (ROOT / "piper_gui_native.py").read_text(encoding="utf-8")
    assert "from piper_gui.native_app import" in facade
    assert (ROOT / "piper_gui/native_app.py").is_file()


def test_maintenance_implementations_are_not_mixed_into_repository_root():
    moved_names = (
        "install_host_dependencies.sh",
        "install_piper_can_service.sh",
        "check_piper_can.sh",
        "enable_piper.sh",
        "disable_piper.sh",
        "calibrate_bounds.sh",
        "calibrate_joint6_zero.sh",
        "piper_calibrate_bounds.py",
        "piper_joint6_zero.py",
        "piper_gui_automation.py",
        "reset_arm.py",
        "reset_arm.sh",
        "reset_piper.py",
        "reset_piper.sh",
    )
    assert not [name for name in moved_names if (ROOT / name).exists()]
