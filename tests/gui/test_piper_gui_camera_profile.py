from pathlib import Path

import pytest

from piper_gui.camera_profile import (
    CameraProfile,
    read_camera_profile,
    validate_camera_profile,
    write_camera_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_camera_profile_round_trip_is_atomic_and_idempotent(tmp_path):
    path = Path(tmp_path) / "rgb_profile.conf"
    saved = write_camera_profile(path, 1280, 720, 15)
    assert saved == CameraProfile(1280, 720, 15)
    assert read_camera_profile(path) == saved
    before = path.read_bytes()
    write_camera_profile(path, 1280, 720, 15)
    assert path.read_bytes() == before


@pytest.mark.parametrize("profile", [
    (640, 480, 60),
    (320, 240, 30),
    (1280, 720, 20),
    ("bad", 480, 30),
])
def test_camera_profile_rejects_unqualified_modes(profile):
    with pytest.raises(ValueError):
        validate_camera_profile(*profile)


def test_camera_profile_refuses_malformed_saved_file(tmp_path):
    path = Path(tmp_path) / "rgb_profile.conf"
    path.write_text("640x480@30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="width,height,fps"):
        read_camera_profile(path)


@pytest.mark.parametrize("script_name", [
    "start_l515_camera.sh",
    "start_l515_camera_low_bandwidth.sh",
])
def test_camera_launch_reads_saved_profile_and_retains_environment_override(
        script_name):
    source = (PROJECT_ROOT / "L515_camera" /
              script_name).read_text(encoding="utf-8")
    assert "rgb_profile.conf" in source
    assert 'PIPER_CAMERA_COLOR_WIDTH:-$saved_color_width' in source
    assert 'PIPER_CAMERA_COLOR_HEIGHT:-$saved_color_height' in source
    assert 'PIPER_CAMERA_COLOR_FPS:-$saved_color_fps' in source
    assert 'rgb_camera.profile:="$rgb_profile"' in source
