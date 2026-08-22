"""Validated L515 RGB profile saved for the next mission camera startup."""

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile


DEFAULT_CAMERA_PROFILE = (640, 480, 30)
CAMERA_PROFILE_FPS = {
    (640, 480): (15, 30),
    (960, 540): (15, 30),
    (1280, 720): (15, 30),
    (1920, 1080): (15, 30),
}


@dataclass(frozen=True)
class CameraProfile:
    """One admitted L515 RGB stream profile."""

    width: int
    height: int
    fps: int


def default_camera_profile_path(project_root):
    """Return the profile file read by a newly started camera process."""
    return Path(project_root) / "L515_camera/rgb_profile.conf"


def validate_camera_profile(width, height, fps):
    """Return an admitted integer profile or reject it."""
    try:
        profile = CameraProfile(int(width), int(height), int(fps))
    except (TypeError, ValueError) as exc:
        raise ValueError("camera profile must contain integer values") from exc
    allowed_fps = CAMERA_PROFILE_FPS.get((profile.width, profile.height))
    if allowed_fps is None or profile.fps not in allowed_fps:
        raise ValueError(
            "unsupported L515 RGB profile: %dx%d@%d"
            % (profile.width, profile.height, profile.fps))
    return profile


def read_camera_profile(path):
    """Read and validate the saved width,height,FPS record."""
    try:
        values = Path(path).read_text(encoding="utf-8").strip().split(",")
    except OSError:
        raise
    if len(values) != 3:
        raise ValueError("camera profile must be width,height,fps")
    return validate_camera_profile(*values)


def write_camera_profile(path, width, height, fps):
    """Atomically save one validated profile for the next camera startup."""
    profile = validate_camera_profile(width, height, fps)
    config_path = Path(path)
    text = "%d,%d,%d\n" % (profile.width, profile.height, profile.fps)
    if config_path.exists() and config_path.read_text(encoding="utf-8") == text:
        return profile
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(config_path.parent),
                prefix=".%s." % config_path.name, suffix=".tmp",
                delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if config_path.exists():
            os.chmod(str(temporary_path), config_path.stat().st_mode)
        os.replace(str(temporary_path), str(config_path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return profile
