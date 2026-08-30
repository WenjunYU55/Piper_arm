"""Persist the final camera-to-target aim tolerance for the next mission."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile


DEFAULT_AIM_TOLERANCE_DEG = 5.0
MIN_AIM_TOLERANCE_DEG = 1.0
MAX_AIM_TOLERANCE_DEG = 90.0

_AIM_LINE = re.compile(
    r'^(?P<prefix>\s*final_capture_aim_tolerance_deg\s*:\s*)'
    r'(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))'
    r'(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)


@dataclass(frozen=True)
class ScanAimSettings:
    """Validated next-mission scan-view camera aim tolerance."""

    final_capture_aim_tolerance_deg: float


def default_scan_aim_path(project_root):
    """Return the executor configuration loaded by the mission stack."""
    return Path(project_root) / (
        'piper_ros_foxy/src/piper_mobile_manipulation/'
        'config/scan_execution_params.yaml')


def validate_scan_aim(value):
    """Validate later-view gate while the first lock retains its 5° cap."""
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('camera aim tolerance must be a number') from exc
    if not MIN_AIM_TOLERANCE_DEG <= tolerance <= MAX_AIM_TOLERANCE_DEG:
        raise ValueError(
            'camera aim tolerance must be between %.1f and %.1f degrees'
            % (MIN_AIM_TOLERANCE_DEG, MAX_AIM_TOLERANCE_DEG))
    return ScanAimSettings(tolerance)


def _single_match(text):
    matches = list(_AIM_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            'scan execution configuration must contain exactly one '
            'final_capture_aim_tolerance_deg entry (found %d)' % len(matches))
    return matches[0]


def read_scan_aim(path):
    """Read the authoritative next-mission value."""
    text = Path(path).read_text(encoding='utf-8')
    return validate_scan_aim(_single_match(text).group('value'))


def write_scan_aim(path, value):
    """Atomically update only the aim tolerance in the existing YAML."""
    settings = validate_scan_aim(value)
    config_path = Path(path)
    text = config_path.read_text(encoding='utf-8')
    match = _single_match(text)
    replacement = '%s%.1f%s' % (
        match.group('prefix'), settings.final_capture_aim_tolerance_deg,
        match.group('suffix'))
    updated = text[:match.start()] + replacement + text[match.end():]
    if updated == text:
        return settings
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=str(config_path.parent),
                prefix='.%s.' % config_path.name, suffix='.tmp',
                delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary_path), config_path.stat().st_mode)
        os.replace(str(temporary_path), str(config_path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return settings
