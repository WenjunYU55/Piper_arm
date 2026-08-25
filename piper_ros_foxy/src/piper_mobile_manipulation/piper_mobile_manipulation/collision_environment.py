"""
Persistent next-mission support-floor selection.

The installed PiPER/L515/Bunker geometry is invariant.  This small boundary
selects only the support plane used when the next mission-owned planning stack
starts.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile


TABLETOP_FLOOR = 'tabletop'
GROUND_FLOOR = 'ground'
SELECTABLE_FLOOR_PROFILES = (TABLETOP_FLOOR, GROUND_FLOOR)
FLOOR_HEIGHTS_M = {
    TABLETOP_FLOOR: 0.005,
    GROUND_FLOOR: -0.466,
}
FLOOR_PROFILE_LABELS = {
    TABLETOP_FLOOR: 'Tabletop floor (z = +0.005 m)',
    GROUND_FLOOR: 'Tracked-robot ground (z = -0.466 m)',
}

_FLOOR_PROFILE_LINE = re.compile(
    r'^(?P<prefix>\s*floor_profile\s*:\s*)'
    r'(?P<quote>["\']?)(?P<value>[a-z0-9_]+)(?P=quote)'
    r'(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)


@dataclass(frozen=True)
class CollisionEnvironment:
    """Immutable floor choice captured before a mission starts children."""

    floor_profile: str
    floor_z_m: float


def default_collision_environment_path(project_root):
    """Return the source configuration shared by the GUI and coordinator."""
    return Path(project_root) / (
        'piper_ros_foxy/src/piper_mobile_manipulation/'
        'config/collision_environment.yaml')


def _validated_profile(value):
    profile = str(value).strip().lower()
    if profile not in SELECTABLE_FLOOR_PROFILES:
        raise ValueError(
            'floor profile must be exactly tabletop or ground')
    return profile


def _single_match(text):
    matches = list(_FLOOR_PROFILE_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            'collision environment must contain exactly one floor_profile '
            'entry (found %d)' % len(matches))
    return matches[0]


def read_collision_environment(path):
    """Read and validate the floor profile saved for the next mission."""
    text = Path(path).read_text(encoding='utf-8')
    profile = _validated_profile(_single_match(text).group('value'))
    return CollisionEnvironment(profile, FLOOR_HEIGHTS_M[profile])


def write_collision_environment(path, floor_profile):
    """Atomically change only the next-mission floor profile."""
    profile = _validated_profile(floor_profile)
    config_path = Path(path)
    text = config_path.read_text(encoding='utf-8')
    match = _single_match(text)
    replacement = '%s"%s"%s' % (
        match.group('prefix'), profile, match.group('suffix'))
    updated = text[:match.start()] + replacement + text[match.end():]
    if updated == text:
        return CollisionEnvironment(profile, FLOOR_HEIGHTS_M[profile])

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
    return CollisionEnvironment(profile, FLOOR_HEIGHTS_M[profile])
