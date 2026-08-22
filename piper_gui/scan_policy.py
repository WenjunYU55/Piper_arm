"""
Small configuration boundary for selecting scan viewpoint generation.

The GUI does not implement the policies. It only changes existing ROS
parameters in ``scan_planning_params.yaml`` before the next scan stack starts.
"""

from pathlib import Path
from dataclasses import dataclass
import os
import re
import tempfile


LEGACY_POLICY = "legacy"
VOXEL_NBV_POLICY = "voxel_nbv"
RAY_NBV_POLICY = "ray_nbv"
SELECTABLE_POLICIES = (LEGACY_POLICY, VOXEL_NBV_POLICY, RAY_NBV_POLICY)

TARGET_SECTOR_REGION = "target_sector"
UPPER_HEMISPHERE_REGION = "upper_hemisphere"
FULL_SPHERE_REGION = "full_sphere"
SELECTABLE_RAY_REGIONS = (
    TARGET_SECTOR_REGION,
    UPPER_HEMISPHERE_REGION,
    FULL_SPHERE_REGION,
)
MIN_RAY_COUNT = 1
MAX_RAY_COUNT = 1000

POLICY_LABELS = {
    LEGACY_POLICY: "Legacy viewpoint heuristic",
    VOXEL_NBV_POLICY: "Voxel NBV (exact point lattice)",
    RAY_NBV_POLICY: "Voxel NBV (frozen rays)",
}

RAY_REGION_LABELS = {
    TARGET_SECTOR_REGION: "180° target-facing sector",
    UPPER_HEMISPHERE_REGION: "360° upper hemisphere",
    FULL_SPHERE_REGION: "360° full sphere",
}

_POLICY_LINE = re.compile(
    r'^(?P<prefix>\s*view_selection_policy\s*:\s*)'
    r'(?P<quote>["\']?)(?P<value>[a-z0-9_]+)(?P=quote)'
    r'(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)

_RAY_REGION_LINE = re.compile(
    r'^(?P<prefix>\s*ray_sampling_region\s*:\s*)'
    r'(?P<quote>["\']?)(?P<value>[a-z0-9_]+)(?P=quote)'
    r'(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)

_RAY_COUNT_LINE = re.compile(
    r'^(?P<prefix>\s*ray_count\s*:\s*)'
    r'(?P<value>[0-9]+)(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)


@dataclass(frozen=True)
class ScanPolicySettings:
    """Saved planner choices applied when the next scan stack starts."""

    policy: str
    ray_region: str
    ray_count: int


def default_scan_policy_path(project_root):
    """Return the source configuration loaded by the mission scan stack."""
    return Path(project_root) / (
        "piper_ros_foxy/src/piper_mobile_manipulation/"
        "config/scan_planning_params.yaml")


def _single_match(text, pattern=_POLICY_LINE, setting='view_selection_policy'):
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            "scan configuration must contain exactly one %s entry (found %d)"
            % (setting, len(matches)))
    return matches[0]


def _validated_settings(policy, ray_region, ray_count):
    selected_policy = str(policy)
    selected_region = str(ray_region)
    count = int(ray_count)
    if selected_policy not in SELECTABLE_POLICIES:
        raise ValueError("unsupported viewpoint policy: %s" % selected_policy)
    if selected_region not in SELECTABLE_RAY_REGIONS:
        raise ValueError("unsupported ray region: %s" % selected_region)
    if count < MIN_RAY_COUNT or count > MAX_RAY_COUNT:
        raise ValueError(
            "ray count must be between %d and %d"
            % (MIN_RAY_COUNT, MAX_RAY_COUNT))
    return ScanPolicySettings(selected_policy, selected_region, count)


def _atomic_write(config_path, text):
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
        os.chmod(str(temporary_path), config_path.stat().st_mode)
        os.replace(str(temporary_path), str(config_path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_scan_policy(path):
    """Read and validate the currently saved selectable policy."""
    text = Path(path).read_text(encoding="utf-8")
    policy = _single_match(text).group("value")
    if policy not in SELECTABLE_POLICIES:
        raise ValueError("unsupported saved viewpoint policy: %s" % policy)
    return policy


def read_scan_settings(path):
    """Read and validate the complete GUI-owned planner selection."""
    text = Path(path).read_text(encoding="utf-8")
    policy = _single_match(text).group("value")
    region = _single_match(
        text, _RAY_REGION_LINE, 'ray_sampling_region').group("value")
    count = int(_single_match(
        text, _RAY_COUNT_LINE, 'ray_count').group("value"))
    return _validated_settings(policy, region, count)


def write_scan_policy(path, policy):
    """Atomically change only the existing policy value and preserve the YAML."""
    if policy not in SELECTABLE_POLICIES:
        raise ValueError("unsupported viewpoint policy: %s" % policy)

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    match = _single_match(text)
    replacement = '%s"%s"%s' % (
        match.group("prefix"), policy, match.group("suffix"))
    updated = text[:match.start()] + replacement + text[match.end():]
    if updated == text:
        return policy

    _atomic_write(config_path, updated)
    return policy


def write_scan_settings(path, policy, ray_region, ray_count):
    """Atomically persist policy and ray geometry for the next mission."""
    settings = _validated_settings(policy, ray_region, ray_count)
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    policy_match = _single_match(text)
    region_match = _single_match(
        text, _RAY_REGION_LINE, 'ray_sampling_region')
    count_match = _single_match(text, _RAY_COUNT_LINE, 'ray_count')

    replacements = [
        (policy_match.start(), policy_match.end(), '%s"%s"%s' % (
            policy_match.group('prefix'), settings.policy,
            policy_match.group('suffix'))),
        (region_match.start(), region_match.end(), '%s"%s"%s' % (
            region_match.group('prefix'), settings.ray_region,
            region_match.group('suffix'))),
        (count_match.start(), count_match.end(), '%s%d%s' % (
            count_match.group('prefix'), settings.ray_count,
            count_match.group('suffix'))),
    ]
    updated = text
    for start, end, replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    if updated != text:
        _atomic_write(config_path, updated)
    return settings
