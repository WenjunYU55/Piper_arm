"""Small configuration boundary for selecting the scan viewpoint policy.

The GUI does not implement either policy.  It only changes the existing ROS
parameter in ``scan_planning_params.yaml`` before the next scan stack starts.
"""

from pathlib import Path
import os
import re
import tempfile


LEGACY_POLICY = "legacy"
VOXEL_NBV_POLICY = "voxel_nbv"
SELECTABLE_POLICIES = (LEGACY_POLICY, VOXEL_NBV_POLICY)

POLICY_LABELS = {
    LEGACY_POLICY: "Legacy viewpoint heuristic",
    VOXEL_NBV_POLICY: "Voxel NBV (adaptive coverage)",
}

_POLICY_LINE = re.compile(
    r'^(?P<prefix>\s*view_selection_policy\s*:\s*)'
    r'(?P<quote>["\']?)(?P<value>[a-z0-9_]+)(?P=quote)'
    r'(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)


def default_scan_policy_path(project_root):
    """Return the source configuration loaded by the mission scan stack."""
    return Path(project_root) / (
        "piper_ros_foxy/src/piper_mobile_manipulation/"
        "config/scan_planning_params.yaml")


def _single_match(text):
    matches = list(_POLICY_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            "scan configuration must contain exactly one "
            "view_selection_policy entry (found %d)" % len(matches))
    return matches[0]


def read_scan_policy(path):
    """Read and validate the currently saved selectable policy."""
    text = Path(path).read_text(encoding="utf-8")
    policy = _single_match(text).group("value")
    if policy not in SELECTABLE_POLICIES:
        raise ValueError("unsupported saved viewpoint policy: %s" % policy)
    return policy


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

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(config_path.parent),
                prefix=".%s." % config_path.name, suffix=".tmp",
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
    return policy
