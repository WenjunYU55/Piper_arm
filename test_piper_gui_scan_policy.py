from pathlib import Path

import pytest

from piper_gui.scan_policy import (
    LEGACY_POLICY,
    VOXEL_NBV_POLICY,
    read_scan_policy,
    write_scan_policy,
)


def _config(tmp_path, policy=LEGACY_POLICY):
    path = Path(tmp_path) / "scan_planning_params.yaml"
    path.write_text(
        "/**:\n  ros__parameters:\n"
        "    max_viewpoints: 25\n"
        '    view_selection_policy: "%s"  # retained comment\n'
        "    dry_run: true\n" % policy,
        encoding="utf-8",
    )
    return path


def test_policy_switch_changes_only_the_existing_value(tmp_path):
    path = _config(tmp_path)
    before = path.read_text(encoding="utf-8")

    assert read_scan_policy(path) == LEGACY_POLICY
    assert write_scan_policy(path, VOXEL_NBV_POLICY) == VOXEL_NBV_POLICY
    assert read_scan_policy(path) == VOXEL_NBV_POLICY

    after = path.read_text(encoding="utf-8")
    assert after == before.replace('"legacy"', '"voxel_nbv"')
    assert "# retained comment" in after


def test_writing_selected_policy_is_idempotent(tmp_path):
    path = _config(tmp_path, VOXEL_NBV_POLICY)
    before = path.read_bytes()
    write_scan_policy(path, VOXEL_NBV_POLICY)
    assert path.read_bytes() == before


@pytest.mark.parametrize("policy", ["voxel_nbv_shadow", "unknown", ""])
def test_gui_rejects_non_selectable_policy(tmp_path, policy):
    path = _config(tmp_path)
    with pytest.raises(ValueError, match="unsupported viewpoint policy"):
        write_scan_policy(path, policy)


@pytest.mark.parametrize("content", [
    "/**:\n  ros__parameters:\n    dry_run: true\n",
    ("/**:\n  ros__parameters:\n"
     '    view_selection_policy: "legacy"\n'
     '    view_selection_policy: "voxel_nbv"\n'),
])
def test_gui_refuses_ambiguous_or_missing_policy_entry(tmp_path, content):
    path = Path(tmp_path) / "scan_planning_params.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_scan_policy(path)
