"""GUI scan-policy regressions."""

from pathlib import Path

import pytest

from piper_gui.scan_policy import (
    FULL_SPHERE_REGION,
    LEGACY_POLICY,
    RAY_NBV_POLICY,
    TARGET_SECTOR_REGION,
    VOXEL_NBV_POLICY,
    read_scan_policy,
    read_scan_settings,
    write_scan_policy,
    write_scan_settings,
)


def _config(tmp_path, policy=LEGACY_POLICY):
    path = Path(tmp_path) / "scan_planning_params.yaml"
    path.write_text(
        "/**:\n  ros__parameters:\n"
        "    max_viewpoints: 25\n"
        '    ray_sampling_region: "target_sector"\n'
        "    ray_count: 175\n"
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


def test_complete_ray_settings_change_atomically_without_touching_other_yaml(
        tmp_path):
    path = _config(tmp_path, VOXEL_NBV_POLICY)
    before = path.read_text(encoding="utf-8")

    saved = write_scan_settings(
        path, RAY_NBV_POLICY, FULL_SPHERE_REGION, 240)

    assert saved.policy == RAY_NBV_POLICY
    assert read_scan_settings(path) == saved
    after = path.read_text(encoding="utf-8")
    expected = before.replace(
        '"voxel_nbv"', '"ray_nbv"').replace(
            '"target_sector"', '"full_sphere"').replace(
                "ray_count: 175", "ray_count: 240")
    assert after == expected
    assert "max_viewpoints: 25" in after
    assert "# retained comment" in after


@pytest.mark.parametrize("count", [0, 1001])
def test_complete_settings_reject_unbounded_ray_count(tmp_path, count):
    path = _config(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="ray count must be between"):
        write_scan_settings(
            path, RAY_NBV_POLICY, TARGET_SECTOR_REGION, count)
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
