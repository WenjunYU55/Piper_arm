#!/usr/bin/env python3
"""Create or update the offline real L515 baseline test-set manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURES_ROOT = Path("/home/prl/Piper_arm/L515_camera/captures")
DEFAULT_TEST_SET_ROOT = SCRIPT_DIR / "test_sets" / "real_l515_baseline"

REQUIRED_FILES = {
    "rgb": "rgb.png",
    "depth": "depth.npy",
    "detection_mask": "detection_mask.png",
    "camera_info": "camera_info.yaml",
    "target_3d": "target_3d.yaml",
    "metadata": "metadata.yaml",
}
OPTIONAL_FILES = {
    "scan_quality": "scan_quality.yaml",
    "occlusion_status": "occlusion_status.yaml",
}
CATEGORIES = (
    "clear_cube",
    "partial_occlusion",
    "heavy_occlusion",
    "hand_blocker",
    "edge_cases",
    "lost_target",
    "unknown",
)
USER_EDITABLE_KEYS = (
    "category",
    "expected_state",
    "target",
    "occluder",
    "notes",
)
ANALYSIS_KEYS = (
    "static_analysis_available",
    "static_analysis_decision",
    "static_analysis_output_path",
    "metrics",
)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def existing_capture_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {}
    for entry in manifest.get("captures", []):
        if isinstance(entry, dict) and entry.get("capture_name"):
            entries[str(entry["capture_name"])] = entry
    return entries


def file_presence(capture_dir: Path) -> dict[str, bool]:
    files = {}
    for key, filename in {**REQUIRED_FILES, **OPTIONAL_FILES}.items():
        files[key] = (capture_dir / filename).is_file()
    return files


def static_analysis_for(capture_name: str) -> dict[str, Any]:
    analysis_path = SCRIPT_DIR / "outputs" / capture_name / "analysis.yaml"
    if not analysis_path.is_file():
        return {
            "static_analysis_available": False,
            "static_analysis_decision": "",
            "static_analysis_output_path": "",
            "metrics": {},
        }
    analysis = read_yaml(analysis_path)
    return {
        "static_analysis_available": True,
        "static_analysis_decision": str(analysis.get("decision", "")),
        "static_analysis_output_path": str(analysis_path),
        "metrics": analysis.get("metrics", {}) if isinstance(analysis.get("metrics"), dict) else {},
    }


def ensure_test_set_dirs(test_set_root: Path) -> None:
    test_set_root.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (test_set_root / category).mkdir(parents=True, exist_ok=True)


def ensure_symlink(capture_dir: Path, test_set_root: Path, category: str) -> str:
    if category not in CATEGORIES:
        category = "unknown"
    link_path = test_set_root / category / capture_dir.name
    for existing_category in CATEGORIES:
        stale_path = test_set_root / existing_category / capture_dir.name
        if stale_path == link_path:
            continue
        if stale_path.is_symlink() and stale_path.resolve() == capture_dir.resolve():
            stale_path.unlink()
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == capture_dir.resolve():
            return str(link_path)
        return str(link_path)
    link_path.symlink_to(capture_dir, target_is_directory=True)
    return str(link_path)


def build_entry(capture_dir: Path, previous: dict[str, Any], test_set_root: Path) -> dict[str, Any]:
    capture_name = capture_dir.name
    entry = {
        "capture_name": capture_name,
        "source_path": str(capture_dir),
        "test_set_path": "",
        "category": "unknown",
        "expected_state": "unknown",
        "target": "green cube",
        "occluder": "unknown",
        "notes": "",
        "files_present": file_presence(capture_dir),
        "static_analysis_available": False,
        "static_analysis_decision": "",
        "static_analysis_output_path": "",
        "metrics": {},
    }

    for key in USER_EDITABLE_KEYS:
        if key in previous:
            entry[key] = previous[key]
    for key in ANALYSIS_KEYS:
        if key in previous:
            entry[key] = previous[key]

    analysis = static_analysis_for(capture_name)
    entry.update(analysis)
    entry["test_set_path"] = ensure_symlink(capture_dir, test_set_root, str(entry["category"]))
    return entry


def capture_dirs(captures_root: Path) -> list[Path]:
    if not captures_root.is_dir():
        return []
    return sorted(path for path in captures_root.iterdir() if path.is_dir())


def create_manifest(captures_root: Path, test_set_root: Path) -> dict[str, Any]:
    ensure_test_set_dirs(test_set_root)
    manifest_path = test_set_root / "manifest.yaml"
    previous_manifest = read_yaml(manifest_path)
    previous_entries = existing_capture_entries(previous_manifest)

    captures = [
        build_entry(capture_dir, previous_entries.get(capture_dir.name, {}), test_set_root)
        for capture_dir in capture_dirs(captures_root)
    ]

    manifest = {
        "manifest_type": "real_l515_baseline_test_set",
        "created_or_updated_utc": datetime.now(timezone.utc).isoformat(),
        "captures_root": str(captures_root),
        "test_set_root": str(test_set_root),
        "storage_mode": "symlink",
        "categories": list(CATEGORIES),
        "defaults": {
            "category": "unknown",
            "expected_state": "unknown",
            "target": "green cube",
            "occluder": "unknown",
        },
        "captures": captures,
    }
    write_yaml(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update the real L515 baseline manifest.")
    parser.add_argument("--captures-root", type=Path, default=DEFAULT_CAPTURES_ROOT)
    parser.add_argument("--test-set-root", type=Path, default=DEFAULT_TEST_SET_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = create_manifest(args.captures_root.expanduser().resolve(), args.test_set_root.expanduser().resolve())
    manifest_path = Path(manifest["test_set_root"]) / "manifest.yaml"
    print("Manifest written:", manifest_path)
    print("Captures listed:", len(manifest["captures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
