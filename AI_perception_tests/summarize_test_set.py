#!/usr/bin/env python3
"""Summarise the real L515 baseline manifest and analysis results."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "test_sets" / "real_l515_baseline" / "manifest.yaml"
REQUIRED_FILE_KEYS = ("rgb", "depth", "detection_mask", "camera_info", "target_3d", "metadata")
DECISIONS = ("CLEAR", "PARTIAL", "BLOCKED", "LOST", "UNKNOWN")


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


def complete_files(entry: dict[str, Any]) -> bool:
    files = entry.get("files_present", {})
    if not isinstance(files, dict):
        return False
    return all(bool(files.get(key)) for key in REQUIRED_FILE_KEYS)


def missing_required(entry: dict[str, Any]) -> list[str]:
    files = entry.get("files_present", {})
    if not isinstance(files, dict):
        return list(REQUIRED_FILE_KEYS)
    return [key for key in REQUIRED_FILE_KEYS if not bool(files.get(key))]


def normalise_state(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in ("PARTIALLY_OCCLUDED", "PARTIAL_OCCLUSION"):
        return "PARTIAL"
    if text in ("HEAVILY_OCCLUDED", "HEAVY_OCCLUSION", "OCCLUDED", "BLOCKED"):
        return "BLOCKED"
    if text in DECISIONS:
        return text
    return "UNKNOWN" if text in ("", "UNKNOWN") else text


def summarise(manifest_path: Path) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    captures = [entry for entry in manifest.get("captures", []) if isinstance(entry, dict)]

    category_counts = Counter(str(entry.get("category", "unknown")) for entry in captures)
    decision_counts = Counter(
        normalise_state(entry.get("static_analysis_decision", "UNKNOWN"))
        for entry in captures
        if entry.get("static_analysis_available")
    )
    for decision in DECISIONS:
        decision_counts.setdefault(decision, 0)

    missing_entries = []
    mismatches = []
    complete_count = 0
    analysed_count = 0
    for entry in captures:
        if complete_files(entry):
            complete_count += 1
        else:
            missing_entries.append(
                {
                    "capture_name": entry.get("capture_name", ""),
                    "source_path": entry.get("source_path", ""),
                    "missing_required_files": missing_required(entry),
                }
            )

        if entry.get("static_analysis_available"):
            analysed_count += 1

        expected = normalise_state(entry.get("expected_state", "unknown"))
        actual = normalise_state(entry.get("static_analysis_decision", "UNKNOWN"))
        if expected != "UNKNOWN" and actual != "UNKNOWN" and expected != actual:
            mismatches.append(
                {
                    "capture_name": entry.get("capture_name", ""),
                    "expected_state": expected,
                    "static_analysis_decision": actual,
                    "analysis_path": entry.get("static_analysis_output_path", ""),
                }
            )

    summary = {
        "summary_type": "real_l515_baseline_summary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "capture_count": len(captures),
        "category_counts": dict(sorted(category_counts.items())),
        "complete_file_count": complete_count,
        "missing_file_count": len(missing_entries),
        "analysed_count": analysed_count,
        "static_analysis_decision_counts": {decision: int(decision_counts[decision]) for decision in DECISIONS},
        "expected_state_mismatches": mismatches,
        "captures_missing_required_files": missing_entries,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("Captures:", summary["capture_count"])
    print("Complete files:", summary["complete_file_count"])
    print("Missing files:", summary["missing_file_count"])
    print("Analysed:", summary["analysed_count"])
    print("Categories:")
    for category, count in summary["category_counts"].items():
        print("  %s: %s" % (category, count))
    print("Static analyser decisions:")
    for decision, count in summary["static_analysis_decision_counts"].items():
        print("  %s: %s" % (decision, count))
    print("Mismatches:", len(summary["expected_state_mismatches"]))
    print("Captures missing required files:", len(summary["captures_missing_required_files"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise an offline L515 test set manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    summary = summarise(manifest_path)
    summary_path = manifest_path.parent / "summary.yaml"
    write_yaml(summary_path, summary)
    print_summary(summary)
    print("Summary written:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
