#!/usr/bin/env python3
"""Run the static RGB-D analyser for every capture in a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from static_scene_analyzer import DEFAULT_PARAMS, analyze_capture, write_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "test_sets" / "real_l515_baseline" / "manifest.yaml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
METRIC_KEYS = (
    "mask_area_px",
    "target_depth_m",
    "valid_depth_ratio",
    "closer_region_area_px",
    "closer_region_ratio",
)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def analysis_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = analysis.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def run_batch(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    captures = manifest.get("captures", [])
    if not isinstance(captures, list):
        raise RuntimeError("manifest.yaml has no captures list")

    analysed = 0
    skipped = 0
    for entry in captures:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        capture_name = str(entry.get("capture_name", "")).strip()
        source_path = Path(str(entry.get("source_path", ""))).expanduser()
        if not capture_name or not source_path.is_dir():
            entry["static_analysis_available"] = False
            entry["static_analysis_decision"] = "UNKNOWN"
            entry["static_analysis_output_path"] = ""
            entry["metrics"] = {}
            skipped += 1
            continue

        output_dir = output_root / capture_name
        analysis = analyze_capture(source_path.resolve(), output_dir.resolve(), dict(DEFAULT_PARAMS))
        analysis_path = output_dir / "analysis.yaml"
        write_yaml(analysis_path, analysis)

        entry["static_analysis_available"] = bool(analysis_path.is_file())
        entry["static_analysis_decision"] = str(analysis.get("decision", "UNKNOWN"))
        entry["static_analysis_output_path"] = str(analysis_path)
        entry["metrics"] = analysis_metrics(analysis)
        analysed += 1

    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)

    return {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "analysed": analysed,
        "skipped": skipped,
        "total_entries": len(captures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static analysis for every capture in manifest.yaml.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_batch(args.manifest.expanduser().resolve(), args.output_root.expanduser().resolve())
    print("Manifest updated:", result["manifest_path"])
    print("Output root:", result["output_root"])
    print("Analysed:", result["analysed"])
    print("Skipped:", result["skipped"])
    print("Total entries:", result["total_entries"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
