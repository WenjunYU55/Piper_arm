#!/usr/bin/env python3
"""Run offline Grounded-SAM-2 bundled GroundingDINO over the L515 manifest."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from run_groundingdino_on_capture import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPO_DIR,
    DEFAULT_TEXT_THRESHOLD,
    GroundingDinoUnavailable,
    run_on_capture,
)


SCRIPT_DIR = Path(__file__).resolve().parent
AI_TEST_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "manifest.yaml"
DEFAULT_RESULTS = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "groundingdino_results.yaml"
DEFAULT_PROMPT = "green cube . cube . box . hand . leaf . branch . stem . fruit . tool . wire . unknown object ."


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


def compact_result(capture_entry: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    return {
        "capture_name": capture_entry.get("capture_name", ""),
        "source_path": capture_entry.get("source_path", ""),
        "category": capture_entry.get("category", "unknown"),
        "expected_state": capture_entry.get("expected_state", "unknown"),
        "target": capture_entry.get("target", "green cube"),
        "occluder": capture_entry.get("occluder", "unknown"),
        "status": "ok",
        "detection_count": len(inference.get("detections", [])),
        "detected_labels": inference.get("detected_labels", []),
        "boxes_yaml": inference.get("outputs", {}).get("boxes_yaml", ""),
        "debug_png": inference.get("outputs", {}).get("debug_png", ""),
    }


def error_result(capture_entry: dict[str, Any], status: str, error: Exception | str) -> dict[str, Any]:
    return {
        "capture_name": capture_entry.get("capture_name", ""),
        "source_path": capture_entry.get("source_path", ""),
        "category": capture_entry.get("category", "unknown"),
        "expected_state": capture_entry.get("expected_state", "unknown"),
        "target": capture_entry.get("target", "green cube"),
        "occluder": capture_entry.get("occluder", "unknown"),
        "status": status,
        "error": str(error),
        "detection_count": 0,
        "detected_labels": [],
        "boxes_yaml": "",
        "debug_png": "",
    }


def run_batch(
    manifest_path: Path,
    results_path: Path,
    prompt: str,
    output_root: Path,
    repo_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    captures = manifest.get("captures", [])
    if not isinstance(captures, list):
        raise RuntimeError("manifest has no captures list: %s" % manifest_path)

    results = []
    ok_count = 0
    failed_count = 0
    for entry in captures:
        if not isinstance(entry, dict):
            continue
        source_path = Path(str(entry.get("source_path", "")))
        try:
            inference = run_on_capture(
                capture_dir=source_path,
                prompt=prompt,
                output_root=output_root,
                repo_dir=repo_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )
            results.append(compact_result(entry, inference))
            ok_count += 1
        except GroundingDinoUnavailable as exc:
            results.append(error_result(entry, "groundingdino_unavailable", exc))
            failed_count += 1
            break
        except Exception as exc:
            results.append(error_result(entry, "failed", exc))
            failed_count += 1

    payload = {
        "results_type": "offline_grounded_sam2_groundingdino_real_l515_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "prompt": prompt,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "device": device,
        "output_root": str(output_root),
        "capture_count": len(captures),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "results": results,
    }
    write_yaml(results_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline GroundingDINO over manifest.yaml captures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GROUNDINGDINO_REPO_DIR", DEFAULT_REPO_DIR)))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CONFIG_PATH", DEFAULT_CONFIG_PATH)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--device", default=os.environ.get("GROUNDINGDINO_DEVICE", "cpu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_batch(
        manifest_path=args.manifest.expanduser().resolve(),
        results_path=args.results.expanduser().resolve(),
        prompt=args.prompt,
        output_root=args.output_root.expanduser().resolve(),
        repo_dir=args.repo_dir.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    print("Results written:", args.results.expanduser().resolve())
    print("OK:", payload["ok_count"])
    print("Failed:", payload["failed_count"])
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
