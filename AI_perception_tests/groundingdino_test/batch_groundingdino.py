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
    DEFAULT_LOCAL_BOX_THRESHOLD,
    DEFAULT_OBSTACLE_PROMPT,
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
DEFAULT_PROMPT = (
    "green cube . cube . box . hand . pen . tissue . paper tissue . paper . "
    "fruit . tool . wire . unknown object ."
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


def compact_result(capture_entry: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    summary = inference.get("summary", {})
    target_detected = bool(summary.get("target_detected", False))
    expected_state = str(capture_entry.get("expected_state", "unknown")).upper()
    expected_target_visible = expected_state not in ("LOST", "UNKNOWN")
    return {
        "capture_name": capture_entry.get("capture_name", ""),
        "source_path": capture_entry.get("source_path", ""),
        "category": capture_entry.get("category", "unknown"),
        "expected_state": expected_state,
        "target": capture_entry.get("target", "green cube"),
        "occluder": capture_entry.get("occluder", "unknown"),
        "status": "ok",
        "detection_count": len(inference.get("detections", [])),
        "detected_labels": inference.get("detected_labels", []),
        "target_detected": target_detected,
        "target_confidence": float(summary.get("target_confidence", 0.0)),
        "model_target_detected": bool(summary.get("model_target_detected", target_detected)),
        "target_source": summary.get("target_source", "groundingdino" if target_detected else "none"),
        "best_target_detection": summary.get("best_target_detection"),
        "obstacle_candidate_count": len(summary.get("obstacle_candidates", [])),
        "unsafe_candidate_count": len(summary.get("unsafe_candidates", [])),
        "candidate_safe_class_count": len(summary.get("candidate_safe_class_detections", [])),
        "target_local_detection_count": len(inference.get("target_local_detections", [])),
        "target_eval": target_eval_label(target_detected, expected_target_visible),
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
        "target_detected": False,
        "target_confidence": 0.0,
        "model_target_detected": False,
        "target_source": "none",
        "best_target_detection": None,
        "obstacle_candidate_count": 0,
        "unsafe_candidate_count": 0,
        "candidate_safe_class_count": 0,
        "target_local_detection_count": 0,
        "target_eval": "ERROR",
        "boxes_yaml": "",
        "debug_png": "",
    }


def target_eval_label(target_detected: bool, expected_target_visible: bool) -> str:
    if target_detected and expected_target_visible:
        return "TP"
    if target_detected and not expected_target_visible:
        return "FP"
    if not target_detected and expected_target_visible:
        return "FN"
    return "TN"


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "ERROR": 0}
    by_category: dict[str, dict[str, int]] = {}
    unsafe_count = 0
    candidate_safe_count = 0
    for result in results:
        label = str(result.get("target_eval", "ERROR"))
        counts[label] = counts.get(label, 0) + 1
        category = str(result.get("category", "unknown"))
        by_category.setdefault(category, {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "ERROR": 0})
        by_category[category][label] = by_category[category].get(label, 0) + 1
        if int(result.get("unsafe_candidate_count", 0)) > 0:
            unsafe_count += 1
        if int(result.get("candidate_safe_class_count", 0)) > 0:
            candidate_safe_count += 1

    evaluated = max(1, counts["TP"] + counts["FP"] + counts["FN"] + counts["TN"])
    precision_den = max(1, counts["TP"] + counts["FP"])
    recall_den = max(1, counts["TP"] + counts["FN"])
    return {
        "target_eval_counts": counts,
        "target_eval_by_category": by_category,
        "target_precision": float(counts["TP"] / precision_den),
        "target_recall": float(counts["TP"] / recall_den),
        "target_accuracy": float((counts["TP"] + counts["TN"]) / evaluated),
        "captures_with_unsafe_candidates": unsafe_count,
        "captures_with_candidate_safe_classes": candidate_safe_count,
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
    obstacle_prompt: str,
    local_box_threshold: float,
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
                obstacle_prompt=obstacle_prompt,
                local_box_threshold=local_box_threshold,
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
        "obstacle_prompt": obstacle_prompt,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "local_box_threshold": float(local_box_threshold),
        "device": device,
        "output_root": str(output_root),
        "capture_count": len(captures),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "aggregate": aggregate_results(results),
        "results": results,
    }
    write_yaml(results_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline GroundingDINO over manifest.yaml captures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--obstacle-prompt", default=DEFAULT_OBSTACLE_PROMPT)
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GROUNDINGDINO_REPO_DIR", DEFAULT_REPO_DIR)))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CONFIG_PATH", DEFAULT_CONFIG_PATH)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--local-box-threshold", type=float, default=DEFAULT_LOCAL_BOX_THRESHOLD)
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
        obstacle_prompt=args.obstacle_prompt,
        local_box_threshold=args.local_box_threshold,
    )
    print("Results written:", args.results.expanduser().resolve())
    print("OK:", payload["ok_count"])
    print("Failed:", payload["failed_count"])
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
