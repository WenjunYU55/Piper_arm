#!/usr/bin/env python3
"""Combine offline GroundingDINO, SAM2, and static labels into a readiness report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
AI_TEST_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "manifest.yaml"
DEFAULT_GROUNDINGDINO_RESULTS = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "groundingdino_results.yaml"
DEFAULT_SAM2_RESULTS = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "sam2_results.yaml"
DEFAULT_OUTPUT = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "ai_readiness_summary.yaml"
MIN_TARGET_CONFIDENCE = 0.35
MIN_MASK_AREA_PX = 100
MIN_MASK_DEPTH_VALID_RATIO = 0.40


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


def by_capture(results_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = results_payload.get("results", [])
    if not isinstance(results, list):
        return {}
    return {
        str(result.get("capture_name", "")): result
        for result in results
        if isinstance(result, dict) and result.get("capture_name")
    }


def readiness_for_capture(
    entry: dict[str, Any],
    grounding: dict[str, Any] | None,
    sam2: dict[str, Any] | None,
) -> dict[str, Any]:
    grounding = grounding or {}
    sam2 = sam2 or {}
    reasons = []

    model_target_detected = bool(grounding.get("target_detected", False))
    model_target_confidence = float(grounding.get("target_confidence", 0.0))
    target_prompt_source = str(sam2.get("target_prompt_source", ""))
    fallback_target_confidence = float(sam2.get("target_prompt_confidence", 0.0))

    target_mask_available = int(sam2.get("target_mask_count", 0)) > 0
    target_mask_area = int(sam2.get("target_mask_area_px", 0))
    target_mask_depth_ratio = float(sam2.get("target_mask_depth_valid_ratio", 0.0))
    mask_reliable = (
        target_mask_available
        and target_mask_area >= MIN_MASK_AREA_PX
        and target_mask_depth_ratio >= MIN_MASK_DEPTH_VALID_RATIO
    )
    if not target_mask_available:
        reasons.append("target_mask_missing")
    elif not mask_reliable:
        reasons.append("target_mask_or_depth_unreliable")

    fallback_target_detected = target_prompt_source == "hsv_detection_mask" and mask_reliable
    target_confidence = max(model_target_confidence, fallback_target_confidence if fallback_target_detected else 0.0)
    target_visible = (
        (model_target_detected and model_target_confidence >= MIN_TARGET_CONFIDENCE)
        or (fallback_target_detected and fallback_target_confidence >= MIN_TARGET_CONFIDENCE)
    )
    if not target_visible:
        reasons.append("target_not_detected_or_low_confidence")

    depth_metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
    static_depth_reliable = float(depth_metrics.get("valid_depth_ratio", 0.0)) >= MIN_MASK_DEPTH_VALID_RATIO
    depth_reliable = mask_reliable or static_depth_reliable
    if not depth_reliable:
        reasons.append("depth_unreliable")

    obstacle_count = int(sam2.get("obstacle_mask_count", 0))
    closer_obstacle_masks = int(sam2.get("closer_obstacle_mask_count", 0))
    expected_state = str(entry.get("expected_state", "unknown")).upper()
    static_state = str(entry.get("static_analysis_decision", "UNKNOWN")).upper()
    occluder_text = str(entry.get("occluder", "unknown"))
    occluder_visible = obstacle_count > 0 or closer_obstacle_masks > 0 or expected_state in ("PARTIAL", "BLOCKED")
    blocked_or_occluded = (
        closer_obstacle_masks > 0
        or expected_state in ("PARTIAL", "BLOCKED")
        or static_state in ("PARTIAL", "BLOCKED")
    )

    unsafe_candidate = int(sam2.get("unsafe_obstacle_mask_count", 0)) > 0
    candidate_safe_class = int(sam2.get("candidate_safe_obstacle_mask_count", 0)) > 0
    if unsafe_candidate:
        reasons.append("unsafe_or_unknown_occluder")

    manipulation_evidence_good = target_visible and depth_reliable and mask_reliable and occluder_visible and candidate_safe_class and not unsafe_candidate
    if not manipulation_evidence_good:
        reasons.append("insufficient_safe_manipulation_evidence")

    # Offline reports may identify candidate-safe perception evidence, but they must not authorize motion.
    safe_to_consider_manipulation = False
    if safe_to_consider_manipulation:
        reasons.append("real_motion_disabled")
    else:
        reasons.append("offline_advisory_only_real_motion_disabled")

    return {
        "capture_name": entry.get("capture_name", ""),
        "category": entry.get("category", "unknown"),
        "expected_state": expected_state,
        "static_analysis_decision": static_state,
        "target": entry.get("target", "green cube"),
        "occluder": occluder_text,
        "target_visible": bool(target_visible),
        "target_confidence": target_confidence,
        "target_prompt_source": target_prompt_source,
        "target_mask_available": bool(target_mask_available),
        "mask_reliable": bool(mask_reliable),
        "depth_reliable": bool(depth_reliable),
        "occluder_visible": bool(occluder_visible),
        "blocked_or_occluded": bool(blocked_or_occluded),
        "candidate_safe_class": bool(candidate_safe_class),
        "unsafe_candidate": bool(unsafe_candidate),
        "safe_to_consider_manipulation": bool(safe_to_consider_manipulation),
        "blocked_reason": "; ".join(dict.fromkeys(reasons)),
        "groundingdino_status": grounding.get("status", "missing"),
        "sam2_status": sam2.get("status", "missing"),
        "groundingdino_boxes_yaml": grounding.get("boxes_yaml", ""),
        "sam2_masks_yaml": sam2.get("masks_yaml", ""),
    }


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "target_visible": 0,
        "mask_reliable": 0,
        "depth_reliable": 0,
        "occluder_visible": 0,
        "blocked_or_occluded": 0,
        "candidate_safe_class": 0,
        "unsafe_candidate": 0,
        "safe_to_consider_manipulation": 0,
    }
    for entry in entries:
        for key in counts:
            if entry.get(key):
                counts[key] += 1
    return counts


def summarize(
    manifest_path: Path,
    groundingdino_results_path: Path,
    sam2_results_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    captures = manifest.get("captures", [])
    if not isinstance(captures, list):
        raise RuntimeError("manifest has no captures list: %s" % manifest_path)
    grounding_by_capture = by_capture(read_yaml(groundingdino_results_path))
    sam2_by_capture = by_capture(read_yaml(sam2_results_path))

    readiness = []
    for entry in captures:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("capture_name", ""))
        readiness.append(readiness_for_capture(entry, grounding_by_capture.get(name), sam2_by_capture.get(name)))

    payload = {
        "results_type": "offline_ai_manipulation_readiness_summary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "groundingdino_results_path": str(groundingdino_results_path),
        "sam2_results_path": str(sam2_results_path),
        "capture_count": len(readiness),
        "policy": {
            "offline_only": True,
            "real_motion_allowed": False,
            "safe_to_consider_manipulation_default": False,
            "note": "This report is advisory perception validation only and must not trigger robot motion.",
        },
        "aggregate": aggregate(readiness),
        "captures": readiness,
    }
    write_yaml(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create offline AI manipulation-readiness summary.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--groundingdino-results", type=Path, default=DEFAULT_GROUNDINGDINO_RESULTS)
    parser.add_argument("--sam2-results", type=Path, default=DEFAULT_SAM2_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = summarize(
        manifest_path=args.manifest.expanduser().resolve(),
        groundingdino_results_path=args.groundingdino_results.expanduser().resolve(),
        sam2_results_path=args.sam2_results.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
    )
    print("Readiness summary written:", args.output.expanduser().resolve())
    print("Capture count:", payload["capture_count"])
    print("Aggregate:", payload["aggregate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
