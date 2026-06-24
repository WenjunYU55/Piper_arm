#!/usr/bin/env python3
"""Run offline SAM2 mask refinement for every capture in a manifest."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sam2_refine_on_capture import (
    DEFAULT_GROUNDED_SAM2_REPO_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    refine_capture,
)


SCRIPT_DIR = Path(__file__).resolve().parent
AI_TEST_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "manifest.yaml"
DEFAULT_RESULTS = AI_TEST_DIR / "test_sets" / "real_l515_baseline" / "sam2_results.yaml"
MIN_CANDIDATE_SAFE_CONFIDENCE = 0.45
MIN_DEPTH_CONFIRMED_SAFE_CONFIDENCE = 0.40


def is_candidate_safe_obstacle(mask: dict[str, Any]) -> bool:
    if not mask.get("is_candidate_safe_class"):
        return False
    confidence = float(mask.get("semantic_confidence", mask.get("confidence", 0.0)))
    if confidence >= MIN_CANDIDATE_SAFE_CONFIDENCE:
        return True
    depth_confirmed = bool(mask.get("depth_confirmed", False) or mask.get("explained_by_semantic_label"))
    return depth_confirmed and confidence >= MIN_DEPTH_CONFIRMED_SAFE_CONFIDENCE


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


def compact_result(entry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    masks = result.get("masks", [])
    target_masks = [mask for mask in masks if isinstance(mask, dict) and mask.get("mask_role") == "target"]
    rejected_obstacles = [
        mask
        for mask in masks
        if isinstance(mask, dict)
        and mask.get("mask_role") == "obstacle"
        and not mask.get("target_relevant_occluder", False)
    ]
    obstacle_masks = [
        mask
        for mask in masks
        if isinstance(mask, dict)
        and mask.get("mask_role") == "obstacle"
        and mask.get("target_relevant_occluder", False)
    ]
    closer_obstacles = [mask for mask in obstacle_masks if mask.get("closer_than_target")]
    depth_obstacles = [mask for mask in obstacle_masks if mask.get("prompt_source") == "target_local_depth"]
    candidate_safe_obstacles = [
        mask
        for mask in obstacle_masks
        if is_candidate_safe_obstacle(mask)
    ]
    unsafe_obstacles = [
        mask
        for mask in obstacle_masks
        if mask.get("is_unsafe_candidate") or mask not in candidate_safe_obstacles
    ]
    target_prompt_source = str(target_masks[0].get("prompt_source", "")) if target_masks else ""
    return {
        "capture_name": entry.get("capture_name", ""),
        "source_path": entry.get("source_path", ""),
        "category": entry.get("category", "unknown"),
        "expected_state": entry.get("expected_state", "unknown"),
        "target": entry.get("target", "green cube"),
        "occluder": entry.get("occluder", "unknown"),
        "status": result.get("status", "unknown"),
        "reason": result.get("reason", ""),
        "mask_count": len(target_masks) + len(obstacle_masks),
        "raw_mask_candidate_count": len(masks),
        "rejected_obstacle_mask_count": len(rejected_obstacles),
        "target_mask_count": len(target_masks),
        "obstacle_mask_count": len(obstacle_masks),
        "closer_obstacle_mask_count": len(closer_obstacles),
        "depth_obstacle_mask_count": len(depth_obstacles),
        "unsafe_obstacle_mask_count": len(unsafe_obstacles),
        "candidate_safe_obstacle_mask_count": len(candidate_safe_obstacles),
        "target_mask_area_px": int(target_masks[0].get("mask_area_px", 0)) if target_masks else 0,
        "target_mask_depth_valid_ratio": float(target_masks[0].get("valid_depth_ratio", 0.0)) if target_masks else 0.0,
        "target_prompt_source": target_prompt_source,
        "target_prompt_confidence": float(target_masks[0].get("confidence", 0.0)) if target_masks else 0.0,
        "masks_yaml": result.get("outputs", {}).get("masks_yaml", ""),
        "overlay_png": result.get("outputs", {}).get("overlay_png", ""),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    closer_count = 0
    target_mask_count = 0
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if int(result.get("closer_obstacle_mask_count", 0)) > 0:
            closer_count += 1
        if int(result.get("target_mask_count", 0)) > 0:
            target_mask_count += 1
    return {
        "status_counts": status_counts,
        "captures_with_target_masks": target_mask_count,
        "captures_with_closer_obstacle_masks": closer_count,
    }


def run_batch(
    manifest_path: Path,
    results_path: Path,
    output_root: Path,
    repo_dir: Path,
    sam2_config: str,
    sam2_checkpoint: Path,
    device: str,
    max_masks: int,
) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    captures = manifest.get("captures", [])
    if not isinstance(captures, list):
        raise RuntimeError("manifest has no captures list: %s" % manifest_path)

    results = []
    for entry in captures:
        if not isinstance(entry, dict):
            continue
        capture_dir = Path(str(entry.get("source_path", ""))).expanduser()
        boxes_path = output_root / str(entry.get("capture_name", capture_dir.name)) / "groundingdino" / "groundingdino_boxes.yaml"
        try:
            result = refine_capture(
                capture_dir=capture_dir,
                groundingdino_boxes=boxes_path,
                output_root=output_root,
                repo_dir=repo_dir,
                sam2_config=sam2_config,
                sam2_checkpoint=sam2_checkpoint,
                device=device,
                max_masks=max_masks,
            )
        except Exception as exc:
            result = {"status": "failed", "reason": str(exc), "masks": [], "outputs": {"masks_yaml": "", "overlay_png": ""}}
        results.append(compact_result(entry, result))

    payload = {
        "results_type": "offline_sam2_refinement_real_l515_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "sam2_config": sam2_config,
        "sam2_checkpoint": str(sam2_checkpoint),
        "device": device,
        "capture_count": len(captures),
        "aggregate": aggregate(results),
        "results": results,
    }
    write_yaml(results_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline SAM2 refinement over manifest.yaml captures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GROUNDED_SAM2_REPO_DIR", DEFAULT_GROUNDED_SAM2_REPO_DIR)))
    parser.add_argument("--sam2-config", default=os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG))
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path(os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT)))
    parser.add_argument("--device", default=os.environ.get("SAM2_DEVICE", os.environ.get("GROUNDINGDINO_DEVICE", "cpu")))
    parser.add_argument("--max-masks", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_batch(
        manifest_path=args.manifest.expanduser().resolve(),
        results_path=args.results.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        repo_dir=args.repo_dir.expanduser().resolve(),
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint.expanduser().resolve(),
        device=args.device,
        max_masks=max(1, args.max_masks),
    )
    print("Results written:", args.results.expanduser().resolve())
    print("Status counts:", payload["aggregate"]["status_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
