#!/usr/bin/env python3
"""Refine offline GroundingDINO boxes with SAM2 masks for one saved L515 capture."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
AI_TEST_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = AI_TEST_DIR / "outputs"
DEFAULT_GROUNDED_SAM2_REPO_DIR = SCRIPT_DIR / "Grounded-SAM-2"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_SAM2_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"
TARGET_DEPTH_MARGIN_M = 0.03
TARGET_SEARCH_MARGIN_PX = 24
MIN_DEPTH_OCCLUDER_AREA_PX = 20
MIN_HSV_FALLBACK_AREA_PX = 100
TARGET_MASK_NEAR_MARGIN_PX = 6
MIN_TARGET_NEAR_OVERLAP_PX = 10
MIN_SEMANTIC_DEPTH_COVERAGE = 0.50


class Sam2Unavailable(RuntimeError):
    pass


def add_repo_to_path(repo_dir: Path) -> None:
    if repo_dir.is_dir() and str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def require_sam2(repo_dir: Path):
    add_repo_to_path(repo_dir)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise Sam2Unavailable(
            "SAM2 is not importable from the Grounded-SAM-2 checkout. Install the "
            "Grounded-SAM-2 dependencies in the isolated env. Original error: %s" % exc
        ) from exc
    return build_sam2, SAM2ImagePredictor


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


def clean_previous_mask_outputs(output_dir: Path) -> None:
    for pattern in ("mask_*.png", "rejected_mask_*.png"):
        for path in output_dir.glob(pattern):
            path.unlink()


def unavailable_payload(capture_dir: Path, output_dir: Path, status: str, reason: str) -> dict[str, Any]:
    return {
        "results_type": "offline_sam2_mask_refinement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_name": capture_dir.name,
        "capture_path": str(capture_dir),
        "status": status,
        "reason": reason,
        "masks": [],
        "outputs": {"masks_yaml": str(output_dir / "sam2_masks.yaml")},
    }


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32, copy=False) * 0.001
    depth_m = depth.astype(np.float32, copy=False)
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    if finite.size > 0 and float(np.nanmedian(finite)) > 20.0:
        return depth_m * 0.001
    return depth_m


def load_depth(capture_dir: Path) -> np.ndarray | None:
    depth_path = capture_dir / "depth.npy"
    if not depth_path.is_file():
        return None
    return depth_to_meters(np.load(str(depth_path)))


def target_depth_from_capture(capture_dir: Path, fallback_depth: np.ndarray | None = None) -> float | None:
    target = read_yaml(capture_dir / "target_3d.yaml")
    for value in (target.get("depth"), target.get("point", {}).get("z") if isinstance(target.get("point"), dict) else None):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric > 0.0:
            return numeric
    if fallback_depth is None:
        return None
    mask = cv2.imread(str(capture_dir / "detection_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != fallback_depth.shape[:2]:
        return None
    values = fallback_depth[(mask > 0) & np.isfinite(fallback_depth) & (fallback_depth > 0.0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def hsv_target_fallback(capture_dir: Path) -> dict[str, Any] | None:
    mask = cv2.imread(str(capture_dir / "detection_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_bool = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool, connectivity=8)
    if component_count <= 1:
        return None

    target = read_yaml(capture_dir / "target_3d.yaml")
    if not bool(target.get("valid", False)):
        return None
    source_u = int(round(float(target.get("source_u", -1))))
    source_v = int(round(float(target.get("source_v", -1))))
    if not (0 <= source_u < mask.shape[1] and 0 <= source_v < mask.shape[0]):
        return None
    component = int(labels[source_v, source_u])
    if component <= 0 or int(stats[component, cv2.CC_STAT_AREA]) < MIN_HSV_FALLBACK_AREA_PX:
        return None

    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    width = int(stats[component, cv2.CC_STAT_WIDTH])
    height = int(stats[component, cv2.CC_STAT_HEIGHT])
    if width <= 0 or height <= 0:
        return None
    return {
        "label": "green cube (HSV fallback)",
        "confidence": float(target.get("measurement_confidence", 0.0)),
        "box_xyxy_pixels": [float(x), float(y), float(x + width), float(y + height)],
        "is_target_candidate": True,
        "is_unsafe_candidate": False,
        "is_candidate_safe_class": False,
        "prompt_source": "hsv_detection_mask",
    }


def boxes_are_near(box: list[float], target_box: list[float], margin_px: int) -> bool:
    x0, y0, x1, y1 = [float(value) for value in box]
    tx0, ty0, tx1, ty1 = [float(value) for value in target_box]
    return not (
        x1 < tx0 - margin_px
        or x0 > tx1 + margin_px
        or y1 < ty0 - margin_px
        or y0 > ty1 + margin_px
    )


def detections_for_masks(
    grounding: dict[str, Any], capture_dir: Path, max_masks: int
) -> list[dict[str, Any]]:
    summary = grounding.get("summary", {})
    selected = []
    target = summary.get("best_target_detection")
    target_source = str(summary.get("target_source", "groundingdino"))
    if not isinstance(target, dict):
        target = hsv_target_fallback(capture_dir)
        target_source = "hsv_detection_mask"
    if isinstance(target, dict):
        selected.append(dict(target, mask_role="target", prompt_source=target_source))
    target_box = target.get("box_xyxy_pixels", []) if isinstance(target, dict) else []
    for record in summary.get("obstacle_candidates", []):
        if not isinstance(record, dict):
            continue
        box = record.get("box_xyxy_pixels", [])
        if len(target_box) == 4 and len(box) == 4 and boxes_are_near(box, target_box, TARGET_SEARCH_MARGIN_PX):
            selected.append(dict(record, mask_role="obstacle", prompt_source="groundingdino_near_target"))
    if not selected:
        for record in grounding.get("detections", []):
            if isinstance(record, dict):
                selected.append(dict(record, mask_role="unknown"))
    selected.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return selected[:max_masks]


def depth_occluder_mask(
    capture_dir: Path,
    depth_m: np.ndarray | None,
    target_depth_m: float | None,
    target_mask: np.ndarray | None,
) -> np.ndarray | None:
    if depth_m is None or target_depth_m is None or target_mask is None:
        return None
    base_mask = target_mask
    if base_mask.shape[:2] != depth_m.shape[:2]:
        return None
    base_u8 = base_mask.astype(np.uint8)
    kernel_size = TARGET_SEARCH_MARGIN_PX * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    near_target = cv2.dilate(base_u8, kernel) > 0
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    closer = near_target & ~base_mask.astype(bool) & valid & (depth_m < target_depth_m - TARGET_DEPTH_MARGIN_M)
    closer_u8 = cv2.morphologyEx(closer.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(np.count_nonzero(closer_u8)) < MIN_DEPTH_OCCLUDER_AREA_PX:
        return None
    return closer_u8 > 0


def best_mask(masks: Any, scores: Any, index: int) -> tuple[np.ndarray, float]:
    masks_np = np.asarray(masks)
    scores_np = np.asarray(scores)
    if masks_np.ndim == 2:
        return masks_np > 0, float(scores_np.reshape(-1)[0]) if scores_np.size else 0.0
    if masks_np.ndim == 3:
        if masks_np.shape[0] == scores_np.reshape(-1).shape[0]:
            score_index = int(np.argmax(scores_np.reshape(-1)))
            return masks_np[score_index] > 0, float(scores_np.reshape(-1)[score_index])
        return masks_np[index] > 0, 0.0
    if masks_np.ndim == 4:
        candidates = masks_np[index]
        candidate_scores = scores_np[index].reshape(-1) if scores_np.ndim > 1 else scores_np.reshape(-1)
        score_index = int(np.argmax(candidate_scores)) if candidate_scores.size else 0
        return candidates[score_index] > 0, float(candidate_scores[score_index]) if candidate_scores.size else 0.0
    raise RuntimeError("unexpected SAM2 mask shape: %s" % (masks_np.shape,))


def mask_metrics(mask_bool: np.ndarray, depth_m: np.ndarray | None, target_depth_m: float | None) -> dict[str, Any]:
    area = int(np.count_nonzero(mask_bool))
    ys, xs = np.nonzero(mask_bool)
    bbox = {"available": False}
    centroid = {"available": False}
    if xs.size and ys.size:
        bbox = {"available": True, "x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max())}
        centroid = {"available": True, "u": float(np.mean(xs)), "v": float(np.mean(ys))}

    depth_payload = {
        "depth_available": False,
        "valid_depth_ratio": 0.0,
        "median_depth_m": None,
        "closer_than_target": False,
    }
    if depth_m is not None and depth_m.shape[:2] == mask_bool.shape[:2] and area > 0:
        values = depth_m[mask_bool & np.isfinite(depth_m) & (depth_m > 0.0)]
        valid_ratio = float(values.size / max(1, area))
        median_depth = float(np.median(values)) if values.size else None
        closer = False
        if median_depth is not None and target_depth_m is not None:
            closer = bool(median_depth < target_depth_m - TARGET_DEPTH_MARGIN_M)
        depth_payload = {
            "depth_available": True,
            "valid_depth_ratio": valid_ratio,
            "median_depth_m": median_depth,
            "closer_than_target": closer,
        }

    return {"mask_area_px": area, "bbox": bbox, "centroid": centroid, **depth_payload}


def write_overlay(rgb_bgr: np.ndarray, masks: list[dict[str, Any]], output_path: Path) -> None:
    overlay = rgb_bgr.copy()
    for item in masks:
        role = item.get("mask_role", "unknown")
        if role == "obstacle" and not item.get("target_relevant_occluder", False):
            continue
        mask = cv2.imread(item["mask_png"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color = (0, 255, 0) if role == "target" else (0, 0, 255)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    cv2.imwrite(str(output_path), overlay)


def mark_target_relevant_occluders(mask_records: list[dict[str, Any]], target_mask: np.ndarray | None) -> None:
    if target_mask is None:
        for record in mask_records:
            if record.get("mask_role") == "obstacle":
                record["target_relevant_occluder"] = False
        return
    kernel_size = TARGET_MASK_NEAR_MARGIN_PX * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    target_near = cv2.dilate(target_mask.astype(np.uint8), kernel) > 0
    target_record = next((record for record in mask_records if record.get("mask_role") == "target"), {})
    target_depth_reliable = float(target_record.get("valid_depth_ratio", 0.0)) >= 0.40
    for record in mask_records:
        if record.get("mask_role") != "obstacle":
            continue
        obstacle_mask = cv2.imread(record["mask_png"], cv2.IMREAD_GRAYSCALE)
        if obstacle_mask is None:
            record["target_relevant_occluder"] = False
            continue
        obstacle_bool = obstacle_mask > 0
        direct_overlap = int(np.count_nonzero(obstacle_bool & target_mask))
        near_overlap = int(np.count_nonzero(obstacle_bool & target_near))
        obstacle_area = max(1, int(np.count_nonzero(obstacle_bool)))
        target_area = max(1, int(np.count_nonzero(target_mask)))
        duplicate_target_ratio = float(direct_overlap / min(obstacle_area, target_area))
        depth_support = bool(record.get("closer_than_target", False)) or not target_depth_reliable
        record["target_overlap_px"] = direct_overlap
        record["target_near_overlap_px"] = near_overlap
        record["duplicate_target_ratio"] = duplicate_target_ratio
        min_near_overlap = 20 if record.get("prompt_source") == "target_local_depth" else MIN_TARGET_NEAR_OVERLAP_PX
        record["target_relevant_occluder"] = bool(
            near_overlap >= min_near_overlap
            and depth_support
            and duplicate_target_ratio < 0.5
        )


def associate_depth_with_semantic_obstacles(mask_records: list[dict[str, Any]]) -> None:
    semantic_records = [
        record
        for record in mask_records
        if record.get("mask_role") == "obstacle"
        and record.get("prompt_source") != "target_local_depth"
        and record.get("target_relevant_occluder", False)
        and record.get("mask_png")
    ]
    for depth_record in mask_records:
        if (
            depth_record.get("prompt_source") != "target_local_depth"
            or not depth_record.get("target_relevant_occluder", False)
            or not depth_record.get("mask_png")
        ):
            continue
        depth_mask = cv2.imread(depth_record["mask_png"], cv2.IMREAD_GRAYSCALE)
        if depth_mask is None:
            continue
        depth_bool = depth_mask > 0
        depth_area = int(np.count_nonzero(depth_bool))
        best_record = None
        best_coverage = 0.0
        for semantic_record in semantic_records:
            semantic_mask = cv2.imread(semantic_record["mask_png"], cv2.IMREAD_GRAYSCALE)
            if semantic_mask is None:
                continue
            semantic_bool = semantic_mask > 0
            semantic_area = int(np.count_nonzero(semantic_bool))
            intersection = int(np.count_nonzero(depth_bool & semantic_bool))
            coverage = float(intersection / max(1, min(depth_area, semantic_area)))
            if coverage > best_coverage:
                best_record = semantic_record
                best_coverage = coverage
        if best_record is None or best_coverage < MIN_SEMANTIC_DEPTH_COVERAGE:
            continue
        semantic_label = str(best_record.get("label", "unknown"))
        semantic_confidence = float(best_record.get("confidence", 0.0))
        best_record["depth_confirmed"] = True
        best_record["semantic_depth_coverage"] = best_coverage
        depth_record["explained_by_semantic_label"] = semantic_label
        depth_record["semantic_confidence"] = semantic_confidence
        depth_record["semantic_depth_coverage"] = best_coverage
        depth_record["is_unsafe_candidate"] = bool(best_record.get("is_unsafe_candidate", False))
        depth_record["is_candidate_safe_class"] = bool(best_record.get("is_candidate_safe_class", False))
        depth_record["label"] = "depth foreground (%s)" % semantic_label


def refine_capture(
    capture_dir: Path,
    groundingdino_boxes: Path,
    output_root: Path,
    repo_dir: Path,
    sam2_config: str,
    sam2_checkpoint: Path,
    device: str,
    max_masks: int,
) -> dict[str, Any]:
    capture_dir = capture_dir.expanduser().resolve()
    output_dir = output_root.expanduser().resolve() / capture_dir.name / "sam2"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_mask_outputs(output_dir)

    rgb_path = capture_dir / "rgb.png"
    if not rgb_path.is_file():
        payload = unavailable_payload(capture_dir, output_dir, "missing_rgb", "capture is missing rgb.png")
        write_yaml(output_dir / "sam2_masks.yaml", payload)
        return payload
    grounding = read_yaml(groundingdino_boxes.expanduser().resolve())
    if not grounding:
        payload = unavailable_payload(capture_dir, output_dir, "missing_groundingdino", "GroundingDINO boxes YAML is missing or empty")
        write_yaml(output_dir / "sam2_masks.yaml", payload)
        return payload
    if not sam2_checkpoint.expanduser().is_file():
        payload = unavailable_payload(capture_dir, output_dir, "sam2_unavailable", "SAM2 checkpoint not found: %s" % sam2_checkpoint)
        write_yaml(output_dir / "sam2_masks.yaml", payload)
        return payload

    try:
        build_sam2, SAM2ImagePredictor = require_sam2(repo_dir.expanduser().resolve())
    except Sam2Unavailable as exc:
        payload = unavailable_payload(capture_dir, output_dir, "sam2_unavailable", str(exc))
        write_yaml(output_dir / "sam2_masks.yaml", payload)
        return payload

    import torch

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        payload = unavailable_payload(capture_dir, output_dir, "failed", "failed to read rgb.png")
        write_yaml(output_dir / "sam2_masks.yaml", payload)
        return payload
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    detections = detections_for_masks(grounding, capture_dir, max_masks)
    if not detections:
        payload = unavailable_payload(capture_dir, output_dir, "no_detections", "no GroundingDINO detections available for SAM2")
        write_yaml(output_dir / "sam2_masks.yaml", payload)
        return payload

    boxes = np.asarray([record["box_xyxy_pixels"] for record in detections], dtype=np.float32)
    depth_m = load_depth(capture_dir)
    target_depth_m = target_depth_from_capture(capture_dir, depth_m)

    sam2_model = build_sam2(sam2_config, str(sam2_checkpoint.expanduser().resolve()), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(rgb)
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=True)

    mask_records = []
    for index, detection in enumerate(detections):
        mask_bool, score = best_mask(masks, scores, index)
        mask_path = output_dir / ("mask_%02d_%s.png" % (index, detection.get("mask_role", "unknown")))
        cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)
        record = {
            "index": index,
            "label": detection.get("label", ""),
            "confidence": float(detection.get("confidence", 0.0)),
            "mask_role": detection.get("mask_role", "unknown"),
            "prompt_source": detection.get("prompt_source", "groundingdino"),
            "sam2_score": float(score),
            "box_xyxy_pixels": detection.get("box_xyxy_pixels", []),
            "is_target_candidate": bool(detection.get("is_target_candidate", False)),
            "is_unsafe_candidate": bool(detection.get("is_unsafe_candidate", False)),
            "is_candidate_safe_class": bool(detection.get("is_candidate_safe_class", False)),
            "mask_png": str(mask_path),
        }
        record.update(mask_metrics(mask_bool, depth_m, target_depth_m))
        mask_records.append(record)

    target_mask = next(
        (
            cv2.imread(record["mask_png"], cv2.IMREAD_GRAYSCALE) > 0
            for record in mask_records
            if record.get("mask_role") == "target"
        ),
        None,
    )
    depth_mask = depth_occluder_mask(capture_dir, depth_m, target_depth_m, target_mask)
    if depth_mask is not None:
        mask_path = output_dir / ("mask_%02d_obstacle_depth.png" % len(mask_records))
        cv2.imwrite(str(mask_path), depth_mask.astype(np.uint8) * 255)
        record = {
            "index": len(mask_records),
            "label": "unknown depth foreground",
            "confidence": 1.0,
            "mask_role": "obstacle",
            "prompt_source": "target_local_depth",
            "sam2_score": 0.0,
            "box_xyxy_pixels": [],
            "is_target_candidate": False,
            "is_unsafe_candidate": True,
            "is_candidate_safe_class": False,
            "mask_png": str(mask_path),
        }
        record.update(mask_metrics(depth_mask, depth_m, target_depth_m))
        mask_records.append(record)

    mark_target_relevant_occluders(mask_records, target_mask)
    associate_depth_with_semantic_obstacles(mask_records)
    for record in mask_records:
        if record.get("mask_role") != "obstacle" or record.get("target_relevant_occluder", False):
            continue
        mask_path = Path(str(record.get("mask_png", "")))
        if mask_path.is_file():
            mask_path.unlink()
        record["mask_png"] = ""
        record["rejection_reason"] = "not_target_relevant"

    overlay_path = output_dir / "sam2_overlay.png"
    write_overlay(rgb_bgr, mask_records, overlay_path)
    payload = {
        "results_type": "offline_sam2_mask_refinement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_name": capture_dir.name,
        "capture_path": str(capture_dir),
        "status": "ok",
        "groundingdino_boxes": str(groundingdino_boxes),
        "sam2_config": sam2_config,
        "sam2_checkpoint": str(sam2_checkpoint),
        "device": device,
        "target_depth_m": target_depth_m,
        "masks": mask_records,
        "outputs": {
            "masks_yaml": str(output_dir / "sam2_masks.yaml"),
            "overlay_png": str(overlay_path),
        },
    }
    write_yaml(output_dir / "sam2_masks.yaml", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline SAM2 mask refinement on one saved L515 capture.")
    parser.add_argument("capture_folder", type=Path)
    parser.add_argument("--groundingdino-boxes", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GROUNDED_SAM2_REPO_DIR", DEFAULT_GROUNDED_SAM2_REPO_DIR)))
    parser.add_argument("--sam2-config", default=os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG))
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path(os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT)))
    parser.add_argument("--device", default=os.environ.get("SAM2_DEVICE", os.environ.get("GROUNDINGDINO_DEVICE", "cpu")))
    parser.add_argument("--max-masks", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = args.capture_folder.expanduser().resolve()
    boxes_path = args.groundingdino_boxes
    if boxes_path is None:
        boxes_path = args.output_root.expanduser().resolve() / capture_dir.name / "groundingdino" / "groundingdino_boxes.yaml"
    result = refine_capture(
        capture_dir=capture_dir,
        groundingdino_boxes=boxes_path,
        output_root=args.output_root,
        repo_dir=args.repo_dir,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        max_masks=max(1, args.max_masks),
    )
    print("Status:", result["status"])
    print("Masks YAML:", result["outputs"]["masks_yaml"])
    return 0 if result["status"] in ("ok", "no_detections", "sam2_unavailable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
