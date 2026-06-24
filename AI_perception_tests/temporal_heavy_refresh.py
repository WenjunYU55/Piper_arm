#!/usr/bin/env python3
"""Adapter that runs existing offline GroundingDINO/SAM2 code for temporal refresh events."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
GROUNDING_TEST_DIR = SCRIPT_DIR / "groundingdino_test"
if str(GROUNDING_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(GROUNDING_TEST_DIR))

from batch_groundingdino import DEFAULT_PROMPT  # noqa: E402
from batch_sam2_refine import is_candidate_safe_obstacle  # noqa: E402
from run_groundingdino_on_capture import (  # noqa: E402
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOCAL_BOX_THRESHOLD,
    DEFAULT_OBSTACLE_PROMPT,
    DEFAULT_REPO_DIR,
    DEFAULT_TEXT_THRESHOLD,
    run_on_capture,
)
from sam2_refine_on_capture import (  # noqa: E402
    DEFAULT_GROUNDED_SAM2_REPO_DIR,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    refine_capture,
)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def prepare_event_capture(
    capture_dir: Path,
    rgb_path: Path,
    depth: np.ndarray | None,
    tracked_mask: np.ndarray,
    tracking_confidence: float,
) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(rgb_path), str(capture_dir / "rgb.png"))
    if depth is not None:
        np.save(str(capture_dir / "depth.npy"), depth)
    cv2.imwrite(str(capture_dir / "detection_mask.png"), tracked_mask.astype(np.uint8) * 255)

    ys, xs = np.nonzero(tracked_mask)
    if xs.size and ys.size:
        source_u = float(np.mean(xs))
        source_v = float(np.mean(ys))
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
    else:
        source_u = -1.0
        source_v = -1.0
        width = 0.0
        height = 0.0

    depth_m = None
    valid_depth_ratio = 0.0
    if depth is not None and depth.shape[:2] == tracked_mask.shape[:2] and xs.size:
        depth_values = depth.astype(np.float32, copy=False)
        if np.issubdtype(depth.dtype, np.integer):
            depth_values = depth_values * 0.001
        valid = tracked_mask & np.isfinite(depth_values) & (depth_values > 0.0)
        values = depth_values[valid]
        valid_depth_ratio = float(values.size / max(1, int(np.count_nonzero(tracked_mask))))
        if values.size:
            depth_m = float(np.median(values))

    target_payload = {
        "valid": bool(xs.size and ys.size),
        "source_u": source_u,
        "source_v": source_v,
        "detection_width": width,
        "detection_height": height,
        "roi_width": width,
        "roi_height": height,
        "measurement_confidence": float(tracking_confidence),
        "valid_depth_ratio": valid_depth_ratio,
        "depth": depth_m,
        "point": {"x": 0.0, "y": 0.0, "z": depth_m or 0.0},
        "depth_source": "temporal_tracked_mask",
    }
    write_yaml(capture_dir / "target_3d.yaml", target_payload)
    write_yaml(
        capture_dir / "metadata.yaml",
        {
            "source": "offline_temporal_heavy_refresh",
            "dry_run": True,
            "real_arm_motion": False,
        },
    )


def run_heavy_refresh(capture_dir: Path, output_root: Path, device: str = "cpu") -> dict[str, Any]:
    grounding = run_on_capture(
        capture_dir=capture_dir,
        prompt=DEFAULT_PROMPT,
        output_root=output_root,
        repo_dir=DEFAULT_REPO_DIR,
        config_path=DEFAULT_CONFIG_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        box_threshold=DEFAULT_BOX_THRESHOLD,
        text_threshold=DEFAULT_TEXT_THRESHOLD,
        device=device,
        obstacle_prompt=DEFAULT_OBSTACLE_PROMPT,
        local_box_threshold=DEFAULT_LOCAL_BOX_THRESHOLD,
    )
    grounding_boxes = Path(grounding["outputs"]["boxes_yaml"])
    sam2 = refine_capture(
        capture_dir=capture_dir,
        groundingdino_boxes=grounding_boxes,
        output_root=output_root,
        repo_dir=DEFAULT_GROUNDED_SAM2_REPO_DIR,
        sam2_config=DEFAULT_SAM2_CONFIG,
        sam2_checkpoint=DEFAULT_SAM2_CHECKPOINT,
        device=device,
        max_masks=8,
    )
    masks = [mask for mask in sam2.get("masks", []) if isinstance(mask, dict)]
    targets = [mask for mask in masks if mask.get("mask_role") == "target" and mask.get("mask_png")]
    obstacles = [
        mask
        for mask in masks
        if mask.get("mask_role") == "obstacle" and mask.get("target_relevant_occluder", False)
    ]
    target_mask_path = str(targets[0]["mask_png"]) if targets else ""
    candidate_safe_obstacles = [mask for mask in obstacles if is_candidate_safe_obstacle(mask)]
    unsafe_obstacles = [
        mask
        for mask in obstacles
        if mask.get("is_unsafe_candidate", False) or mask not in candidate_safe_obstacles
    ]
    obstacle_masks = [
        {
            "label": str(mask.get("label", "unknown")),
            "confidence": float(mask.get("confidence", 0.0)),
            "mask_png": str(mask.get("mask_png", "")),
            "candidate_movable": bool(mask in candidate_safe_obstacles),
            "unsafe": bool(mask in unsafe_obstacles),
            "closer_than_target": bool(mask.get("closer_than_target", False)),
            "prompt_source": str(mask.get("prompt_source", "")),
        }
        for mask in obstacles
        if mask.get("mask_png")
    ]
    return {
        "status": "ok" if target_mask_path else "target_mask_missing",
        "target_source": grounding.get("summary", {}).get("target_source", "none"),
        "target_confidence": float(grounding.get("summary", {}).get("target_confidence", 0.0)),
        "target_mask_png": target_mask_path,
        "obstacle_count": len(obstacles),
        "obstacle_labels": [str(mask.get("label", "unknown")) for mask in obstacles],
        "obstacle_confidences": [float(mask.get("confidence", 0.0)) for mask in obstacles],
        "unsafe_obstacle_count": len(unsafe_obstacles),
        "candidate_safe_obstacle_count": len(candidate_safe_obstacles),
        "obstacle_masks": obstacle_masks,
        "groundingdino_boxes_yaml": str(grounding_boxes),
        "groundingdino_debug_png": str(grounding.get("outputs", {}).get("debug_png", "")),
        "sam2_masks_yaml": str(sam2.get("outputs", {}).get("masks_yaml", "")),
        "sam2_overlay_png": str(sam2.get("outputs", {}).get("overlay_png", "")),
        "dry_run": True,
        "real_arm_motion": False,
    }


def load_target_mask(result: dict[str, Any]) -> np.ndarray | None:
    path = Path(str(result.get("target_mask_png", "")))
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (mask > 0) if mask is not None else None
