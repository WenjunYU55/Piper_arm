#!/usr/bin/env python3
"""Run offline Grounded-SAM-2 bundled GroundingDINO inference on one saved L515 capture."""

from __future__ import annotations

import argparse
import importlib
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
DEFAULT_REPO_DIR = DEFAULT_GROUNDED_SAM2_REPO_DIR / "grounding_dino"
DEFAULT_CONFIG_PATH = DEFAULT_REPO_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_CHECKPOINT_PATH = SCRIPT_DIR / "weights" / "groundingdino_swint_ogc.pth"
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_LOCAL_BOX_THRESHOLD = 0.30
DEFAULT_TARGET_PROMPT = "green cube ."
DEFAULT_OBSTACLE_PROMPT = (
    "pen . | "
    "hand . finger ."
)
LOCAL_CROP_MIN_SIZE_PX = 256
LOCAL_CROP_HALF_EXTENT_SCALE = 3.0
MIN_TRACKED_MASK_FALLBACK_AREA_PX = 100
MIN_TRACKED_MASK_FALLBACK_CONFIDENCE = 0.35
MIN_TARGET_SEMANTIC_CONFIDENCE = 0.60
MIN_TARGET_GREEN_FRACTION = 0.15
MIN_TARGET_BOX_ASPECT_RATIO = 0.50
MAX_TARGET_BOX_ASPECT_RATIO = 1.60
TARGET_GREEN_HUE_MIN = 35
TARGET_GREEN_HUE_MAX = 95
TARGET_GREEN_SATURATION_MIN = 60
TARGET_GREEN_VALUE_MIN = 35
# Target selection is intentionally strict. Generic "cube" and "box" prompts
# can promote cardboard packaging to the target and poison the SAM2 session.
TARGET_TERMS = ("green cube",)
UNSAFE_TERMS = (
    "hand",
    "human",
    "person",
    "finger",
)
CANDIDATE_SAFE_TERMS = (
    "pen",
    "paper",
    "tissue",
    "wire",
    "cable",
    "cardboard",
)
LOCAL_GROUP_RELATIVE_CONFIDENCE = 0.75


class GroundingDinoUnavailable(RuntimeError):
    pass


def add_repo_to_path(repo_dir: Path) -> None:
    candidates = []
    if repo_dir.is_dir():
        candidates.append(repo_dir)
        if repo_dir.name == "grounding_dino":
            candidates.append(repo_dir.parent)
    for candidate in reversed(candidates):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def require_groundingdino(repo_dir: Path):
    add_repo_to_path(repo_dir)
    try:
        import torch
        from groundingdino.util.inference import annotate, load_image, load_model, predict
    except Exception as exc:
        raise GroundingDinoUnavailable(
            "GroundingDINO is not importable from the Grounded-SAM-2 checkout. Install the "
            "Grounded-SAM-2 dependencies in the isolated env, or set GROUNDINGDINO_REPO_DIR "
            "to its bundled grounding_dino folder. Original error: %s" % exc
        ) from exc

    deform_modules = []
    for module_name in (
        "groundingdino.models.GroundingDINO.ms_deform_attn",
        "grounding_dino.groundingdino.models.GroundingDINO.ms_deform_attn",
    ):
        try:
            deform_modules.append(importlib.import_module(module_name))
        except ImportError:
            continue
    for ms_deform_attn in deform_modules:
        if hasattr(ms_deform_attn, "_C"):
            continue

        class TorchCudaDeformableAttention:
            @staticmethod
            def ms_deform_attn_forward(
                value,
                value_spatial_shapes,
                _value_level_start_index,
                sampling_locations,
                attention_weights,
                _im2col_step,
            ):
                if not value.is_cuda:
                    raise RuntimeError("GroundingDINO fallback requires CUDA tensors")
                return ms_deform_attn.multi_scale_deformable_attn_pytorch(
                    value,
                    value_spatial_shapes,
                    sampling_locations,
                    attention_weights,
                )

        ms_deform_attn._C = TorchCudaDeformableAttention
    if not torch.cuda.is_available():
        raise GroundingDinoUnavailable("CUDA-only GroundingDINO requires torch.cuda.is_available()")
    return annotate, load_image, load_model, predict


def validate_inputs(capture_dir: Path, config_path: Path, checkpoint_path: Path) -> Path:
    rgb_path = capture_dir / "rgb.png"
    if not capture_dir.is_dir():
        raise FileNotFoundError("capture folder does not exist: %s" % capture_dir)
    if not rgb_path.is_file():
        raise FileNotFoundError("capture is missing rgb.png: %s" % rgb_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            "GroundingDINO config not found: %s. Pass --config or set GROUNDINGDINO_CONFIG_PATH." % config_path
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "GroundingDINO checkpoint not found: %s. Pass --checkpoint or set GROUNDINGDINO_CHECKPOINT_PATH." % checkpoint_path
        )
    return rgb_path


def tensor_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def box_cxcywh_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    x0 = (cx - bw / 2.0) * width
    y0 = (cy - bh / 2.0) * height
    x1 = (cx + bw / 2.0) * width
    y1 = (cy + bh / 2.0) * height
    return [
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    ]


def detection_records(
    boxes: Any,
    logits: Any,
    phrases: Any,
    width: int,
    height: int,
    detection_source: str = "full_frame",
    crop_origin: tuple[int, int] = (0, 0),
    full_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    boxes_list = tensor_to_list(boxes)
    logits_list = tensor_to_list(logits)
    phrases_list = list(phrases)
    records = []
    for index, box in enumerate(boxes_list):
        confidence = float(logits_list[index]) if index < len(logits_list) else 0.0
        label = str(phrases_list[index]) if index < len(phrases_list) else ""
        local_box_norm = [float(v) for v in box]
        x0, y0, x1, y1 = box_cxcywh_to_xyxy(local_box_norm, width, height)
        origin_x, origin_y = crop_origin
        x0 += origin_x
        x1 += origin_x
        y0 += origin_y
        y1 += origin_y
        full_width, full_height = full_size or (width, height)
        box_norm = [
            ((x0 + x1) / 2.0) / full_width,
            ((y0 + y1) / 2.0) / full_height,
            (x1 - x0) / full_width,
            (y1 - y0) / full_height,
        ]
        area_px = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        records.append(
            {
                "label": label,
                "confidence": confidence,
                "box_cxcywh_norm": box_norm,
                "box_xyxy_pixels": [x0, y0, x1, y1],
                "box_area_px": float(area_px),
                "detection_source": detection_source,
                "is_target_local_candidate": detection_source == "target_crop",
                "is_target_candidate": label_matches(label, TARGET_TERMS),
                "is_unsafe_candidate": label_matches(label, UNSAFE_TERMS),
                "is_candidate_safe_class": label_matches(label, CANDIDATE_SAFE_TERMS),
            }
        )
    return records


def label_matches(label: str, terms: tuple[str, ...]) -> bool:
    normalized = " " + label.lower().replace("_", " ") + " "
    return any((" " + term + " ") in normalized for term in terms)


def green_pixel_fraction(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Return the fraction of supported pixels matching the calibrated green."""
    if (
        image_bgr is None
        or mask is None
        or image_bgr.ndim != 3
        or image_bgr.shape[:2] != mask.shape[:2]
    ):
        return 0.0
    support = np.asarray(mask) > 0
    count = int(np.count_nonzero(support))
    if count <= 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= TARGET_GREEN_HUE_MIN)
        & (hsv[:, :, 0] <= TARGET_GREEN_HUE_MAX)
        & (hsv[:, :, 1] >= TARGET_GREEN_SATURATION_MIN)
        & (hsv[:, :, 2] >= TARGET_GREEN_VALUE_MIN)
    )
    return float(np.count_nonzero(green & support) / count)


def box_support_mask(shape: tuple[int, int], box: list[float]) -> np.ndarray:
    """Rasterize one clipped detection box for deterministic appearance checks."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(box) != 4:
        return mask
    height, width = shape
    x0, y0, x1, y1 = [float(value) for value in box]
    left = max(0, min(width, int(np.floor(x0))))
    top = max(0, min(height, int(np.floor(y0))))
    right = max(0, min(width, int(np.ceil(x1))))
    bottom = max(0, min(height, int(np.ceil(y1))))
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 255
    return mask


def target_mask_appearance_validation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Validate that a proposed target mask contains enough observed green."""
    green_fraction = green_pixel_fraction(image_bgr, mask)
    accepted = green_fraction >= MIN_TARGET_GREEN_FRACTION
    return {
        "accepted": bool(accepted),
        "green_fraction": float(green_fraction),
        "minimum_green_fraction": float(MIN_TARGET_GREEN_FRACTION),
        "rejection_reasons": [] if accepted else [
            "green fraction %.3f is below %.3f"
            % (green_fraction, MIN_TARGET_GREEN_FRACTION)
        ],
    }


def target_detection_validation(
    detection: dict[str, Any],
    image_bgr: np.ndarray,
) -> dict[str, Any]:
    """Validate semantic confidence, observed colour, and box proportions."""
    confidence = float(detection.get("confidence", 0.0))
    box = detection.get("box_xyxy_pixels", [])
    mask = box_support_mask(image_bgr.shape[:2], box)
    appearance = target_mask_appearance_validation(image_bgr, mask)
    aspect_ratio = 0.0
    if len(box) == 4:
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
        aspect_ratio = width / height if height > 0.0 else 0.0
    reasons = []
    if confidence < MIN_TARGET_SEMANTIC_CONFIDENCE:
        reasons.append(
            "semantic confidence %.3f is below %.3f"
            % (confidence, MIN_TARGET_SEMANTIC_CONFIDENCE)
        )
    reasons.extend(appearance["rejection_reasons"])
    if not (
        MIN_TARGET_BOX_ASPECT_RATIO
        <= aspect_ratio
        <= MAX_TARGET_BOX_ASPECT_RATIO
    ):
        reasons.append(
            "box aspect ratio %.3f is outside [%.3f, %.3f]"
            % (
                aspect_ratio,
                MIN_TARGET_BOX_ASPECT_RATIO,
                MAX_TARGET_BOX_ASPECT_RATIO,
            )
        )
    return {
        "accepted": not reasons,
        "semantic_confidence": confidence,
        "minimum_semantic_confidence": MIN_TARGET_SEMANTIC_CONFIDENCE,
        "green_fraction": appearance["green_fraction"],
        "minimum_green_fraction": MIN_TARGET_GREEN_FRACTION,
        "box_aspect_ratio": aspect_ratio,
        "minimum_box_aspect_ratio": MIN_TARGET_BOX_ASPECT_RATIO,
        "maximum_box_aspect_ratio": MAX_TARGET_BOX_ASPECT_RATIO,
        "rejection_reasons": reasons,
    }


def validate_target_detections(
    detections: list[dict[str, Any]],
    image_bgr: np.ndarray,
) -> list[dict[str, Any]]:
    """Mark semantic target candidates valid only after appearance checks."""
    rejected = []
    for detection in detections:
        semantic_candidate = bool(detection.get("is_target_candidate", False))
        detection["is_semantic_target_candidate"] = semantic_candidate
        if not semantic_candidate:
            continue
        validation = target_detection_validation(detection, image_bgr)
        detection["target_validation"] = validation
        detection["is_target_candidate"] = bool(validation["accepted"])
        if not validation["accepted"]:
            rejected.append(detection)
    return rejected


def best_detection(detections: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [record for record in detections if record.get(key)]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(record.get("confidence", 0.0)))


def detection_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    target = best_detection(detections, "is_target_candidate")
    obstacles = [
        record
        for record in detections
        if not record.get("is_target_candidate") and record.get("is_target_local_candidate")
    ]
    unsafe = [record for record in obstacles if record.get("is_unsafe_candidate")]
    candidate_safe = [record for record in obstacles if record.get("is_candidate_safe_class")]
    return {
        "best_target_detection": target,
        "target_detected": target is not None,
        "target_confidence": float(target.get("confidence", 0.0)) if target else 0.0,
        "obstacle_candidates": obstacles,
        "unsafe_candidates": unsafe,
        "candidate_safe_class_detections": candidate_safe,
        "has_unsafe_candidate": len(unsafe) > 0,
        "has_candidate_safe_class": len(candidate_safe) > 0,
    }


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def tracked_mask_target_fallback(capture_dir: Path) -> dict[str, Any] | None:
    mask = cv2.imread(str(capture_dir / "detection_mask.png"), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(capture_dir / "rgb.png"), cv2.IMREAD_COLOR)
    target = read_yaml(capture_dir / "target_3d.yaml")
    tracking_confidence = float(target.get("measurement_confidence", 0.0))
    if (
        mask is None
        or image is None
        or image.shape[:2] != mask.shape[:2]
        or not bool(target.get("valid", False))
        or tracking_confidence < MIN_TRACKED_MASK_FALLBACK_CONFIDENCE
    ):
        return None
    source_u = int(round(float(target.get("source_u", -1))))
    source_v = int(round(float(target.get("source_v", -1))))
    if not (0 <= source_u < mask.shape[1] and 0 <= source_v < mask.shape[0]):
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), connectivity=8)
    if count <= 1:
        return None
    component = int(labels[source_v, source_u])
    if component <= 0 or int(stats[component, cv2.CC_STAT_AREA]) < MIN_TRACKED_MASK_FALLBACK_AREA_PX:
        return None
    component_mask = (labels == component).astype(np.uint8) * 255
    appearance = target_mask_appearance_validation(image, component_mask)
    if not appearance["accepted"]:
        return None
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    width = int(stats[component, cv2.CC_STAT_WIDTH])
    height = int(stats[component, cv2.CC_STAT_HEIGHT])
    return {
        "label": "tracked target mask fallback",
        "confidence": tracking_confidence,
        "box_xyxy_pixels": [float(x), float(y), float(x + width), float(y + height)],
        "box_area_px": float(width * height),
        "detection_source": "tracked_target_mask",
        "is_target_local_candidate": False,
        "is_target_candidate": True,
        "is_unsafe_candidate": False,
        "is_candidate_safe_class": False,
        "is_semantic_target_candidate": False,
        "target_validation": dict(
            appearance,
            source="tracked_target_mask",
            semantic_confidence=tracking_confidence,
        ),
    }


def target_crop_bounds(target_box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in target_box]
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    half_width = max(
        LOCAL_CROP_MIN_SIZE_PX / 2.0,
        (x1 - x0) * LOCAL_CROP_HALF_EXTENT_SCALE,
    )
    half_height = max(
        LOCAL_CROP_MIN_SIZE_PX / 2.0,
        (y1 - y0) * LOCAL_CROP_HALF_EXTENT_SCALE,
    )
    crop_x0 = max(0, int(round(center_x - half_width)))
    crop_y0 = max(0, int(round(center_y - half_height)))
    crop_x1 = min(width, int(round(center_x + half_width)))
    crop_y1 = min(height, int(round(center_y + half_height)))
    return crop_x0, crop_y0, crop_x1, crop_y1


def box_iou(first: list[float], second: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in first]
    bx0, by0, bx1, by1 = [float(value) for value in second]
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def suppress_duplicate_boxes(detections: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    kept = []
    for detection in sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        box = detection.get("box_xyxy_pixels", [])
        if len(box) != 4:
            continue
        if any(box_iou(box, existing["box_xyxy_pixels"]) >= iou_threshold for existing in kept):
            continue
        kept.append(detection)
    return kept


def parse_prompt_groups(prompt: str) -> list[str]:
    groups = [group.strip() for group in prompt.split("|") if group.strip()]
    return groups or [prompt.strip()]


def retain_strong_group_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not detections:
        return []
    strongest = max(float(detection.get("confidence", 0.0)) for detection in detections)
    minimum = strongest * LOCAL_GROUP_RELATIVE_CONFIDENCE
    return [detection for detection in detections if float(detection.get("confidence", 0.0)) >= minimum]


def draw_local_detections(image: Any, detections: list[dict[str, Any]], path: Path) -> None:
    debug = image.copy()
    for record in detections:
        x0, y0, x1, y1 = [int(round(value)) for value in record["box_xyxy_pixels"]]
        cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 0, 255), 2)
        text = "%s %.2f" % (record.get("label", ""), float(record.get("confidence", 0.0)))
        cv2.putText(debug, text, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(str(path), debug)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def run_on_capture(
    capture_dir: Path,
    prompt: str,
    output_root: Path,
    repo_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    box_threshold: float,
    text_threshold: float,
    device: str,
    obstacle_prompt: str = DEFAULT_OBSTACLE_PROMPT,
    local_box_threshold: float = DEFAULT_LOCAL_BOX_THRESHOLD,
) -> dict[str, Any]:
    capture_dir = capture_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    repo_dir = repo_dir.expanduser().resolve()
    rgb_path = validate_inputs(capture_dir, config_path, checkpoint_path)
    annotate, load_image, load_model, predict = require_groundingdino(repo_dir)

    output_dir = output_root.expanduser().resolve() / capture_dir.name / "groundingdino"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(str(config_path), str(checkpoint_path), device=device)
    image_source, image = load_image(str(rgb_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        device=device,
    )
    height, width = image_source.shape[:2]
    full_frame_detections = detection_records(boxes, logits, phrases, width, height)
    rejected_target_candidates = validate_target_detections(
        full_frame_detections,
        image_source,
    )
    model_target = best_detection(full_frame_detections, "is_target_candidate")
    target = model_target or tracked_mask_target_fallback(capture_dir)
    local_detections: list[dict[str, Any]] = []
    local_prompt_results: list[dict[str, Any]] = []
    crop_path = output_dir / "target_crop.png"
    local_debug_path = output_dir / "target_crop_detections.png"
    crop_bounds = None
    if target is not None:
        crop_bounds = target_crop_bounds(target["box_xyxy_pixels"], width, height)
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_bounds
        crop_bgr = image_source[crop_y0:crop_y1, crop_x0:crop_x1]
        if crop_bgr.size > 0:
            cv2.imwrite(str(crop_path), crop_bgr)
            _, crop_image = load_image(str(crop_path))
            crop_height, crop_width = crop_bgr.shape[:2]
            grouped_detections = []
            for prompt_group in parse_prompt_groups(obstacle_prompt):
                local_boxes, local_logits, local_phrases = predict(
                    model=model,
                    image=crop_image,
                    caption=prompt_group,
                    box_threshold=float(local_box_threshold),
                    text_threshold=float(text_threshold),
                    device=device,
                )
                group_detections = detection_records(
                    local_boxes,
                    local_logits,
                    local_phrases,
                    crop_width,
                    crop_height,
                    detection_source="target_crop",
                    crop_origin=(crop_x0, crop_y0),
                    full_size=(width, height),
                )
                for detection in group_detections:
                    detection["prompt_group"] = prompt_group
                retained_detections = retain_strong_group_detections(group_detections)
                grouped_detections.extend(retained_detections)
                local_prompt_results.append(
                    {
                        "prompt": prompt_group,
                        "raw_detection_count": len(group_detections),
                        "retained_detection_count": len(retained_detections),
                    }
                )
            local_detections = suppress_duplicate_boxes(grouped_detections)
            draw_local_detections(image_source, local_detections, local_debug_path)

    detections = list(full_frame_detections)
    if model_target is None and target is not None:
        detections.append(target)
    detections.extend(local_detections)
    summary = detection_summary(detections)
    summary["model_target_detected"] = model_target is not None
    summary["target_source"] = "groundingdino" if model_target is not None else ("tracked_target_mask" if target is not None else "none")
    summary["rejected_target_candidates"] = rejected_target_candidates
    summary["target_validation_policy"] = {
        "minimum_semantic_confidence": MIN_TARGET_SEMANTIC_CONFIDENCE,
        "minimum_green_fraction": MIN_TARGET_GREEN_FRACTION,
        "box_aspect_ratio": [
            MIN_TARGET_BOX_ASPECT_RATIO,
            MAX_TARGET_BOX_ASPECT_RATIO,
        ],
        "fail_closed": True,
    }

    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    debug_path = output_dir / "groundingdino_debug.png"
    cv2.imwrite(str(debug_path), annotated)

    payload = {
        "backend": "IDEA-Research/Grounded-SAM-2 bundled GroundingDINO",
        "grounded_sam2_repo": str(DEFAULT_GROUNDED_SAM2_REPO_DIR),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_name": capture_dir.name,
        "capture_path": str(capture_dir),
        "source_rgb_path": str(rgb_path),
        "prompt": prompt,
        "obstacle_prompt": obstacle_prompt,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "local_box_threshold": float(local_box_threshold),
        "device": device,
        "image_width": int(width),
        "image_height": int(height),
        "detected_labels": [record["label"] for record in detections],
        "detections": detections,
        "full_frame_detections": full_frame_detections,
        "target_local_detections": local_detections,
        "target_local_prompt_results": local_prompt_results,
        "target_crop_bounds_xyxy": list(crop_bounds) if crop_bounds is not None else None,
        "summary": summary,
        "outputs": {
            "boxes_yaml": str(output_dir / "groundingdino_boxes.yaml"),
            "debug_png": str(debug_path),
            "target_crop_png": str(crop_path) if crop_path.is_file() else "",
            "target_crop_debug_png": str(local_debug_path) if local_debug_path.is_file() else "",
        },
    }
    write_yaml(output_dir / "groundingdino_boxes.yaml", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline GroundingDINO on one saved L515 capture.")
    parser.add_argument("capture_folder", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GROUNDINGDINO_REPO_DIR", DEFAULT_REPO_DIR)))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CONFIG_PATH", DEFAULT_CONFIG_PATH)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--obstacle-prompt", default=DEFAULT_OBSTACLE_PROMPT)
    parser.add_argument("--local-box-threshold", type=float, default=DEFAULT_LOCAL_BOX_THRESHOLD)
    parser.add_argument("--device", default=os.environ.get("GROUNDINGDINO_DEVICE", "cpu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_on_capture(
            capture_dir=args.capture_folder,
            prompt=args.prompt,
            output_root=args.output_root,
            repo_dir=args.repo_dir,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
            obstacle_prompt=args.obstacle_prompt,
            local_box_threshold=args.local_box_threshold,
        )
    except GroundingDinoUnavailable as exc:
        print("GroundingDINO unavailable:", exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print("GroundingDINO inference failed:", exc, file=sys.stderr)
        return 1
    print("Detections:", len(result["detections"]))
    print("Boxes YAML:", result["outputs"]["boxes_yaml"])
    print("Debug image:", result["outputs"]["debug_png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
