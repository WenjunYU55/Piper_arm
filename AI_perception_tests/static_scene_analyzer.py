#!/usr/bin/env python3
"""Offline static RGB-D capture analyzer."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import yaml


EXPECTED_FILES = (
    "rgb.png",
    "depth.npy",
    "detection_mask.png",
    "camera_info.yaml",
    "target_3d.yaml",
    "metadata.yaml",
)

OPTIONAL_FILES = (
    "scan_quality.yaml",
    "occlusion_status.yaml",
)

DEFAULT_PARAMS = {
    "occlusion_depth_margin_m": 0.03,
    "near_mask_dilation_px": 20,
    "min_occluder_area_px": 80,
    "partial_occlusion_ratio": 0.05,
    "heavy_occlusion_ratio": 0.20,
    "min_valid_depth_ratio": 0.40,
    "min_mask_area_px": 100,
    "min_valid_depth_m": 0.15,
    "max_valid_depth_m": 1.20,
}


def file_status(capture_dir: Path) -> dict[str, Any]:
    status = {}
    for filename in EXPECTED_FILES + OPTIONAL_FILES:
        path = capture_dir / filename
        status[filename] = {
            "exists": path.is_file(),
            "required": filename in EXPECTED_FILES,
            "path": str(path),
        }
    return status


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32, copy=False) * 0.001

    depth_m = depth.astype(np.float32, copy=False)
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    if finite.size > 0 and float(np.nanmedian(finite)) > 20.0:
        return depth_m * 0.001
    return depth_m


def valid_depth_mask(depth_m: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return (
        np.isfinite(depth_m)
        & (depth_m >= float(params["min_valid_depth_m"]))
        & (depth_m <= float(params["max_valid_depth_m"]))
    )


def finite_positive(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric) and numeric > 0.0:
        return numeric
    return None


def target_depth_from_yaml(target_3d: dict[str, Any]) -> Tuple[Optional[float], str]:
    for key in ("depth",):
        value = finite_positive(target_3d.get(key))
        if value is not None:
            return value, f"target_3d.{key}"

    point = target_3d.get("point")
    if isinstance(point, dict):
        value = finite_positive(point.get("z"))
        if value is not None:
            return value, "target_3d.point.z"

    return None, "unavailable"


def estimate_target_depth(
    depth_m: np.ndarray, mask_bool: np.ndarray, valid_mask: np.ndarray
) -> Tuple[Optional[float], str]:
    masked = depth_m[mask_bool & valid_mask]
    if masked.size == 0:
        return None, "unavailable"
    return float(np.median(masked)), "masked_depth_median"


def dilated_near_mask(mask_bool: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return mask_bool.copy()
    kernel_size = 2 * int(dilation_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_bool.astype(np.uint8) * 255, kernel, iterations=1)
    return dilated > 0


def remove_small_regions(mask_bool: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    cleaned = np.zeros(mask_bool.shape, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = True
    return cleaned


def classify_static(metrics: dict[str, Any], params: dict[str, Any]) -> Tuple[str, str]:
    mask_area = int(metrics["mask_area_px"])
    valid_ratio = float(metrics["valid_depth_ratio"])
    target_depth = metrics["target_depth_m"]
    closer_area = int(metrics["closer_region_area_px"])
    closer_ratio = float(metrics["closer_region_ratio"])
    min_area = int(params["min_occluder_area_px"])

    if mask_area < int(params["min_mask_area_px"]):
        return "LOST", "target mask missing or too small"
    if target_depth is None:
        return "UNKNOWN", "target depth unavailable"
    if valid_ratio < float(params["min_valid_depth_ratio"]) * 0.5:
        return "LOST", "too few valid depth pixels inside target mask"
    if closer_area >= min_area and closer_ratio >= float(params["heavy_occlusion_ratio"]):
        return "BLOCKED", "large foreground region closer than target depth"
    if closer_area >= min_area and closer_ratio >= float(params["partial_occlusion_ratio"]):
        return "PARTIAL", "foreground region near target mask"
    if valid_ratio < float(params["min_valid_depth_ratio"]):
        return "UNKNOWN", "valid target depth ratio is low"
    return "CLEAR", "no significant foreground closer region"


def draw_contours(image: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)


def write_debug_overlay(
    output_path: Path,
    rgb: np.ndarray,
    target_mask: np.ndarray,
    closer_mask: np.ndarray,
    metrics: dict[str, Any],
    decision: str,
) -> None:
    overlay = rgb.copy()
    draw_contours(overlay, target_mask, (0, 255, 0), 2)
    draw_contours(overlay, closer_mask, (0, 0, 255), 2)

    target_depth = metrics["target_depth_m"]
    target_depth_text = "n/a" if target_depth is None else f"{target_depth:.3f} m"
    lines = [
        f"mask area: {metrics['mask_area_px']} px",
        f"target depth: {target_depth_text}",
        f"valid depth ratio: {metrics['valid_depth_ratio']:.3f}",
        f"closer area: {metrics['closer_region_area_px']} px",
        f"closer ratio: {metrics['closer_region_ratio']:.3f}",
        f"decision: {decision}",
    ]

    x, y = 12, 26
    for line in lines:
        cv2.putText(overlay, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    cv2.imwrite(str(output_path), overlay)


def analyze_capture(capture_dir: Path, output_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    files = file_status(capture_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis: dict[str, Any] = {
        "analysis_type": "static_rgbd_scene_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_folder": str(capture_dir),
        "output_folder": str(output_dir),
        "files": files,
        "parameters": params,
        "metrics": {},
        "decision": "UNKNOWN",
        "reason": "",
        "outputs": {
            "analysis_yaml": str(output_dir / "analysis.yaml"),
            "debug_overlay_png": str(output_dir / "debug_overlay.png"),
            "foreground_closer_mask_png": str(output_dir / "foreground_closer_mask.png"),
        },
    }

    missing_required = [name for name in EXPECTED_FILES if not files[name]["exists"]]
    if missing_required:
        analysis["reason"] = "missing required files: " + ", ".join(missing_required)
        return analysis

    rgb = cv2.imread(str(capture_dir / "rgb.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(capture_dir / "detection_mask.png"), cv2.IMREAD_GRAYSCALE)
    depth = np.load(str(capture_dir / "depth.npy"))
    target_3d = read_yaml(capture_dir / "target_3d.yaml")

    if rgb is None:
        analysis["reason"] = "failed to read rgb.png"
        return analysis
    if mask is None:
        analysis["reason"] = "failed to read detection_mask.png"
        return analysis
    if depth.shape[:2] != mask.shape[:2]:
        analysis["reason"] = "depth.npy and detection_mask.png dimensions differ"
        return analysis
    if rgb.shape[:2] != mask.shape[:2]:
        analysis["reason"] = "rgb.png and detection_mask.png dimensions differ"
        return analysis

    mask_bool = mask > 0
    mask_area = int(np.count_nonzero(mask_bool))
    depth_m = depth_to_meters(np.asarray(depth))
    valid_mask = valid_depth_mask(depth_m, params)
    valid_target_depth = depth_m[mask_bool & valid_mask]
    valid_depth_count = int(valid_target_depth.size)
    valid_depth_ratio = float(valid_depth_count / max(1, mask_area))
    depth_mean = float(np.mean(valid_target_depth)) if valid_depth_count else None
    depth_stddev = float(np.std(valid_target_depth)) if valid_depth_count else None

    target_depth, target_depth_source = target_depth_from_yaml(target_3d)
    if target_depth is None:
        target_depth, target_depth_source = estimate_target_depth(depth_m, mask_bool, valid_mask)

    if target_depth is None:
        foreground_closer = np.zeros(mask_bool.shape, dtype=bool)
    else:
        near_mask = dilated_near_mask(mask_bool, int(params["near_mask_dilation_px"]))
        search_region = near_mask & ~mask_bool
        foreground_closer = (
            search_region
            & valid_mask
            & (depth_m < float(target_depth) - float(params["occlusion_depth_margin_m"]))
        )
        foreground_closer = remove_small_regions(foreground_closer, int(params["min_occluder_area_px"]))

    closer_area = int(np.count_nonzero(foreground_closer))
    near_area = int(np.count_nonzero(dilated_near_mask(mask_bool, int(params["near_mask_dilation_px"])) & ~mask_bool))
    closer_ratio = float(closer_area / max(1, near_area))

    metrics = {
        "mask_area_px": mask_area,
        "target_depth_m": target_depth,
        "target_depth_source": target_depth_source,
        "valid_depth_ratio": valid_depth_ratio,
        "valid_depth_count_px": valid_depth_count,
        "depth_mean_m": depth_mean,
        "depth_stddev_m": depth_stddev,
        "foreground_closer_mask": "foreground_closer_mask.png",
        "near_region_area_px": near_area,
        "closer_region_area_px": closer_area,
        "closer_region_ratio": closer_ratio,
    }
    decision, reason = classify_static(metrics, params)

    cv2.imwrite(str(output_dir / "foreground_closer_mask.png"), foreground_closer.astype(np.uint8) * 255)
    write_debug_overlay(output_dir / "debug_overlay.png", rgb, mask_bool, foreground_closer, metrics, decision)

    analysis["metrics"] = metrics
    analysis["decision"] = decision
    analysis["reason"] = reason
    return analysis


def print_summary(analysis: dict[str, Any]) -> None:
    print(f"Capture folder: {analysis['capture_folder']}")
    print(f"Output folder: {analysis['output_folder']}")
    print(f"Decision: {analysis['decision']}")
    if analysis.get("reason"):
        print(f"Reason: {analysis['reason']}")

    metrics = analysis.get("metrics") or {}
    if metrics:
        print(f"Mask area px: {metrics['mask_area_px']}")
        print(f"Target depth m: {metrics['target_depth_m']}")
        print(f"Valid depth ratio: {metrics['valid_depth_ratio']:.3f}")
        print(f"Closer area px: {metrics['closer_region_area_px']}")
        print(f"Closer ratio: {metrics['closer_region_ratio']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a saved RGB-D capture folder offline.")
    parser.add_argument("capture_folder", type=Path, help="Path to a saved RGB-D capture folder.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root folder for analysis outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = args.capture_folder.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / capture_dir.name

    analysis = analyze_capture(capture_dir, output_dir, dict(DEFAULT_PARAMS))
    analysis_path = output_dir / "analysis.yaml"
    write_yaml(analysis_path, analysis)

    print_summary(analysis)
    print(f"Analysis written to: {analysis_path}")
    return 0 if analysis["decision"] != "UNKNOWN" or analysis.get("metrics") else 1


if __name__ == "__main__":
    raise SystemExit(main())
