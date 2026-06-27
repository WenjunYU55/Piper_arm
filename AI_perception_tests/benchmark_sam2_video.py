#!/usr/bin/env python3
"""Benchmark pretrained SAM2 video mask propagation on a saved active scan."""

from __future__ import annotations

import argparse
import resource
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
GROUNDING_DIR = SCRIPT_DIR / "groundingdino_test"
SAM2_REPO = GROUNDING_DIR / "Grounded-SAM-2"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = GROUNDING_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def find_default_seed(scan_name: str) -> Path:
    candidates = sorted(
        (
            SCRIPT_DIR
            / "outputs"
            / "sam2_video"
            / (scan_name + "_heavy")
            / "heavy_outputs"
            / "frame_000"
            / "sam2"
        ).glob("mask_*_target.png")
    )
    if not candidates:
        raise FileNotFoundError("no frame-0 SAM2 target mask found for %s" % scan_name)
    return candidates[0]


def prepare_jpegs(rgb_paths: list[Path], directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    for index, source in enumerate(rgb_paths):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unreadable RGB frame: %s" % source)
        if not cv2.imwrite(str(directory / ("%05d.jpg" % index)), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError("could not write benchmark JPEG %d" % index)


def run_benchmark(scan_dir: Path, seed_mask_path: Path, output_dir: Path) -> dict:
    frames_dir = scan_dir / "frames"
    rgb_paths = sorted(frames_dir.glob("view_*_rgb.png"))
    if not rgb_paths:
        raise FileNotFoundError("no view_*_rgb.png frames in %s" % frames_dir)
    seed = cv2.imread(str(seed_mask_path), cv2.IMREAD_GRAYSCALE)
    first = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if seed is None or first is None or seed.shape != first.shape[:2]:
        raise ValueError("seed mask must match the first RGB frame")

    output_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir = output_dir / "jpeg_frames"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    prepare_jpegs(rgb_paths, jpeg_dir)
    shutil.rmtree(mask_dir, ignore_errors=True)
    shutil.rmtree(overlay_dir, ignore_errors=True)
    mask_dir.mkdir()
    overlay_dir.mkdir()

    if str(SAM2_REPO) not in sys.path:
        sys.path.insert(0, str(SAM2_REPO))
    from sam2.build_sam import build_sam2_video_predictor

    build_started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cpu", apply_postprocessing=True
    )
    model_build_sec = time.perf_counter() - build_started
    init_started = time.perf_counter()
    state = predictor.init_state(video_path=str(jpeg_dir), offload_video_to_cpu=True)
    state_init_sec = time.perf_counter() - init_started
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed > 0)

    frame_records = []
    propagation_started = time.perf_counter()
    previous_yield = propagation_started
    with torch.inference_mode():
        for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
            now = time.perf_counter()
            prediction = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            cv2.imwrite(str(mask_dir / ("frame_%03d.png" % frame_index)), prediction.astype(np.uint8) * 255)
            rgb = cv2.imread(str(rgb_paths[frame_index]), cv2.IMREAD_COLOR)
            overlay = rgb.copy()
            overlay[prediction] = (0.45 * overlay[prediction] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.imwrite(str(overlay_dir / ("frame_%03d.png" % frame_index)), overlay)
            reference_path = frames_dir / ("view_%03d_mask.png" % frame_index)
            reference_image = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
            iou = mask_iou(prediction, reference_image > 0) if reference_image is not None else None
            frame_records.append(
                {
                    "frame_index": int(frame_index),
                    "object_ids": [int(item) for item in object_ids],
                    "mask_area_px": int(np.count_nonzero(prediction)),
                    "reference_iou": iou,
                    "yield_interval_sec": float(now - previous_yield),
                }
            )
            previous_yield = now
    propagation_sec = time.perf_counter() - propagation_started
    evaluated = [record["reference_iou"] for record in frame_records if record["reference_iou"] is not None]
    report = {
        "scan_dir": str(scan_dir),
        "seed_mask": str(seed_mask_path),
        "sam2_config": SAM2_CONFIG,
        "sam2_checkpoint": str(SAM2_CHECKPOINT),
        "device": "cpu",
        "frame_count": len(frame_records),
        "model_build_sec": float(model_build_sec),
        "state_init_sec": float(state_init_sec),
        "propagation_sec": float(propagation_sec),
        "mean_propagation_fps": float(len(frame_records) / propagation_sec) if propagation_sec else 0.0,
        "mean_reference_iou": float(np.mean(evaluated)) if evaluated else None,
        "min_reference_iou": float(np.min(evaluated)) if evaluated else None,
        "max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "frames": frame_records,
        "dry_run": True,
        "real_arm_motion": False,
    }
    with (output_dir / "sam2_video_benchmark.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--seed-mask", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    scan_dir = args.scan_dir.expanduser().resolve()
    seed = args.seed_mask or find_default_seed(scan_dir.name)
    output = args.output_dir or (SCRIPT_DIR / "outputs" / "sam2_video_benchmark" / scan_dir.name)
    result = run_benchmark(scan_dir, seed.expanduser().resolve(), output.expanduser().resolve())
    print(yaml.safe_dump({key: value for key, value in result.items() if key != "frames"}, sort_keys=False))


if __name__ == "__main__":
    main()
