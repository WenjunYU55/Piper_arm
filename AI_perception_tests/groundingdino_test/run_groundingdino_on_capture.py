#!/usr/bin/env python3
"""Run offline Grounded-SAM-2 bundled GroundingDINO inference on one saved L515 capture."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
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
        from groundingdino.util.inference import annotate, load_image, load_model, predict
    except Exception as exc:
        raise GroundingDinoUnavailable(
            "GroundingDINO is not importable from the Grounded-SAM-2 checkout. Install the "
            "Grounded-SAM-2 dependencies in the isolated env, or set GROUNDINGDINO_REPO_DIR "
            "to its bundled grounding_dino folder. Original error: %s" % exc
        ) from exc
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


def detection_records(boxes: Any, logits: Any, phrases: Any, width: int, height: int) -> list[dict[str, Any]]:
    boxes_list = tensor_to_list(boxes)
    logits_list = tensor_to_list(logits)
    phrases_list = list(phrases)
    records = []
    for index, box in enumerate(boxes_list):
        confidence = float(logits_list[index]) if index < len(logits_list) else 0.0
        label = str(phrases_list[index]) if index < len(phrases_list) else ""
        box_norm = [float(v) for v in box]
        records.append(
            {
                "label": label,
                "confidence": confidence,
                "box_cxcywh_norm": box_norm,
                "box_xyxy_pixels": box_cxcywh_to_xyxy(box_norm, width, height),
            }
        )
    return records


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
    detections = detection_records(boxes, logits, phrases, width, height)

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
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "device": device,
        "image_width": int(width),
        "image_height": int(height),
        "detected_labels": [record["label"] for record in detections],
        "detections": detections,
        "outputs": {
            "boxes_yaml": str(output_dir / "groundingdino_boxes.yaml"),
            "debug_png": str(debug_path),
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
