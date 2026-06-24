#!/usr/bin/env python3
"""Run lightweight target-mask tracking over a saved active-scan RGB-D sequence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from temporal_tracking import TemporalMaskTracker, TemporalTrackerConfig, largest_component


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "temporal_tracking"


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def discover_frames(scan_dir: Path) -> list[dict[str, Path | int]]:
    frames_dir = scan_dir / "frames" if (scan_dir / "frames").is_dir() else scan_dir
    frames = []
    for rgb_path in sorted(frames_dir.glob("view_*_rgb.png")):
        prefix = rgb_path.name[: -len("_rgb.png")]
        try:
            index = int(prefix.split("_")[-1])
        except ValueError:
            continue
        frames.append(
            {
                "index": index,
                "rgb": rgb_path,
                "depth": frames_dir / (prefix + "_depth.npy"),
                "mask": frames_dir / (prefix + "_mask.png"),
                "metadata": frames_dir / (prefix + "_metadata.yaml"),
            }
        )
    return frames


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read RGB frame: %s" % path)
    return image


def read_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (mask > 0) if mask is not None else None


def read_depth(path: Path) -> np.ndarray | None:
    return np.load(str(path)) if path.is_file() else None


def clean_output(output_dir: Path) -> None:
    if not output_dir.is_dir():
        return
    for pattern in ("frame_*_mask.png", "frame_*_overlay.png"):
        for path in output_dir.glob(pattern):
            path.unlink()


def mask_iou(first: np.ndarray, second: np.ndarray | None) -> float | None:
    if second is None or first.shape[:2] != second.shape[:2]:
        return None
    first_target = largest_component(first)
    second_target = largest_component(second)
    intersection = int(np.count_nonzero(first_target & second_target))
    union = int(np.count_nonzero(first_target | second_target))
    return float(intersection / union) if union else None


def write_overlay(image: np.ndarray, mask: np.ndarray, record: dict[str, Any], path: Path) -> None:
    overlay = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (0, 255, 0) if record["target_valid"] else (0, 0, 255)
    cv2.drawContours(overlay, contours, -1, color, 2)
    lines = [
        "%s conf=%.2f" % (record["mode"], record["tracking_confidence"]),
        "refresh=%s obstacle=%s" % (
            record["heavy_refresh_requested"],
            record["obstacle_persistent"],
        ),
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            overlay,
            text,
            (10, 24 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (10, 24 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), overlay)


def run_sequence(
    scan_dir: Path,
    output_dir: Path,
    config: TemporalTrackerConfig,
    initial_mask_path: Path | None,
    simulate_saved_mask_refresh: bool,
    execute_heavy_refresh: bool,
    heavy_device: str,
) -> dict[str, Any]:
    frames = discover_frames(scan_dir)
    if not frames:
        raise RuntimeError("no view_NNN_rgb.png frames found under %s" % scan_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output(output_dir)
    if simulate_saved_mask_refresh and execute_heavy_refresh:
        raise ValueError("saved-mask refresh simulation and real heavy refresh are mutually exclusive")
    if execute_heavy_refresh:
        from temporal_heavy_refresh import (
            load_target_mask,
            prepare_event_capture,
            run_heavy_refresh,
        )

    tracker = TemporalMaskTracker(config)
    records = []
    refresh_request_count = 0
    refresh_applied_count = 0
    lost_count = 0
    persistent_obstacle_count = 0
    heavy_events = []

    for sequence_index, frame in enumerate(frames):
        rgb = read_rgb(frame["rgb"])
        depth = read_depth(frame["depth"])
        saved_mask = read_mask(frame["mask"])
        if sequence_index == 0:
            initial_mask = read_mask(initial_mask_path) if initial_mask_path is not None else saved_mask
            if initial_mask is None:
                raise RuntimeError("first frame has no initial mask; pass --initial-mask")
            heavy_event = None
            if execute_heavy_refresh:
                event_capture = output_dir / "heavy_inputs" / ("frame_%03d" % int(frame["index"]))
                prepare_event_capture(event_capture, frame["rgb"], depth, initial_mask, 1.0)
                try:
                    heavy_event = run_heavy_refresh(
                        event_capture,
                        output_dir / "heavy_outputs",
                        device=heavy_device,
                    )
                except Exception as exc:
                    heavy_event = {"status": "failed", "error": str(exc)}
                heavy_event.update(
                    {
                        "source_frame_index": int(frame["index"]),
                        "trigger": "initialization",
                    }
                )
                heavy_events.append(heavy_event)
                heavy_mask = load_target_mask(heavy_event)
                if heavy_mask is None:
                    raise RuntimeError("heavy initialization failed: %s" % heavy_event)
                initial_mask = heavy_mask
            result = tracker.initialize(rgb, initial_mask, depth)
            if execute_heavy_refresh:
                result.mode = "HEAVY_INITIALIZED"
            tracked_mask = tracker.mask.copy()
            refresh_applied = False
        else:
            result, tracked_mask = tracker.step(rgb, depth)
            refresh_applied = False
            heavy_event = None
            if result.heavy_refresh_requested:
                refresh_request_count += 1
            if result.mode == "LOST":
                lost_count += 1
            if result.obstacle_persistent:
                persistent_obstacle_count += 1
            if execute_heavy_refresh and result.heavy_refresh_requested:
                event_capture = output_dir / "heavy_inputs" / ("frame_%03d" % int(frame["index"]))
                prepare_event_capture(
                    event_capture,
                    frame["rgb"],
                    depth,
                    tracked_mask,
                    result.tracking_confidence,
                )
                try:
                    heavy_event = run_heavy_refresh(
                        event_capture,
                        output_dir / "heavy_outputs",
                        device=heavy_device,
                    )
                except Exception as exc:
                    heavy_event = {"status": "failed", "error": str(exc)}
                heavy_event.update(
                    {
                        "source_frame_index": int(frame["index"]),
                        "trigger": result.heavy_refresh_reason,
                    }
                )
                heavy_events.append(heavy_event)
                heavy_mask = load_target_mask(heavy_event)
                if heavy_mask is not None:
                    refreshed = tracker.apply_heavy_refresh(rgb, heavy_mask, depth)
                    refreshed.heavy_refresh_requested = True
                    refreshed.heavy_refresh_reason = result.heavy_refresh_reason
                    result = refreshed
                    tracked_mask = tracker.mask.copy()
                    refresh_applied = True
                    refresh_applied_count += 1
            elif simulate_saved_mask_refresh and result.heavy_refresh_requested and saved_mask is not None:
                try:
                    refreshed = tracker.apply_heavy_refresh(rgb, saved_mask, depth)
                except ValueError:
                    refreshed = None
                if refreshed is not None:
                    refreshed.heavy_refresh_requested = True
                    refreshed.heavy_refresh_reason = result.heavy_refresh_reason
                    result = refreshed
                    tracked_mask = tracker.mask.copy()
                    refresh_applied = True
                    refresh_applied_count += 1

        mask_path = output_dir / ("frame_%03d_mask.png" % int(frame["index"]))
        overlay_path = output_dir / ("frame_%03d_overlay.png" % int(frame["index"]))
        cv2.imwrite(str(mask_path), tracked_mask.astype(np.uint8) * 255)
        record = result.to_dict()
        record.update(
            {
                "source_frame_index": int(frame["index"]),
                "source_rgb": str(frame["rgb"]),
                "source_depth": str(frame["depth"]) if frame["depth"].is_file() else "",
                "source_saved_mask": str(frame["mask"]) if frame["mask"].is_file() else "",
                "refresh_applied": refresh_applied,
                "heavy_event": heavy_event,
                "saved_mask_iou": mask_iou(tracked_mask, saved_mask),
                "tracked_mask_png": str(mask_path),
                "overlay_png": str(overlay_path),
                "dry_run": True,
                "real_arm_motion": False,
            }
        )
        write_overlay(rgb, tracked_mask, record, overlay_path)
        records.append(record)

    mode_counts: dict[str, int] = {}
    for record in records:
        mode = str(record["mode"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    saved_mask_ious = [
        float(record["saved_mask_iou"])
        for record in records
        if record.get("saved_mask_iou") is not None
    ]
    payload = {
        "results_type": "offline_temporal_target_tracking",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scan_dir": str(scan_dir),
        "output_dir": str(output_dir),
        "frame_count": len(records),
        "initial_mask_path": str(initial_mask_path) if initial_mask_path else str(frames[0]["mask"]),
        "simulate_saved_mask_refresh": bool(simulate_saved_mask_refresh),
        "execute_heavy_refresh": bool(execute_heavy_refresh),
        "heavy_device": heavy_device,
        "policy": {
            "heavy_models_continuous": False,
            "heavy_refresh_is_request_only": not (simulate_saved_mask_refresh or execute_heavy_refresh),
            "dry_run": True,
            "real_arm_motion": False,
        },
        "config": vars(config),
        "aggregate": {
            "mode_counts": mode_counts,
            "refresh_request_count": refresh_request_count,
            "refresh_applied_count": refresh_applied_count,
            "lost_frame_count": lost_count,
            "persistent_obstacle_frame_count": persistent_obstacle_count,
            "heavy_event_count": len(heavy_events),
            "saved_mask_iou_frame_count": len(saved_mask_ious),
            "mean_saved_mask_iou": float(np.mean(saved_mask_ious)) if saved_mask_ious else None,
            "min_saved_mask_iou": float(np.min(saved_mask_ious)) if saved_mask_ious else None,
        },
        "frames": records,
        "heavy_events": heavy_events,
    }
    write_yaml(output_dir / "temporal_tracking_results.yaml", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track an initialized target mask through a saved RGB-D scan.")
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initial-mask", type=Path, default=None)
    parser.add_argument("--simulate-saved-mask-refresh", action="store_true")
    parser.add_argument("--execute-heavy-refresh", action="store_true")
    parser.add_argument("--heavy-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.50)
    parser.add_argument("--low-tracking-confidence", type=float, default=0.25)
    parser.add_argument("--max-missed-frames", type=int, default=5)
    parser.add_argument("--refresh-interval-frames", type=int, default=90)
    parser.add_argument("--scene-change-threshold", type=float, default=45.0)
    parser.add_argument("--obstacle-persistence-frames", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_dir = args.scan_dir.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / scan_dir.name
    config = TemporalTrackerConfig(
        min_tracking_confidence=float(args.min_tracking_confidence),
        low_tracking_confidence=float(args.low_tracking_confidence),
        max_missed_frames=max(0, int(args.max_missed_frames)),
        refresh_interval_frames=max(1, int(args.refresh_interval_frames)),
        scene_change_threshold=float(args.scene_change_threshold),
        obstacle_persistence_frames=max(1, int(args.obstacle_persistence_frames)),
    )
    payload = run_sequence(
        scan_dir=scan_dir,
        output_dir=output_dir.expanduser().resolve(),
        config=config,
        initial_mask_path=args.initial_mask.expanduser().resolve() if args.initial_mask else None,
        simulate_saved_mask_refresh=bool(args.simulate_saved_mask_refresh),
        execute_heavy_refresh=bool(args.execute_heavy_refresh),
        heavy_device=args.heavy_device,
    )
    print("Results written:", Path(payload["output_dir"]) / "temporal_tracking_results.yaml")
    print("Frames:", payload["frame_count"])
    print("Aggregate:", payload["aggregate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
