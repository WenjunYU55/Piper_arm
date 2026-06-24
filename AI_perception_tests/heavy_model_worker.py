#!/usr/bin/env python3
"""ROS-free filesystem worker for GroundingDINO/SAM2 refresh requests."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml


InferenceFunction = Callable[[Path, Path, str], dict]


def atomic_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def default_inference(capture_dir: Path, output_dir: Path, device: str) -> dict:
    # Keep heavy imports out of startup, tests, and every ROS/Foxy process.
    from temporal_heavy_refresh import run_heavy_refresh

    return run_heavy_refresh(capture_dir, output_dir, device=device)


def prepare_capture_metadata(
    capture_dir: Path,
    depth: np.ndarray | None,
    tracked_mask: np.ndarray,
    tracking_confidence: float,
) -> None:
    mask = tracked_mask > 0
    ys, xs = np.nonzero(mask)
    width = float(xs.max() - xs.min() + 1) if xs.size else 0.0
    height = float(ys.max() - ys.min() + 1) if ys.size else 0.0
    depth_m = None
    valid_ratio = 0.0
    if depth is not None and depth.shape[:2] == mask.shape and xs.size:
        values_m = depth.astype(np.float32, copy=False)
        if np.issubdtype(depth.dtype, np.integer):
            values_m = values_m * 0.001
        valid = mask & np.isfinite(values_m) & (values_m > 0.0)
        values = values_m[valid]
        valid_ratio = float(values.size / max(1, int(np.count_nonzero(mask))))
        if values.size:
            depth_m = float(np.median(values))
    target = {
        "valid": bool(xs.size),
        "source_u": float(np.mean(xs)) if xs.size else -1.0,
        "source_v": float(np.mean(ys)) if ys.size else -1.0,
        "detection_width": width,
        "detection_height": height,
        "roi_width": width,
        "roi_height": height,
        "measurement_confidence": float(tracking_confidence),
        "valid_depth_ratio": valid_ratio,
        "depth": depth_m,
        "point": {"x": 0.0, "y": 0.0, "z": depth_m or 0.0},
        "depth_source": "temporal_tracked_mask",
    }
    atomic_yaml(capture_dir / "target_3d.yaml", target)
    atomic_yaml(
        capture_dir / "metadata.yaml",
        {"source": "live_heavy_refresh_worker", "dry_run": True, "real_arm_motion": False},
    )


class HeavyModelWorker:
    def __init__(
        self,
        spool_dir: Path,
        device: str = "cpu",
        inference: InferenceFunction = default_inference,
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.device = device
        self.inference = inference
        self.requests = self.spool_dir / "requests"
        self.processing = self.spool_dir / "processing"
        self.responses = self.spool_dir / "responses"
        self.archive = self.spool_dir / "archive"
        self.failed = self.spool_dir / "failed"
        self.model_outputs = self.spool_dir / "model_outputs"
        for directory in (
            self.requests,
            self.processing,
            self.responses,
            self.archive,
            self.failed,
            self.model_outputs,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def recover_interrupted_jobs(self) -> None:
        for job in sorted(self.processing.iterdir()):
            if not job.is_dir():
                continue
            destination = self.requests / job.name
            if not destination.exists():
                os.replace(str(job), str(destination))

    def pending_jobs(self) -> list[Path]:
        jobs = []
        for path in self.requests.iterdir():
            if path.name.endswith(".tmp") or not path.is_dir() or not (path / "READY").is_file():
                continue
            try:
                jobs.append((path.stat().st_mtime, path))
            except FileNotFoundError:
                # The bridge atomically renamed the directory between discovery and stat.
                continue
        return [path for _mtime, path in sorted(jobs, key=lambda item: item[0])]

    def process_one(self) -> bool:
        jobs = self.pending_jobs()
        if not jobs:
            return False
        queued = jobs[0]
        job = self.processing / queued.name
        try:
            os.replace(str(queued), str(job))
        except FileNotFoundError:
            return False

        response_tmp = self.responses / (job.name + ".tmp")
        response = self.responses / job.name
        shutil.rmtree(response_tmp, ignore_errors=True)
        response_tmp.mkdir(parents=True)
        request = {}
        try:
            with (job / "request.yaml").open("r", encoding="utf-8") as stream:
                request = yaml.safe_load(stream) or {}
            rgb = cv2.imread(str(job / "rgb.png"), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError("rgb.png is missing or unreadable")
            tracked = cv2.imread(str(job / "tracked_mask.png"), cv2.IMREAD_GRAYSCALE)
            if tracked is None:
                tracked = np.zeros(rgb.shape[:2], dtype=np.uint8)
            depth_path = job / "depth.npy"
            depth = np.load(str(depth_path), allow_pickle=False) if depth_path.is_file() else None

            capture_dir = self.model_outputs / job.name / "capture"
            capture_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(capture_dir / "rgb.png"), rgb)
            cv2.imwrite(str(capture_dir / "detection_mask.png"), tracked)
            if depth is not None:
                np.save(str(capture_dir / "depth.npy"), depth)

            prepare_capture_metadata(
                capture_dir,
                depth,
                tracked,
                float(request.get("tracking_confidence", 0.0)),
            )
            result = self.inference(capture_dir, self.model_outputs / job.name, self.device)
            mask_path = Path(str(result.get("target_mask_png", "")))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
            status = (
                "ok"
                if mask is not None and mask.shape == rgb.shape[:2] and np.count_nonzero(mask)
                else "target_mask_missing"
            )
            if mask is not None:
                cv2.imwrite(str(response_tmp / "target_mask.png"), mask)
            movable_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            unsafe_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            all_obstacle_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for obstacle in result.get("obstacle_masks", []):
                if not isinstance(obstacle, dict):
                    continue
                obstacle_path = Path(str(obstacle.get("mask_png", "")))
                obstacle_mask = (
                    cv2.imread(str(obstacle_path), cv2.IMREAD_GRAYSCALE)
                    if obstacle_path.is_file()
                    else None
                )
                if obstacle_mask is None or obstacle_mask.shape != rgb.shape[:2]:
                    continue
                obstacle_binary = obstacle_mask > 0
                all_obstacle_mask[obstacle_binary] = 255
                if obstacle.get("candidate_movable", False):
                    movable_mask[obstacle_binary] = 255
                if obstacle.get("unsafe", True):
                    unsafe_mask[obstacle_binary] = 255
            cv2.imwrite(str(response_tmp / "candidate_movable_obstacle_mask.png"), movable_mask)
            cv2.imwrite(str(response_tmp / "unsafe_obstacle_mask.png"), unsafe_mask)
            cv2.imwrite(str(response_tmp / "all_obstacle_mask.png"), all_obstacle_mask)
            payload = dict(result)
            payload.update(
                {
                    "status": status,
                    "job_id": job.name,
                    "request_id": request.get("request_id"),
                    "image_stamp": request.get("image_stamp", {}),
                    "frame_id": request.get("frame_id", ""),
                    "dry_run": True,
                    "real_arm_motion": False,
                }
            )
            atomic_yaml(response_tmp / "result.yaml", payload)
            (response_tmp / "READY").touch()
            os.replace(str(response_tmp), str(response))
            os.replace(str(job), str(self.archive / job.name))
        except Exception as exc:
            failure = {
                "status": "worker_error",
                "job_id": job.name,
                "request_id": request.get("request_id"),
                "image_stamp": request.get("image_stamp", {}),
                "error": "%s: %s" % (type(exc).__name__, exc),
                "dry_run": True,
                "real_arm_motion": False,
            }
            atomic_yaml(
                job / "failure.yaml",
                failure,
            )
            atomic_yaml(response_tmp / "result.yaml", failure)
            (response_tmp / "READY").touch()
            if not response.exists():
                os.replace(str(response_tmp), str(response))
            destination = self.failed / job.name
            shutil.rmtree(destination, ignore_errors=True)
            os.replace(str(job), str(destination))
            shutil.rmtree(response_tmp, ignore_errors=True)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spool-dir",
        type=Path,
        default=Path("/tmp/piper_heavy_refresh"),
        help="Filesystem boundary shared with the ROS bridge",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = HeavyModelWorker(args.spool_dir, device=args.device)
    worker.recover_interrupted_jobs()
    if args.once:
        worker.process_one()
        return
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("Heavy-model worker ready: %s (device=%s)" % (args.spool_dir, args.device), flush=True)
    while running:
        if not worker.process_one():
            time.sleep(max(0.05, args.poll_interval))


if __name__ == "__main__":
    main()
