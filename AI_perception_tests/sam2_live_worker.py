#!/usr/bin/env python3
"""ROS-free CUDA SAM2 streaming worker using a filesystem frame bridge."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR / "groundingdino_test"
SAM2_REPO = AI_DIR / "Grounded-SAM-2"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT = AI_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"


def atomic_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


class Sam2LiveWorker:
    def __init__(
        self,
        spool_dir: Path,
        device: str = "cuda",
        config: str = DEFAULT_CONFIG,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        max_session_frames: int = 8,
    ) -> None:
        self.spool = Path(spool_dir)
        self.frames = self.spool / "frames"
        self.seeds = self.spool / "seeds"
        self.results = self.spool / "results"
        self.consumed = self.spool / "consumed"
        for directory in (self.frames, self.seeds, self.results, self.consumed):
            directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.max_session_frames = max(2, int(max_session_frames))
        self.predictor = None
        self.state = None
        self.last_masks = {}
        self.objects = []
        self.last_frame_key = ""
        self.session_frames = 0
        self.tracking_state = "WAITING_FOR_SEED"

    @staticmethod
    def prune_directories(root: Path, keep: int) -> None:
        directories = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        for path in directories[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    def build(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if not self.checkpoint.is_file():
            raise FileNotFoundError("SAM2 checkpoint not found: %s" % self.checkpoint)
        if str(SAM2_REPO) not in sys.path:
            sys.path.insert(0, str(SAM2_REPO))
        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(
            self.config,
            str(self.checkpoint),
            device=self.device,
            apply_postprocessing=True,
        )

    @staticmethod
    def ready_directories(root: Path) -> list[Path]:
        return sorted(
            path for path in root.iterdir()
            if path.is_dir() and not path.name.endswith(".tmp") and (path / "READY").is_file()
        )

    def reset(self, rgb_bgr: np.ndarray, masks: dict[int, np.ndarray], objects: list[dict], frame_key: str) -> None:
        if self.predictor is None:
            self.build()
        valid_masks = {
            int(object_id): np.asarray(mask) > 0
            for object_id, mask in masks.items()
            if np.asarray(mask).shape == rgb_bgr.shape[:2] and np.count_nonzero(mask)
        }
        if not valid_masks or 1 not in valid_masks:
            raise ValueError("seed must contain a non-empty target mask with object_id=1")
        if self.state is not None:
            self.state = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.state = self.predictor.init_state(video_path=None, offload_state_to_cpu=False)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"
        ):
            frame_index = self.predictor.add_new_frame(self.state, rgb)
            self.state["video_height"], self.state["video_width"] = rgb_bgr.shape[:2]
            object_ids = []
            logits = None
            for object_id, mask in sorted(valid_masks.items()):
                _, object_ids, logits = self.predictor.add_new_mask(
                    self.state, frame_idx=frame_index, obj_id=object_id, mask=mask
                )
        self.last_masks = self.masks_from_logits(object_ids, logits)
        self.objects = [
            dict(record)
            for record in objects
            if int(record.get("object_id", 0)) in self.last_masks
        ]
        self.last_frame_key = frame_key
        self.session_frames = 1
        self.tracking_state = "TRACKING"

    @staticmethod
    def masks_from_logits(object_ids, logits) -> dict[int, np.ndarray]:
        return {
            int(object_id): (logits[index] > 0.0).detach().cpu().numpy().squeeze()
            for index, object_id in enumerate(object_ids)
        }

    def consume_latest_seed(self) -> bool:
        seeds = self.ready_directories(self.seeds)
        if not seeds:
            return False
        seed = seeds[-1]
        rgb = cv2.imread(str(seed / "rgb.jpg"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError("unreadable seed RGB in %s" % seed)
        manifest_path = seed / "seed.yaml"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream) or {}
        objects = manifest.get("objects", [])
        if not objects:
            legacy_mask = cv2.imread(str(seed / "mask.png"), cv2.IMREAD_GRAYSCALE)
            if legacy_mask is None:
                raise ValueError("seed contains no object masks: %s" % seed)
            objects = [{"object_id": 1, "role": "target", "label": "target", "mask_file": "mask.png"}]
        masks = {}
        for record in objects:
            mask = cv2.imread(str(seed / str(record.get("mask_file", ""))), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[int(record.get("object_id", 0))] = mask
        self.reset(rgb, masks, objects, seed.name)
        # Frames captured before the new semantic seed are no longer useful and
        # must not be replayed into the recovered tracking session.
        for stale_frame in self.ready_directories(self.frames):
            if stale_frame.name <= self.last_frame_key:
                shutil.rmtree(stale_frame, ignore_errors=True)
        for old_seed in seeds:
            destination = self.consumed / ("seed_" + old_seed.name)
            shutil.rmtree(destination, ignore_errors=True)
            os.replace(str(old_seed), str(destination))
        print("SAM2 live seed accepted: %s" % seed.name, flush=True)
        return True

    def process_frame(self, frame: Path) -> None:
        rgb = cv2.imread(str(frame / "rgb.jpg"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError("unreadable frame: %s" % frame)
        with (frame / "frame.yaml").open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream) or {}
        started = time.perf_counter()
        target_was_valid = bool(
            1 in self.last_masks and np.count_nonzero(self.last_masks[1])
        )
        if not target_was_valid:
            # Never try to initialize a new SAM2 session from an empty target.
            # Consume this frame and wait for a semantic seed instead of retrying
            # the same failing reset forever.
            predictions = dict(self.last_masks)
        elif self.session_frames >= self.max_session_frames:
            self.reset(rgb, self.last_masks, self.objects, frame.name)
            predictions = self.last_masks
        else:
            rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"
            ):
                frame_index = self.predictor.add_new_frame(self.state, rgb_input)
                _, object_ids, logits = self.predictor.infer_single_frame(self.state, frame_index)
            predictions = self.masks_from_logits(object_ids, logits)
            self.last_masks = predictions
            self.last_frame_key = frame.name
            self.session_frames += 1
        elapsed = time.perf_counter() - started
        result_tmp = self.results / (frame.name + ".tmp")
        result = self.results / frame.name
        shutil.rmtree(result_tmp, ignore_errors=True)
        result_tmp.mkdir(parents=True)
        object_dir = result_tmp / "objects"
        object_dir.mkdir()
        target_mask = predictions.get(1, np.zeros(rgb.shape[:2], dtype=bool))
        cv2.imwrite(str(result_tmp / "mask.png"), target_mask.astype(np.uint8) * 255)
        all_obstacles = np.zeros(rgb.shape[:2], dtype=bool)
        unsafe_obstacles = np.zeros(rgb.shape[:2], dtype=bool)
        movable_obstacles = np.zeros(rgb.shape[:2], dtype=bool)
        object_ids_image = np.zeros(rgb.shape[:2], dtype=np.uint16)
        result_objects = []
        object_metadata = {int(item.get("object_id", 0)): item for item in self.objects}
        for object_id, prediction in sorted(predictions.items()):
            record = dict(object_metadata.get(object_id, {"object_id": object_id, "role": "obstacle"}))
            mask_file = "objects/object_%03d.png" % object_id
            cv2.imwrite(str(result_tmp / mask_file), prediction.astype(np.uint8) * 255)
            record["mask_file"] = mask_file
            record["mask_area_px"] = int(np.count_nonzero(prediction))
            result_objects.append(record)
            object_ids_image[prediction] = np.uint16(object_id)
            if record.get("role") == "obstacle":
                all_obstacles |= prediction
                if bool(record.get("unsafe", True)):
                    unsafe_obstacles |= prediction
                if bool(record.get("candidate_movable", False)):
                    movable_obstacles |= prediction
        cv2.imwrite(str(result_tmp / "all_obstacle_mask.png"), all_obstacles.astype(np.uint8) * 255)
        cv2.imwrite(str(result_tmp / "unsafe_obstacle_mask.png"), unsafe_obstacles.astype(np.uint8) * 255)
        cv2.imwrite(str(result_tmp / "candidate_movable_obstacle_mask.png"), movable_obstacles.astype(np.uint8) * 255)
        cv2.imwrite(str(result_tmp / "object_ids.png"), object_ids_image)
        payload = {
            "status": "ok" if np.count_nonzero(target_mask) else "empty_target_mask",
            "frame_key": frame.name,
            "image_stamp": metadata.get("image_stamp", {}),
            "frame_id": metadata.get("frame_id", ""),
            "mask_area_px": int(np.count_nonzero(target_mask)),
            "object_count": len(result_objects),
            "objects": result_objects,
            "inference_sec": float(elapsed),
            "inference_fps": float(1.0 / elapsed) if elapsed else 0.0,
            "device": self.device,
            "session_frames": self.session_frames,
            "dry_run": True,
            "real_arm_motion": False,
        }
        if np.count_nonzero(target_mask):
            self.tracking_state = "TRACKING"
        else:
            self.tracking_state = "WAITING_FOR_SEED"
            self.state = None
            self.last_masks = {}
            self.session_frames = 0
        payload["tracking_state"] = self.tracking_state
        atomic_yaml(result_tmp / "result.yaml", payload)
        (result_tmp / "READY").touch()
        os.replace(str(result_tmp), str(result))
        destination = self.consumed / ("frame_" + frame.name)
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(str(frame), str(destination))
        self.prune_directories(self.consumed, keep=200)

    def process_once(self) -> bool:
        seeded = self.consume_latest_seed()
        if self.state is None:
            return seeded
        frames = self.ready_directories(self.frames)
        frames = [frame for frame in frames if frame.name > self.last_frame_key]
        if not frames:
            return seeded
        # Real-time operation favors the newest frame over stale queued frames.
        for stale in frames[:-1]:
            shutil.rmtree(stale, ignore_errors=True)
        self.process_frame(frames[-1])
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool-dir", type=Path, default=Path("/tmp/piper_sam2_live"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-session-frames", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = Sam2LiveWorker(
        args.spool_dir, args.device, args.config, args.checkpoint, args.max_session_frames
    )
    worker.build()
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print("SAM2 live worker ready on %s" % device_name, flush=True)
    while running:
        try:
            if not worker.process_once():
                time.sleep(max(0.005, args.poll_interval))
        except Exception as exc:
            print("SAM2 live worker error: %s: %s" % (type(exc).__name__, exc), flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
