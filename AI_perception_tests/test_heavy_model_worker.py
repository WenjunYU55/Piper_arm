#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from heavy_model_worker import HeavyModelWorker


class HeavyModelWorkerTest(unittest.TestCase):
    def test_incomplete_temporary_job_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            temporary_job = spool / "requests" / "job.tmp"
            temporary_job.mkdir(parents=True)
            worker = HeavyModelWorker(spool)
            self.assertEqual(worker.pending_jobs(), [])
            self.assertFalse(worker.process_one())

    def test_job_is_claimed_and_response_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            job = spool / "requests" / "job_1"
            job.mkdir(parents=True)
            rgb = np.zeros((30, 40, 3), dtype=np.uint8)
            rgb[8:22, 12:28] = (0, 255, 0)
            cv2.imwrite(str(job / "rgb.png"), rgb)
            cv2.imwrite(str(job / "tracked_mask.png"), np.zeros(rgb.shape[:2], np.uint8))
            with (job / "request.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump({"request_id": 7, "image_stamp": {"sec": 3, "nanosec": 4}}, stream)
            (job / "READY").touch()

            def fake_inference(capture_dir: Path, output_dir: Path, device: str) -> dict:
                self.assertEqual(device, "cpu")
                mask = np.zeros((30, 40), dtype=np.uint8)
                mask[8:22, 12:28] = 255
                output = output_dir / "fake_target.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output), mask)
                movable = np.zeros((30, 40), dtype=np.uint8)
                movable[5:10, 30:36] = 255
                movable_path = output_dir / "fake_movable.png"
                cv2.imwrite(str(movable_path), movable)
                unsafe = np.zeros((30, 40), dtype=np.uint8)
                unsafe[20:28, 2:8] = 255
                unsafe_path = output_dir / "fake_unsafe.png"
                cv2.imwrite(str(unsafe_path), unsafe)
                return {
                    "status": "ok",
                    "target_mask_png": str(output),
                    "target_confidence": 0.9,
                    "obstacle_masks": [
                        {
                            "label": "marker",
                            "mask_png": str(movable_path),
                            "candidate_movable": True,
                            "unsafe": False,
                        },
                        {"label": "hand", "mask_png": str(unsafe_path), "unsafe": True},
                    ],
                }

            worker = HeavyModelWorker(spool, inference=fake_inference)
            self.assertTrue(worker.process_one())
            response = spool / "responses" / "job_1"
            self.assertTrue((response / "READY").is_file())
            self.assertTrue((response / "target_mask.png").is_file())
            self.assertEqual(
                int(np.count_nonzero(cv2.imread(str(response / "candidate_movable_obstacle_mask.png"), 0))),
                30,
            )
            self.assertEqual(
                int(np.count_nonzero(cv2.imread(str(response / "unsafe_obstacle_mask.png"), 0))),
                48,
            )
            self.assertTrue((spool / "archive" / "job_1").is_dir())
            self.assertFalse(any((spool / "responses").glob("*.tmp")))
            with (response / "result.yaml").open("r", encoding="utf-8") as stream:
                result = yaml.safe_load(stream)
            self.assertEqual(result["request_id"], 7)
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["real_arm_motion"])

    def test_failure_produces_a_consumable_status_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            job = spool / "requests" / "job_failure"
            job.mkdir(parents=True)
            cv2.imwrite(str(job / "rgb.png"), np.zeros((10, 10, 3), dtype=np.uint8))
            with (job / "request.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump({"request_id": 8}, stream)
            (job / "READY").touch()

            def failing_inference(_capture_dir: Path, _output_dir: Path, _device: str) -> dict:
                raise RuntimeError("inference failed")

            worker = HeavyModelWorker(spool, inference=failing_inference)
            self.assertTrue(worker.process_one())
            response = spool / "responses" / "job_failure"
            with (response / "result.yaml").open("r", encoding="utf-8") as stream:
                result = yaml.safe_load(stream)
            self.assertEqual(result["status"], "worker_error")
            self.assertIn("inference failed", result["error"])
            self.assertTrue((spool / "failed" / "job_failure").is_dir())

    def test_target_missing_keeps_rgb_and_reserves_id_one_for_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            job = spool / "requests" / "job_obstacle_only"
            job.mkdir(parents=True)
            rgb = np.zeros((20, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(job / "rgb.png"), rgb)
            with (job / "request.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump({
                    "request_id": 9,
                    "image_stamp": {"sec": 5, "nanosec": 6},
                }, stream)
            (job / "READY").touch()

            def obstacle_only(_capture_dir, output_dir, _device):
                obstacle = np.zeros((20, 30), dtype=np.uint8)
                obstacle[4:12, 15:24] = 255
                path = output_dir / "obstacle.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), obstacle)
                return {
                    "target_mask_png": "",
                    "obstacle_masks": [{
                        "label": "hand",
                        "mask_png": str(path),
                        "unsafe": True,
                    }],
                }

            worker = HeavyModelWorker(spool, inference=obstacle_only)
            self.assertTrue(worker.process_one())
            response = spool / "responses" / "job_obstacle_only"
            with (response / "result.yaml").open("r", encoding="utf-8") as stream:
                result = yaml.safe_load(stream)
            self.assertEqual(result["status"], "target_mask_missing")
            self.assertTrue((response / "rgb.jpg").is_file())
            self.assertEqual(result["obstacle_count"], 1)
            self.assertEqual(result["unsafe_obstacle_count"], 1)
            self.assertEqual(result["tracked_objects"][0]["object_id"], 2)


if __name__ == "__main__":
    unittest.main()
