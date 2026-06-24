#!/usr/bin/env python3

from __future__ import annotations

import unittest

import cv2
import numpy as np

from temporal_tracking import TemporalMaskTracker, TemporalTrackerConfig


def synthetic_frame(offset_x: int = 0, offset_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    x0, y0, x1, y1 = 45 + offset_x, 35 + offset_y, 85 + offset_x, 75 + offset_y
    mask[y0:y1, x0:x1] = 255
    for y in range(y0, y1, 8):
        for x in range(x0, x1, 8):
            color = 220 if ((x + y) // 8) % 2 else 70
            cv2.rectangle(image, (x, y), (min(x + 5, x1 - 1), min(y + 5, y1 - 1)), (color, color, color), -1)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
    return image, mask


def green_cube_frame(offset_x: int = 0, offset_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = 45 + offset_x, 35 + offset_y, 85 + offset_x, 75 + offset_y
    image[y0:y1, x0:x1] = (30, 210, 60)
    mask[y0:y1, x0:x1] = 255
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (20, 120, 30), 2)
    return image, mask


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first > 0
    second_bool = second > 0
    intersection = int(np.count_nonzero(first_bool & second_bool))
    union = int(np.count_nonzero(first_bool | second_bool))
    return intersection / union if union else 0.0


class TemporalMaskTrackerTest(unittest.TestCase):
    def test_color_correction_snaps_prediction_to_green_cube(self):
        config = TemporalTrackerConfig(
            enable_color_correction=True,
            color_search_margin_px=80,
            color_max_centroid_shift_px=100.0,
        )
        tracker = TemporalMaskTracker(config)
        first_image, first_mask = green_cube_frame()
        second_image, expected_mask = green_cube_frame(55, 7)
        # Add a green distractor far outside the target-local search region.
        second_image[20:55, 155:190] = (30, 210, 60)
        depth = np.full(first_mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(first_image, first_mask, depth)

        result, corrected = tracker.step(second_image, depth)

        self.assertTrue(result.color_correction_used)
        self.assertGreater(mask_iou(corrected, expected_mask), 0.80)
        self.assertEqual(result.mode, "TRACKING")

    def test_adaptive_appearance_is_learned_from_seed(self):
        config = TemporalTrackerConfig(
            enable_color_correction=True,
            enable_adaptive_appearance=True,
            use_hsv_fallback=False,
        )
        tracker = TemporalMaskTracker(config)
        first_image, first_mask = green_cube_frame()
        second_image, expected_mask = green_cube_frame(12, 3)
        # Moderate illumination/color shift while preserving the target's learned chroma.
        target = second_image[38:78, 57:97].astype(np.int16)
        second_image[38:78, 57:97] = np.clip(target + (12, -18, 5), 0, 255).astype(np.uint8)
        depth = np.full(first_mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(first_image, first_mask, depth)

        result, corrected = tracker.step(second_image, depth)

        self.assertIsNotNone(tracker.appearance_center)
        self.assertTrue(result.appearance_correction_used)
        self.assertGreater(mask_iou(corrected, expected_mask), 0.75)

    def test_missing_green_target_does_not_follow_background(self):
        config = TemporalTrackerConfig(enable_color_correction=True, require_color_correction=True)
        tracker = TemporalMaskTracker(config)
        image, mask = green_cube_frame()
        depth = np.full(mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(image, mask, depth)

        result, propagated = tracker.step(np.zeros_like(image), depth)

        self.assertFalse(result.color_correction_used)
        self.assertEqual(result.mode, "REFRESH_REQUIRED")
        self.assertTrue(result.heavy_refresh_requested)
        self.assertTrue(np.array_equal(propagated, mask > 0))

    def test_translated_mask_is_propagated(self):
        config = TemporalTrackerConfig(min_tracking_confidence=0.45)
        tracker = TemporalMaskTracker(config)
        first_image, first_mask = synthetic_frame()
        second_image, expected_mask = synthetic_frame(6, 4)
        depth = np.full(first_mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(first_image, first_mask, depth)

        result, propagated = tracker.step(second_image, depth)

        self.assertEqual(result.mode, "TRACKING")
        self.assertGreaterEqual(result.tracking_confidence, config.min_tracking_confidence)
        self.assertGreater(mask_iou(propagated, expected_mask), 0.85)
        self.assertFalse(result.heavy_refresh_requested)

    def test_repeated_failure_expires_track(self):
        config = TemporalTrackerConfig(max_missed_frames=2, scene_change_threshold=255.0)
        tracker = TemporalMaskTracker(config)
        image, mask = synthetic_frame()
        depth = np.full(mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(image, mask, depth)
        blank = np.zeros_like(image)

        modes = [tracker.step(blank, depth)[0].mode for _ in range(3)]

        self.assertEqual(modes[-1], "LOST")
        self.assertTrue(all(mode in ("REFRESH_REQUIRED", "LOST") for mode in modes))

    def test_periodic_refresh_is_requested(self):
        config = TemporalTrackerConfig(refresh_interval_frames=2, scene_change_threshold=255.0)
        tracker = TemporalMaskTracker(config)
        image, mask = synthetic_frame()
        depth = np.full(mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(image, mask, depth)

        first = tracker.step(image, depth)[0]
        second = tracker.step(image, depth)[0]

        self.assertFalse(first.heavy_refresh_requested)
        self.assertTrue(second.heavy_refresh_requested)
        self.assertIn("periodic_refresh", second.heavy_refresh_reason)
        third = tracker.step(image, depth)[0]
        self.assertFalse(third.heavy_refresh_requested)
        self.assertEqual(third.heavy_refresh_reason, "")

    def test_depth_obstacle_requires_persistence(self):
        config = TemporalTrackerConfig(
            obstacle_persistence_frames=3,
            scene_change_threshold=255.0,
            min_depth_obstacle_support_px=20,
        )
        tracker = TemporalMaskTracker(config)
        image, mask = synthetic_frame()
        depth = np.full(mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(image, mask, depth)
        blocked_depth = depth.copy()
        blocked_depth[45:65, 86:92] = 850

        results = [tracker.step(image, blocked_depth)[0] for _ in range(3)]

        self.assertTrue(all(result.obstacle_candidate for result in results))
        self.assertFalse(results[1].obstacle_persistent)
        self.assertTrue(results[2].obstacle_persistent)
        self.assertTrue(results[2].heavy_refresh_requested)
        self.assertIn("persistent_obstacle", results[2].heavy_refresh_reason)
        fourth = tracker.step(image, blocked_depth)[0]
        self.assertFalse(fourth.heavy_refresh_requested)

    def test_direct_target_occlusion_is_counted_as_obstacle(self):
        config = TemporalTrackerConfig(
            obstacle_persistence_frames=2,
            scene_change_threshold=255.0,
            min_depth_obstacle_support_px=20,
        )
        tracker = TemporalMaskTracker(config)
        image, mask = synthetic_frame()
        depth = np.full(mask.shape, 1000, dtype=np.uint16)
        tracker.initialize(image, mask, depth)
        occluded_depth = depth.copy()
        occluded_depth[45:65, 55:75] = 850

        first = tracker.step(image, occluded_depth)[0]
        second = tracker.step(image, occluded_depth)[0]

        self.assertTrue(first.obstacle_candidate)
        self.assertGreaterEqual(first.obstacle_support_px, 20)
        self.assertTrue(second.obstacle_persistent)
        self.assertIn("persistent_obstacle", second.heavy_refresh_reason)


if __name__ == "__main__":
    unittest.main()
