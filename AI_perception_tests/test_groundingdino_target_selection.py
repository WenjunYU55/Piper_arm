#!/usr/bin/env python3
"""Regression checks for the deliberately strict target vocabulary."""

import sys
import unittest
from pathlib import Path


GROUNDING_DIR = Path(__file__).parent / 'groundingdino_test'
sys.path.insert(0, str(GROUNDING_DIR))

from run_groundingdino_on_capture import (  # noqa: E402
    TARGET_TERMS,
    label_matches,
    target_crop_bounds,
)


class TargetSelectionTest(unittest.TestCase):
    def test_green_cube_is_target(self):
        self.assertTrue(label_matches('green cube', TARGET_TERMS))

    def test_generic_cube_is_not_target(self):
        self.assertFalse(label_matches('cube', TARGET_TERMS))

    def test_box_and_cardboard_are_not_targets(self):
        self.assertFalse(label_matches('box', TARGET_TERMS))
        self.assertFalse(label_matches('cardboard box', TARGET_TERMS))

    def test_small_target_uses_256_pixel_obstacle_crop(self):
        bounds = target_crop_bounds([300, 220, 340, 260], 640, 480)
        self.assertEqual(bounds, (192, 112, 448, 368))

    def test_obstacle_crop_is_clipped_to_image(self):
        bounds = target_crop_bounds([0, 0, 40, 40], 640, 480)
        self.assertEqual(bounds, (0, 0, 148, 148))


if __name__ == '__main__':
    unittest.main()
