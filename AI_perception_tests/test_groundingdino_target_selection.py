#!/usr/bin/env python3
"""Regression checks for the deliberately strict target vocabulary."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml


GROUNDING_DIR = Path(__file__).parent / 'groundingdino_test'
sys.path.insert(0, str(GROUNDING_DIR))

from run_groundingdino_on_capture import (  # noqa: E402
    CANDIDATE_SAFE_TERMS,
    DEFAULT_OBSTACLE_PROMPT,
    DEFAULT_TARGET_PROMPT,
    MIN_TARGET_GREEN_FRACTION,
    MIN_TARGET_SEMANTIC_CONFIDENCE,
    TARGET_TERMS,
    UNSAFE_TERMS,
    _GROUNDING_MODEL_CACHE,
    cached_groundingdino_model,
    label_matches,
    target_detection_validation,
    target_mask_appearance_validation,
    tracked_mask_target_fallback,
    target_crop_bounds,
    validate_target_detections,
)
from temporal_heavy_refresh import classify_refined_obstacles  # noqa: E402
from sam2_refine_on_capture import (  # noqa: E402
    _SAM2_MODEL_CACHE,
    cached_sam2_model,
    depth_occluder_mask,
    load_detection_mask,
    target_depth_from_refined_mask,
)


class TargetSelectionTest(unittest.TestCase):
    def test_sam2_model_is_loaded_once_per_process_device(self):
        calls = []
        _SAM2_MODEL_CACHE.clear()

        def build(config, checkpoint, device):
            calls.append((config, checkpoint, device))
            return object()

        checkpoint = Path('/tmp/model.pt')
        first = cached_sam2_model(build, 'config.yaml', checkpoint, 'cuda')
        second = cached_sam2_model(build, 'config.yaml', checkpoint, 'cuda')
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_missing_detection_mask_is_cleanly_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(load_detection_mask(Path(temporary)))

    def test_cube_face_depth_is_not_misclassified_as_an_occluder(self):
        target_mask = np.zeros((120, 160), dtype=bool)
        target_mask[30:80, 50:110] = True
        depth = np.full(target_mask.shape, 0.50, dtype=np.float32)
        depth[target_mask] = 0.39
        depth[30:65, 50:90] = 0.32
        # A small under-segmented strip belongs to the same near cube face.
        depth[25:30, 50:90] = 0.32

        selected_depth, report = target_depth_from_refined_mask(
            depth, target_mask, 0.37)
        result = depth_occluder_mask(
            Path('/unused'), depth, selected_depth, target_mask)

        self.assertEqual(report['status'], 'selected')
        self.assertAlmostEqual(selected_depth, 0.32, places=2)
        self.assertIsNone(result)

    def test_adjacent_support_surface_is_not_a_depth_occluder(self):
        target_mask = np.zeros((120, 160), dtype=bool)
        target_mask[30:80, 50:110] = True
        depth = np.full(target_mask.shape, 0.40, dtype=np.float32)
        depth[82:100, 40:120] = 0.32

        result = depth_occluder_mask(
            Path('/unused'), depth, 0.40, target_mask)

        self.assertIsNone(result)

    def test_exterior_depth_halo_inside_bbox_is_not_an_occluder(self):
        target_mask = np.zeros((120, 160), dtype=bool)
        target_mask[35:85, 55:105] = True
        depth = np.full(target_mask.shape, 0.50, dtype=np.float32)
        depth[target_mask] = 0.40
        # Reproduce a three-sided L515 discontinuity halo. Parts of this
        # component occupy the target bounding box's empty edge pixels, but it
        # never crosses the interior target silhouette.
        depth[29:35, 49:111] = 0.32
        depth[29:60, 49:55] = 0.32
        depth[29:60, 105:111] = 0.32

        result = depth_occluder_mask(
            Path('/unused'), depth, 0.40, target_mask)

        self.assertIsNone(result)

    def test_foreground_crossing_target_envelope_is_a_depth_occluder(self):
        target_mask = np.zeros((120, 160), dtype=bool)
        target_mask[30:80, 50:110] = True
        target_mask[50:70, 72:88] = False
        depth = np.full(target_mask.shape, 0.40, dtype=np.float32)
        depth[45:75, 72:88] = 0.32

        result = depth_occluder_mask(
            Path('/unused'), depth, 0.40, target_mask)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(int(np.count_nonzero(result)), 20)

    def test_grounding_model_is_loaded_once_per_worker_device(self):
        calls = []
        _GROUNDING_MODEL_CACHE.clear()

        def load_model(config, checkpoint, device):
            calls.append((config, checkpoint, device))
            return object()

        config = Path('/tmp/model-config.py')
        checkpoint = Path('/tmp/model-checkpoint.pth')
        first = cached_groundingdino_model(
            load_model, config, checkpoint, 'cuda')
        second = cached_groundingdino_model(
            load_model, config, checkpoint, 'cuda')
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_green_cube_is_target(self):
        self.assertTrue(label_matches('green cube', TARGET_TERMS))

    def test_generic_cube_is_not_target(self):
        self.assertFalse(label_matches('cube', TARGET_TERMS))

    def test_box_and_cardboard_are_not_targets(self):
        self.assertFalse(label_matches('box', TARGET_TERMS))
        self.assertFalse(label_matches('cardboard box', TARGET_TERMS))

    def test_target_prompt_does_not_mix_obstacle_classes(self):
        self.assertIn('green cube', DEFAULT_TARGET_PROMPT)
        for term in ('hand', 'wire', 'cardboard', 'unknown object'):
            self.assertNotIn(term, DEFAULT_TARGET_PROMPT)

    def test_only_human_terms_are_unsafe(self):
        self.assertTrue(label_matches('hand finger', UNSAFE_TERMS))
        self.assertFalse(label_matches('leaf branch', UNSAFE_TERMS))
        self.assertFalse(label_matches('cardboard box', UNSAFE_TERMS))
        self.assertTrue(label_matches('felt tip marker', CANDIDATE_SAFE_TERMS))
        self.assertFalse(label_matches('cardboard box', CANDIDATE_SAFE_TERMS))

    def test_live_obstacle_prompt_is_bounded_to_rigid_sticks_and_hand(self):
        self.assertIn('pen', DEFAULT_OBSTACLE_PROMPT)
        self.assertIn('marker', DEFAULT_OBSTACLE_PROMPT)
        self.assertIn('stick', DEFAULT_OBSTACLE_PROMPT)
        self.assertIn('hand', DEFAULT_OBSTACLE_PROMPT)
        for ignored_label in (
                'leaf', 'branch', 'wire', 'cable', 'paper', 'cardboard',
                'obstacle'):
            self.assertNotIn(ignored_label, DEFAULT_OBSTACLE_PROMPT)

    def test_only_explicit_human_mask_is_reported_unsafe(self):
        movable, unsafe = classify_refined_obstacles([
            {
                'label': 'leaf branch',
                'confidence': 0.8,
                'is_candidate_safe_class': True,
                'is_unsafe_candidate': False,
            },
            {
                'label': 'hand finger',
                'confidence': 0.8,
                'is_candidate_safe_class': False,
                'is_unsafe_candidate': True,
            },
            {
                'label': 'unknown depth foreground',
                'confidence': 0.2,
                'is_candidate_safe_class': False,
                'is_unsafe_candidate': False,
            },
        ])

        self.assertEqual([item['label'] for item in movable], ['leaf branch'])
        self.assertEqual([item['label'] for item in unsafe], ['hand finger'])

    @staticmethod
    def target_detection(confidence=0.85, box=None):
        return {
            'label': 'green cube',
            'confidence': confidence,
            'box_xyxy_pixels': box or [10.0, 10.0, 30.0, 30.0],
            'is_target_candidate': True,
        }

    def test_real_green_cube_candidate_passes_all_target_gates(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[10:30, 10:30] = (0, 255, 0)

        result = target_detection_validation(
            self.target_detection(confidence=0.80),
            image,
        )

        self.assertTrue(result['accepted'])
        self.assertGreaterEqual(
            result['semantic_confidence'],
            MIN_TARGET_SEMANTIC_CONFIDENCE,
        )
        self.assertGreaterEqual(
            result['green_fraction'],
            MIN_TARGET_GREEN_FRACTION,
        )

    def test_low_semantic_confidence_is_not_a_target_even_when_green(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[10:30, 10:30] = (0, 255, 0)

        result = target_detection_validation(
            self.target_detection(confidence=0.59),
            image,
        )

        self.assertFalse(result['accepted'])
        self.assertIn('semantic confidence', result['rejection_reasons'][0])

    def test_non_green_semantic_match_is_rejected(self):
        image = np.full((40, 40, 3), (80, 120, 160), dtype=np.uint8)
        detection = self.target_detection(confidence=0.90)

        rejected = validate_target_detections([detection], image)

        self.assertEqual(rejected, [detection])
        self.assertFalse(detection['is_target_candidate'])
        self.assertFalse(detection['target_validation']['accepted'])

    def test_generic_profile_accepts_non_green_semantic_target(self):
        image = np.full((40, 80, 3), (80, 120, 160), dtype=np.uint8)
        detection = {
            'label': 'red sphere',
            'confidence': 0.90,
            'box_xyxy_pixels': [5.0, 10.0, 65.0, 30.0],
            'is_target_candidate': True,
        }

        result = target_detection_validation(
            detection, image, target_profile='generic_open_vocab')

        self.assertTrue(result['accepted'])
        self.assertEqual(result['minimum_green_fraction'], 0.0)

    def test_implausible_cube_box_aspect_is_rejected(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[10:30, 5:65] = (0, 255, 0)

        result = target_detection_validation(
            self.target_detection(
                confidence=0.90,
                box=[5.0, 10.0, 65.0, 30.0],
            ),
            image,
        )

        self.assertFalse(result['accepted'])
        self.assertTrue(any(
            'aspect ratio' in reason
            for reason in result['rejection_reasons']
        ))

    def test_refined_non_green_mask_is_rejected(self):
        image = np.full((20, 30, 3), (70, 100, 140), dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:16, 8:22] = 255

        result = target_mask_appearance_validation(image, mask)

        self.assertFalse(result['accepted'])
        self.assertLess(result['green_fraction'], MIN_TARGET_GREEN_FRACTION)

    def test_small_target_uses_256_pixel_obstacle_crop(self):
        bounds = target_crop_bounds([300, 220, 340, 260], 640, 480)
        self.assertEqual(bounds, (192, 112, 448, 368))

    def test_obstacle_crop_is_clipped_to_image(self):
        bounds = target_crop_bounds([0, 0, 40, 40], 640, 480)
        self.assertEqual(bounds, (0, 0, 148, 148))

    def test_zero_confidence_tracked_mask_is_not_a_refresh_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            mask = np.zeros((40, 60), dtype=np.uint8)
            mask[10:30, 20:40] = 255
            cv2.imwrite(str(capture / 'detection_mask.png'), mask)
            with (capture / 'target_3d.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump(
                    {
                        'valid': True,
                        'source_u': 30.0,
                        'source_v': 20.0,
                        'measurement_confidence': 0.0,
                    },
                    stream,
                )
            self.assertIsNone(tracked_mask_target_fallback(capture))

    def test_confident_tracked_mask_remains_available_during_healthy_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            mask = np.zeros((40, 60), dtype=np.uint8)
            mask[10:30, 20:40] = 255
            image = np.zeros((40, 60, 3), dtype=np.uint8)
            image[10:30, 20:40] = (0, 255, 0)
            cv2.imwrite(str(capture / 'detection_mask.png'), mask)
            cv2.imwrite(str(capture / 'rgb.png'), image)
            with (capture / 'target_3d.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump(
                    {
                        'valid': True,
                        'source_u': 30.0,
                        'source_v': 20.0,
                        'measurement_confidence': 0.8,
                    },
                    stream,
                )
            fallback = tracked_mask_target_fallback(capture)
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback['confidence'], 0.8)

    def test_non_green_tracked_mask_cannot_perpetuate_false_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            mask = np.zeros((40, 60), dtype=np.uint8)
            mask[10:30, 20:40] = 255
            image = np.full((40, 60, 3), (80, 120, 160), dtype=np.uint8)
            cv2.imwrite(str(capture / 'detection_mask.png'), mask)
            cv2.imwrite(str(capture / 'rgb.png'), image)
            with (capture / 'target_3d.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump(
                    {
                        'valid': True,
                        'source_u': 30.0,
                        'source_v': 20.0,
                        'measurement_confidence': 0.8,
                    },
                    stream,
                )

            self.assertIsNone(tracked_mask_target_fallback(capture))


if __name__ == '__main__':
    unittest.main()
