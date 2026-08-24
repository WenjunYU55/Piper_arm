#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from sam2_live_worker import Sam2LiveWorker


class FakePredictor:
    def init_state(self, **_kwargs):
        return {'masks': {}, 'num_frames': 0}

    def add_new_frame(self, state, _image):
        index = state['num_frames']
        state['num_frames'] += 1
        return index

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        state['masks'][int(obj_id)] = np.asarray(mask) > 0
        return frame_idx, sorted(state['masks']), self.logits(state)

    def infer_single_frame(self, state, frame_idx):
        return frame_idx, sorted(state['masks']), self.logits(state)

    @staticmethod
    def logits(state):
        masks = [state['masks'][key] for key in sorted(state['masks'])]
        values = np.stack(masks)[:, None].astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(values)


class EmptyThenRecoverPredictor(FakePredictor):
    def __init__(self):
        self.return_empty = True

    def infer_single_frame(self, state, frame_idx):
        if self.return_empty:
            keys = sorted(state['masks'])
            shape = next(iter(state['masks'].values())).shape
            logits = torch.full((len(keys), 1, *shape), -1.0, dtype=torch.float32)
            return frame_idx, keys, logits
        return super().infer_single_frame(state, frame_idx)


class ShiftOnInferencePredictor(FakePredictor):
    """Move every mask one column per propagated frame."""

    def __init__(self):
        self.initializations = 0

    def init_state(self, **kwargs):
        self.initializations += 1
        return super().init_state(**kwargs)

    def infer_single_frame(self, state, frame_idx):
        state['masks'] = {
            key: np.roll(mask, 1, axis=1)
            for key, mask in state['masks'].items()
        }
        return super().infer_single_frame(state, frame_idx)


class Sam2LiveWorkerTest(unittest.TestCase):
    @staticmethod
    def write_seed(spool, name, rgb, mask):
        seed = spool / 'seeds' / name
        seed.mkdir(parents=True)
        cv2.imwrite(str(seed / 'rgb.jpg'), rgb)
        cv2.imwrite(str(seed / 'target.png'), mask)
        objects = [
            {'object_id': 1, 'role': 'target', 'label': 'green cube', 'mask_file': 'target.png'}
        ]
        with (seed / 'seed.yaml').open('w', encoding='utf-8') as stream:
            yaml.safe_dump({'objects': objects}, stream)
        (seed / 'READY').touch()

    @staticmethod
    def write_frame(spool, name, rgb):
        frame = spool / 'frames' / name
        frame.mkdir(parents=True)
        cv2.imwrite(str(frame / 'rgb.jpg'), rgb)
        with (frame / 'frame.yaml').open('w', encoding='utf-8') as stream:
            yaml.safe_dump({'image_stamp': {'sec': 1, 'nanosec': 2}, 'frame_id': 'camera'}, stream)
        (frame / 'READY').touch()

    def test_objectless_single_mask_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            worker = Sam2LiveWorker(spool, device='cpu')
            seed = spool / 'seeds' / '0001_old_single_mask'
            seed.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[5:12, 4:10] = 255
            cv2.imwrite(str(seed / 'rgb.jpg'), rgb)
            cv2.imwrite(str(seed / 'mask.png'), mask)
            with (seed / 'seed.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump({'frame_key': seed.name}, stream)
            (seed / 'READY').touch()

            with self.assertRaisesRegex(ValueError, 'no labelled objects'):
                worker.consume_latest_seed()

    def test_labelled_target_and_obstacles_are_propagated(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            worker = Sam2LiveWorker(spool, device='cpu')
            worker.predictor = FakePredictor()
            seed = spool / 'seeds' / '0001_groundingdino'
            seed.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            target = np.zeros((24, 32), dtype=np.uint8)
            target[5:12, 4:10] = 255
            movable = np.zeros_like(target)
            movable[8:15, 18:23] = 255
            unsafe = np.zeros_like(target)
            unsafe[2:7, 25:30] = 255
            cv2.imwrite(str(seed / 'rgb.jpg'), rgb)
            objects = [
                {'object_id': 1, 'role': 'target', 'label': 'green cube', 'mask_file': 'target.png'},
                {'object_id': 2, 'role': 'obstacle', 'label': 'marker', 'candidate_movable': True, 'unsafe': False, 'mask_file': 'movable.png'},
                {'object_id': 3, 'role': 'obstacle', 'label': 'hand', 'candidate_movable': False, 'unsafe': True, 'mask_file': 'unsafe.png'},
            ]
            cv2.imwrite(str(seed / 'target.png'), target)
            cv2.imwrite(str(seed / 'movable.png'), movable)
            cv2.imwrite(str(seed / 'unsafe.png'), unsafe)
            with (seed / 'seed.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump({'objects': objects}, stream)
            (seed / 'READY').touch()
            self.assertTrue(worker.process_once())

            frame = spool / 'frames' / '0002'
            frame.mkdir(parents=True)
            cv2.imwrite(str(frame / 'rgb.jpg'), rgb)
            with (frame / 'frame.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump({'image_stamp': {'sec': 1, 'nanosec': 2}, 'frame_id': 'camera'}, stream)
            (frame / 'READY').touch()
            self.assertTrue(worker.process_once())

            response = spool / 'results' / '0002'
            with (response / 'result.yaml').open('r', encoding='utf-8') as stream:
                result = yaml.safe_load(stream)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['object_count'], 3)
            ids = cv2.imread(str(response / 'object_ids.png'), cv2.IMREAD_UNCHANGED)
            self.assertEqual(set(np.unique(ids)), {0, 1, 2, 3})
            self.assertGreater(np.count_nonzero(cv2.imread(str(response / 'candidate_movable_obstacle_mask.png'), 0)), 0)
            self.assertGreater(np.count_nonzero(cv2.imread(str(response / 'unsafe_obstacle_mask.png'), 0)), 0)

    def test_empty_target_waits_for_new_seed_and_recovers(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            predictor = EmptyThenRecoverPredictor()
            worker = Sam2LiveWorker(spool, device='cpu', max_session_frames=2)
            worker.predictor = predictor
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            target = np.zeros((24, 32), dtype=np.uint8)
            target[5:12, 4:10] = 255

            self.write_seed(spool, '0001_seed', rgb, target)
            self.assertTrue(worker.process_once())
            self.write_frame(spool, '0002', rgb)
            self.assertTrue(worker.process_once())
            with (spool / 'results' / '0002' / 'result.yaml').open('r', encoding='utf-8') as stream:
                lost = yaml.safe_load(stream)
            self.assertEqual(lost['status'], 'empty_target_mask')
            self.assertEqual(lost['tracking_state'], 'WAITING_FOR_SEED')
            self.assertIsNone(worker.state)

            # A later frame must not cause the old empty mask to be used as a reset seed.
            self.write_frame(spool, '0003', rgb)
            self.assertFalse(worker.process_once())

            predictor.return_empty = False
            self.write_seed(spool, '0004_seed', rgb, target)
            self.assertTrue(worker.process_once())
            self.write_frame(spool, '0005', rgb)
            self.assertTrue(worker.process_once())
            with (spool / 'results' / '0005' / 'result.yaml').open('r', encoding='utf-8') as stream:
                recovered = yaml.safe_load(stream)
            self.assertEqual(recovered['status'], 'ok')
            self.assertEqual(recovered['tracking_state'], 'TRACKING')

    def test_reduced_inference_is_restored_to_native_output_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            worker = Sam2LiveWorker(spool, device='cpu', inference_width=16)
            worker.predictor = FakePredictor()
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            target = np.zeros((24, 32), dtype=np.uint8)
            target[4:16, 8:24] = 255
            self.write_seed(spool, '0001_seed', rgb, target)
            self.assertTrue(worker.process_once())
            self.assertEqual(worker.last_masks[1].shape, (12, 16))

            self.write_frame(spool, '0002', rgb)
            self.assertTrue(worker.process_once())
            result_dir = spool / 'results' / '0002'
            mask = cv2.imread(str(result_dir / 'mask.png'), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, (24, 32))
            with (result_dir / 'result.yaml').open('r', encoding='utf-8') as stream:
                result = yaml.safe_load(stream)
            self.assertEqual(result['inference_size'], [16, 12])
            self.assertEqual(result['output_size'], [32, 24])

    def test_obstacle_only_seed_remains_live_without_claiming_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            worker = Sam2LiveWorker(spool, device='cpu')
            worker.predictor = ShiftOnInferencePredictor()
            rgb = np.zeros((12, 16, 3), dtype=np.uint8)
            obstacle = np.zeros((12, 16), dtype=np.uint8)
            obstacle[4:8, 2:5] = 255
            seed = spool / 'seeds' / '0001_obstacle_only'
            seed.mkdir(parents=True)
            cv2.imwrite(str(seed / 'rgb.jpg'), rgb)
            cv2.imwrite(str(seed / 'obstacle.png'), obstacle)
            with (seed / 'seed.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump({'objects': [{
                    'object_id': 2,
                    'role': 'obstacle',
                    'label': 'hand',
                    'unsafe': True,
                    'mask_file': 'obstacle.png',
                }]}, stream)
            (seed / 'READY').touch()
            self.assertTrue(worker.process_once())

            self.write_frame(spool, '0002', rgb)
            self.assertTrue(worker.process_once())
            with (spool / 'results' / '0002' / 'result.yaml').open(
                    'r', encoding='utf-8') as stream:
                result = yaml.safe_load(stream)
            ids = cv2.imread(
                str(spool / 'results' / '0002' / 'object_ids.png'),
                cv2.IMREAD_UNCHANGED)
            self.assertEqual(result['status'], 'empty_target_mask')
            self.assertEqual(result['tracking_state'], 'OBSTACLE_ONLY')
            self.assertIsNotNone(worker.state)
            self.assertEqual(set(np.unique(ids)), {0, 2})

    def test_history_bound_does_not_promote_propagated_mask_to_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            worker = Sam2LiveWorker(spool, device='cpu', max_session_frames=2)
            predictor = ShiftOnInferencePredictor()
            worker.predictor = predictor
            rgb = np.zeros((12, 16, 3), dtype=np.uint8)
            target = np.zeros((12, 16), dtype=np.uint8)
            target[4:8, 2:5] = 255

            self.write_seed(spool, '0001_seed', rgb, target)
            self.assertTrue(worker.process_once())
            self.write_frame(spool, '0002', rgb)
            self.assertTrue(worker.process_once())
            self.write_frame(spool, '0003', rgb)
            self.assertTrue(worker.process_once())

            mask = cv2.imread(
                str(spool / 'results' / '0003' / 'mask.png'),
                cv2.IMREAD_GRAYSCALE,
            )
            expected = np.zeros_like(target)
            expected[4:8, 4:7] = 255
            np.testing.assert_array_equal(mask, expected)
            self.assertEqual(worker.session_frames, 3)
            self.assertEqual(predictor.initializations, 1)
            with (spool / 'results' / '0003' / 'result.yaml').open(
                    'r', encoding='utf-8') as stream:
                result = yaml.safe_load(stream)
            self.assertFalse(result['prediction_promoted_to_conditioning'])

    def test_predictor_history_compaction_retains_conditioning_and_recent_memory(self):
        def records(indices):
            return {index: 'frame-%d' % index for index in indices}

        state = {
            'images': torch.arange(6, dtype=torch.float32).reshape(6, 1),
            'num_frames': 6,
            'cached_features': records(range(6)),
            'frames_already_tracked': records(range(6)),
            'output_dict': {
                'cond_frame_outputs': records([0]),
                'non_cond_frame_outputs': records([1, 2, 3, 4, 5]),
            },
            'output_dict_per_obj': {0: {
                'cond_frame_outputs': records([0]),
                'non_cond_frame_outputs': records([1, 2, 3, 4, 5]),
            }},
            'temp_output_dict_per_obj': {0: {
                'cond_frame_outputs': {},
                'non_cond_frame_outputs': {},
            }},
            'point_inputs_per_obj': {0: {}},
            'mask_inputs_per_obj': {0: records([0])},
            'consolidated_frame_inds': {
                'cond_frame_outputs': {0},
                'non_cond_frame_outputs': {1, 2, 3, 4, 5},
            },
        }

        retained = Sam2LiveWorker.compact_predictor_history(state, 3)

        self.assertEqual(retained, 3)
        self.assertEqual(state['num_frames'], 3)
        self.assertEqual(state['images'].flatten().tolist(), [0.0, 4.0, 5.0])
        self.assertEqual(state['output_dict']['cond_frame_outputs'], {0: 'frame-0'})
        self.assertEqual(
            state['output_dict']['non_cond_frame_outputs'],
            {1: 'frame-4', 2: 'frame-5'},
        )
        self.assertEqual(state['consolidated_frame_inds']['cond_frame_outputs'], {0})


if __name__ == '__main__':
    unittest.main()
