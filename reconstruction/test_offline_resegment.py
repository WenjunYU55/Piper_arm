import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


PATH = Path(__file__).with_name('offline_resegment.py')
SPEC = importlib.util.spec_from_file_location('offline_resegment', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(path, root):
    data = path.read_bytes()
    return {
        'path': path.relative_to(root).as_posix(),
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


def _scan(tmp_path):
    scan = tmp_path / 'scan_001'
    frames = scan / 'frames'
    frames.mkdir(parents=True)
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[4:16, 4:16, 1] = 255
    depth = np.full((20, 20), 400, dtype=np.uint16)
    live = np.zeros((20, 20), dtype=np.uint8)
    live[2:19, 2:19] = 255
    rgb_path = frames / 'view_000_rgb.png'
    depth_png = frames / 'view_000_depth.png'
    depth_npy = frames / 'view_000_depth.npy'
    mask_path = frames / 'view_000_mask.png'
    metadata_path = frames / 'view_000_metadata.yaml'
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(depth_png), depth)
    assert cv2.imwrite(str(mask_path), live)
    with depth_npy.open('wb') as stream:
        np.save(stream, depth)
    metadata_path.write_text(
        'capture_schema_version: 1\n'
        'rgb_file_path: /old/view_000_rgb.png\n'
        'depth_png_file_path: /old/view_000_depth.png\n'
        'mask_file_path: /old/view_000_mask.png\n',
        encoding='utf-8')
    unsigned = {
        'capture_count': 1,
        'target_label': 'green cube',
        'target_profile': 'green_cube',
        'target_prompt': 'green cube .',
        'files': [
            _record(path, scan) for path in (
                rgb_path, depth_png, depth_npy, mask_path, metadata_path)],
    }
    manifest = dict(
        unsigned, manifest_sha256=MODULE.canonical_sha256(unsigned))
    (scan / 'manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    return scan, mask_path


def test_fresh_mask_is_clipped_to_grounding_box(tmp_path):
    mask = np.full((20, 20), 255, dtype=np.uint8)
    path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(path), mask)
    refined = {
        'status': 'ok',
        'masks': [{
            'mask_role': 'target', 'prompt_source': 'groundingdino',
            'mask_png': str(path), 'sam2_score': 0.9, 'confidence': 0.8,
            'box_xyxy_pixels': [2.0, 2.0, 18.0, 18.0],
        }],
    }
    clipped, report = MODULE._fresh_target_mask(refined, (20, 20, 3))
    assert np.count_nonzero(clipped) == 256
    assert report['pixels_removed_outside_groundingdino_box'] == 144


def test_live_mask_fallback_is_not_an_authoritative_offline_mask(tmp_path):
    mask = np.full((12, 12), 255, dtype=np.uint8)
    path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(path), mask)
    refined = {
        'status': 'ok',
        'masks': [{
            'mask_role': 'target', 'prompt_source': 'tracked_target_mask',
            'mask_png': str(path), 'sam2_score': 0.9, 'confidence': 0.8,
            'box_xyxy_pixels': [0.0, 0.0, 12.0, 12.0],
        }],
    }
    with pytest.raises(ValueError, match='0 authoritative'):
        MODULE._fresh_target_mask(refined, (12, 12, 3))


def test_scan_resegmentation_preserves_source_and_omits_live_mask_fallback(
        tmp_path, monkeypatch):
    scan, live_mask_path = _scan(tmp_path)
    live_before = live_mask_path.read_bytes()
    monkeypatch.setattr(MODULE, '_model_identity', lambda: {
        'fake': {'path': '/model', 'sha256': 'a' * 64}})

    def detector(**kwargs):
        assert not (kwargs['capture_dir'] / 'detection_mask.png').exists()
        boxes = kwargs['output_root'] / kwargs['capture_dir'].name \
            / 'groundingdino' / 'groundingdino_boxes.yaml'
        boxes.parent.mkdir(parents=True, exist_ok=True)
        boxes.write_text('{}', encoding='utf-8')
        return {
            'summary': {'target_source': 'groundingdino'},
            'outputs': {'boxes_yaml': str(boxes)},
        }

    def refiner(**kwargs):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 4:16] = 255
        path = kwargs['output_root'] / kwargs['capture_dir'].name \
            / 'sam2' / 'mask_00_target.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), mask)
        return {
            'status': 'ok',
            'masks': [{
                'mask_role': 'target', 'prompt_source': 'groundingdino',
                'mask_png': str(path), 'sam2_score': 0.95,
                'confidence': 0.9,
                'box_xyxy_pixels': [4.0, 4.0, 16.0, 16.0],
            }],
        }

    index_path, index = MODULE.resegment_scan(
        scan, detector=detector, refiner=refiner)
    assert index_path.is_file()
    assert index['frame_count'] == 1
    assert index['live_mask_used_as_model_fallback'] is False
    assert live_mask_path.read_bytes() == live_before
    derived = index_path.parent / index['frames'][0]['mask_path']
    assert np.count_nonzero(cv2.imread(str(derived), 0)) == 144
    assert not (index_path.parent / 'sources').exists()
    comparison = index_path.parent / index['frames'][0]['comparison_path']
    assert comparison.is_file()
