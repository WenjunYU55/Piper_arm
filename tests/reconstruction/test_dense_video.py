"""Dense offline video contracts; fake segmentation is explicit in these unit tests."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/bundlesdf'))
from dataset import validate, convert  # noqa: E402
from extract_video import match_stamps  # noqa: E402
from segment_video import prepare, qualify, validate_extraction  # noqa: E402
from output_validation import tracking_health  # noqa: E402
from piper_mobile_manipulation.reconstruction_recording.core import RecorderConfig  # noqa: E402


class FakeSegmenter:
    """Exercise orchestration and gates; not evidence of neural mask tracking."""

    provenance = {'model': 'TEST_DOUBLE_NOT_SAM2'}
    maximum_retained_frames = 1

    def __init__(self, interrupt_at=None):
        self.calls = 0
        self.interrupt_at = interrupt_at

    def infer(self, rgb, seed=None):
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt('test interruption')
        self.calls += 1
        if seed is not None:
            self.mask = seed.copy()
        return self.mask.copy()


def extracted_fixture(root, count=12, encoding='16UC1'):
    root.mkdir()
    frames = []
    for i in range(count):
        folder = root / ('%08d' % i)
        folder.mkdir()
        rgb = np.repeat((np.indices((120, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)[:, :, None], 3, 2)
        depth = (np.full((120, 160), 500, np.uint16) if encoding == '16UC1'
                 else np.full((120, 160), .5, np.float32))
        cv2.imwrite(str(folder / 'rgb.png'), rgb)
        np.save(str(folder / 'depth.npy'), depth)
        if i == 0:
            mask = np.zeros((120, 160), np.uint8)
            mask[45:75, 65:95] = 255
            cv2.imwrite(str(folder / 'seed.png'), mask)
        stamp = 1_000_000_000 + i*33_333_333
        frames.append(dict(index=i, rgb=folder.name+'/rgb.png', depth=folder.name+'/depth.npy',
                           K=[[200., 0, 80], [0, 200., 60], [0, 0, 1.]], width=160, height=120,
                           base_frame='base_link', camera_optical_frame='camera_color_optical_frame',
                           rgb_stamp_ns=stamp, depth_stamp_ns=stamp, tf_stamp_ns=stamp,
                           T_base_camera=np.eye(4).tolist(), tf_error=None, accepted_view=1 if i == 0 else 0,
                           distortion_model='plumb_bob', distortion_coefficients=[0.]*5,
                           depth_encoding=encoding, depth_dtype=str(depth.dtype),
                           depth_scale_m=.001 if encoding == '16UC1' else 1.))
    data = dict(schema_version=1, dataset_type='piper_unsegmented_rgbd_video', state='ready',
                session_id='fixture', source_bag='/test/fixture/bag', mission_capture_count=1,
                complete=True, configuration=asdict(RecorderConfig()), coverage_target_center=[0, 0, .5],
                seed_mask='00000000/seed.png', frames=frames)
    (root / 'extraction.json').write_text(json.dumps(data))
    return data


def test_pairing_exact_first_unique_and_order_independent():
    assert match_stamps([110, 100, 100, 90], [100, 110, 80, 80], 10) == {100: 100, 110: 110, 90: 80}
    assert match_stamps([10, 20], [11], 2) == {10: 11}
    assert match_stamps([10], [13], 2) == {}
    with pytest.raises(ValueError):
        match_stamps([1], [1], -1)


@pytest.mark.parametrize('encoding', ['16UC1', '32FC1'])
def test_every_30hz_frame_without_moving_state_or_later_live_masks(tmp_path, encoding):
    source = tmp_path / 'extracted'
    extracted_fixture(source, encoding=encoding)
    original = (source / '00000000/rgb.png').read_bytes()
    predictor = FakeSegmenter()
    out = tmp_path / 'prepared'
    report = prepare(source, out, predictor=predictor)
    assert predictor.calls == report['accepted'] == 12
    assert report['mission_capture_count'] == 1
    assert report['rejected_by_reason'] == {}
    assert report['frames'][1]['source'] == 'recorded_video'
    assert report['frames'][1]['mask_source'] == 'sam2_offline_video'
    assert (source / '00000000/rgb.png').read_bytes() == original
    assert validate(out)[0]['frames'] == 12
    assert convert(out, tmp_path / 'converted')['converted_frames'] == 12
    with pytest.raises(FileExistsError):
        prepare(source, out, predictor=FakeSegmenter())


def test_tf_rejection_does_not_stop_temporal_updates(tmp_path):
    source = tmp_path / 'extracted'
    data = extracted_fixture(source)
    data['frames'][4]['T_base_camera'] = None
    data['frames'][4]['tf_error'] = 'missing_or_invalid_tf'
    (source / 'extraction.json').write_text(json.dumps(data))
    predictor = FakeSegmenter()
    report = prepare(source, tmp_path / 'prepared', predictor=predictor)
    assert predictor.calls == 12
    assert report['accepted'] == 11
    assert report['rejected_by_reason'] == {'missing_tf': 1}


def test_interruption_retains_valid_partial_and_next_run_is_independent(tmp_path):
    source = tmp_path / 'extracted'
    extracted_fixture(source)
    with pytest.raises(KeyboardInterrupt):
        prepare(source, tmp_path / 'cancelled', predictor=FakeSegmenter(interrupt_at=6))
    partial = tmp_path / 'cancelled.incomplete'
    report, _ = validate(partial, allow_incomplete=True)
    assert report['frames'] == 6 and not report['complete']
    assert not list(tmp_path.glob('*.inprogress'))
    assert prepare(source, tmp_path / 'next', predictor=FakeSegmenter())['accepted'] == 12


def test_dense_quality_gate_still_rejects_clipping_depth_and_expansion(tmp_path):
    source = tmp_path / 'extracted'
    data = extracted_fixture(source)
    entry = data['frames'][0]
    rgb = cv2.imread(str(source / entry['rgb']))
    depth = np.load(str(source / entry['depth']))
    mask = cv2.imread(str(source / data['seed_mask']), 0)
    cfg = RecorderConfig()
    with pytest.raises(ValueError, match='mask_expansion'):
        qualify(entry, rgb, depth, mask, data['coverage_target_center'], cfg, 1e-8)
    with pytest.raises(ValueError, match='insufficient_depth'):
        qualify(entry, rgb, depth*0, mask, data['coverage_target_center'], cfg, None)
    mask[0, 80] = 255
    with pytest.raises(ValueError, match='target_clipped'):
        qualify(entry, rgb, depth, mask, data['coverage_target_center'], cfg, None)


def test_upstream_failures_are_not_successful_pose_counts():
    result = tracking_health('frame 00000004 few global_corres, mark as FAIL\n'
                             'frame 00000004 too big FAIL\nframe 00000099 too big FAIL\n')
    assert result['explicit_failed_frame_ids'] == ['00000004', '00000099']
    assert result['assessment'] == 'failures_reported'
    assert not result['accuracy_qualified']


@pytest.mark.parametrize('mutation,reason', [
    (lambda d: d['frames'][1].update(rgb_stamp_ns=1_000_000_000), 'Nonmonotonic'),
    (lambda d: d['frames'][1].update(depth_stamp_ns=1), 'synchronization'),
    (lambda d: d['frames'][1].update(tf_stamp_ns=1), 'TF'),
    (lambda d: d['frames'][1].update(distortion_coefficients=[.1]*5), 'calibration changed'),
])
def test_sensor_contract_failure_before_gpu(tmp_path, mutation, reason):
    data = extracted_fixture(tmp_path / 'extracted')
    mutation(data)
    with pytest.raises(ValueError, match=reason):
        validate_extraction(data)
