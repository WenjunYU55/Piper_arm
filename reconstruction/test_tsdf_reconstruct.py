import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).with_name('tsdf_reconstruct.py')
SPEC = importlib.util.spec_from_file_location('tsdf_reconstruct', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stored_base_camera_pose_is_inverted_for_open3d():
    base_camera = np.eye(4)
    base_camera[0, 3] = 0.25
    extrinsic = MODULE.camera_extrinsic_from_metadata({
        'camera_transform': {
            'available': True,
            'matrix_4x4': base_camera.tolist(),
        },
    })
    assert extrinsic[0, 3] == pytest.approx(-0.25)


def test_missing_timestamped_pose_fails_closed():
    with pytest.raises(ValueError, match='timestamped camera transform'):
        MODULE.camera_extrinsic_from_metadata({})


@pytest.mark.parametrize('count', [8, 12, 19, 24])
def test_feature_driven_capture_count_is_accepted(count):
    paths = [Path('view_%03d_metadata.yaml' % index) for index in range(count)]
    assert MODULE.validate_capture_set({'capture_count': count}, paths) == paths


@pytest.mark.parametrize('count', [0, 7, 25])
def test_capture_count_outside_bounded_contract_is_rejected(count):
    with pytest.raises(ValueError, match='8-24'):
        MODULE.validate_capture_set(
            {'capture_count': count}, [Path(str(index)) for index in range(count)])


def test_manifest_and_frame_metadata_count_must_match():
    with pytest.raises(ValueError, match='does not match'):
        MODULE.validate_capture_set(
            {'capture_count': 14}, [Path(str(index)) for index in range(13)])


def test_registration_correction_is_bounded():
    correction = np.eye(4)
    correction[0, 3] = 0.01
    assert MODULE.correction_rejection(
        correction, fitness=0.5, inlier_rmse=0.004) == ''
    correction[0, 3] = 0.03
    assert 'translation' in MODULE.correction_rejection(
        correction, fitness=0.5, inlier_rmse=0.004)
    assert 'overlap' in MODULE.correction_rejection(
        np.eye(4), fitness=0.01, inlier_rmse=0.004)


def test_manifest_integrity_rejects_changed_artifact(tmp_path):
    artifact = tmp_path / 'frame.bin'
    artifact.write_bytes(b'original')
    unsigned = {
        'capture_count': 13,
        'files': [{
            'path': 'frame.bin',
            'bytes': len(b'original'),
            'sha256': hashlib.sha256(b'original').hexdigest(),
        }],
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    manifest = dict(unsigned)
    manifest['manifest_sha256'] = hashlib.sha256(encoded).hexdigest()
    assert MODULE.validate_manifest_integrity(tmp_path, manifest)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='size changed'):
        MODULE.validate_manifest_integrity(tmp_path, manifest)
