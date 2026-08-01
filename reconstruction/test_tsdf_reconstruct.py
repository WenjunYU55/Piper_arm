import importlib.util
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
