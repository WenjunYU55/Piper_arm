"""Verify the external interpreter boundary and synthetic geometry provenance."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_native_launcher_replaces_ros_paths(tmp_path):
    """A sourced Foxy overlay must never enter the offline Python/native search paths."""
    environment = dict(os.environ, PIPER_BUNDLESDF_PYTHON=sys.executable,
                       PIPER_BUNDLESDF_RUNTIME=str(tmp_path),
                       PIPER_BUNDLESDF_CUDA_HOME=str(tmp_path / 'cuda'),
                       PYTHONHOME='/opt/ros/foxy/fake',
                       PYTHONPATH='/opt/ros/foxy/fake', LD_LIBRARY_PATH='/opt/ros/foxy/lib')
    result = subprocess.run([
        'bash', str(ROOT / 'tools/bundlesdf/python_native.sh'), '-c',
        'import os,json; print(json.dumps({k:os.environ[k] for k in '
        '["PYTHONPATH","LD_LIBRARY_PATH","CUDA_HOME"]}))',
    ], env=environment, capture_output=True, text=True, check=True)
    values = json.loads(result.stdout)
    assert values['PYTHONPATH'] == str(tmp_path / 'upstream')
    assert all('/opt/ros' not in value for value in values.values())


def test_native_launcher_missing_environment(tmp_path):
    """Missing offline Python is actionable and never falls back to system ROS Python."""
    result = subprocess.run([
        'bash', str(ROOT / 'tools/bundlesdf/python_native.sh'), '-c', 'print(1)',
    ], env=dict(os.environ, PIPER_BUNDLESDF_PYTHON=str(tmp_path / 'missing')),
        capture_output=True, text=True)
    assert result.returncode == 2
    assert 'setup_native.sh' in result.stderr


def test_analytic_smoke_depth_and_camera_convention():
    """Back-project saved depth into the known cube envelope using camera-to-base TF."""
    spec = importlib.util.spec_from_file_location(
        'synthetic_smoke', ROOT / 'tools/bundlesdf/synthetic_smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rgb, depth, mask, K, transform, target = module.render(15)
    y, x = np.nonzero(mask)
    z = depth[y, x] * .001
    camera = np.column_stack(((x-K[0, 2])/K[0, 0]*z, (y-K[1, 2])/K[1, 1]*z, z))
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    assert np.max(np.abs(base-target)) < .071
    assert np.allclose(np.max(np.abs(base-target), axis=1), .07, atol=.001)
    assert rgb.dtype == np.uint8 and depth.dtype == np.uint16
    assert mask.any() and np.count_nonzero(depth) == np.count_nonzero(mask)


def test_output_validation_rejects_degenerate_texture_and_bad_pose(tmp_path):
    """A process exit code and OBJ header cannot hide the observed one-texel failure."""
    import cv2
    import pytest
    spec = importlib.util.spec_from_file_location(
        'output_validation', ROOT / 'tools/bundlesdf/output_validation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'textured_mesh.obj').write_text(
        'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 2/2 3/3\n')
    (tmp_path / 'ob_in_cam').mkdir()
    pose_file = tmp_path / 'ob_in_cam/0.txt'
    np.savetxt(str(pose_file), np.eye(4))
    texture = np.zeros((4, 4, 3), dtype=np.uint8)
    texture[0, 0] = 100
    cv2.imwrite(str(tmp_path / 'material_0.png'), texture)
    with pytest.raises(ValueError, match='degenerate mesh texture'):
        module.inspect_result(tmp_path)
    texture[1, 1] = 150
    cv2.imwrite(str(tmp_path / 'material_0.png'), texture)
    result = module.inspect_result(tmp_path)
    assert result['texture_nonzero_pixels'] == 2
    assert result['estimated_pose_count'] == 1
    assert not result['accuracy_qualified']
    pose = np.eye(4)
    pose[0, 0] = -1
    np.savetxt(str(pose_file), pose)
    with pytest.raises(ValueError, match='invalid T_camera_object'):
        module.inspect_result(tmp_path)
