"""Verify frame correspondence, handedness, immutable exports and viewing direction."""

import json
from pathlib import Path

import numpy as np
import pytest

from reconstruction.coordinate_frames import (
    BUNDLESDF_PIN, arrow_transform, bundlesdf_frame, export_bundlesdf,
    export_report, file_hash, report_frame, rigid_transform, view_layout,
    write_report_frames,
)


def saved_run(root):
    """Use two different noncommuting rigid transforms to detect wrong composition."""
    (root / 'input').mkdir(parents=True)
    (root / 'result/ob_in_cam').mkdir(parents=True)
    base_camera = np.array([[0., -1, 0, .5], [1, 0, 0, .2], [0, 0, 1, .3], [0, 0, 0, 1]])
    camera_object = np.array([[1., 0, 0, .1], [0, 0, -1, .2], [0, 1, 0, .4], [0, 0, 0, 1]])
    (root / 'run.json').write_text(json.dumps(dict(upstream_commit=BUNDLESDF_PIN)))
    (root / 'input/robot_camera_poses.json').write_text(json.dumps(dict(
        pose_convention='T_base_camera maps optical coordinates to base', base_frame='base_link', frames=[
            dict(filename='00000009.png', rgb_stamp_ns=99, T_base_camera=np.eye(4).tolist()),
            dict(filename='00000000.png', native_frame=17, rgb_stamp_ns=123,
                 T_base_camera=base_camera.tolist())])))
    np.savetxt(root / 'result/ob_in_cam/00000000.txt', camera_object)
    (root / 'result/textured_mesh.obj').write_text(
        'mtllib material_0.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n'
        'vn 0 0 1\nvt 0 0\nusemtl material_0\nf 1/1/1 2/1/1 3/1/1\n')
    (root / 'result/material_0.mtl').write_text('newmtl material_0\nmap_Kd material_0.png\n')
    (root / 'result/material_0.png').write_bytes(b'retained texture bytes')
    return base_camera, camera_object


def test_anchor_uses_matching_frame_and_correct_transform_direction(tmp_path):
    base_camera, camera_object = saved_run(tmp_path)
    frame = bundlesdf_frame(tmp_path)
    assert frame['anchor']['native_frame'] == 17
    assert frame['anchor']['rgb_stamp_ns'] == 123
    point = np.array([.02, .03, .04, 1])
    assert np.allclose(np.asarray(frame['T_base_object']) @ point,
                       base_camera @ (camera_object @ point))
    assert not frame['accuracy_qualified']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'reflection', 'revision'])
def test_unverifiable_orientation_is_rejected(tmp_path, damage):
    saved_run(tmp_path)
    path = tmp_path / 'input/robot_camera_poses.json'
    data = json.loads(path.read_text())
    if damage == 'missing':
        data['frames'].pop()
    elif damage == 'duplicate':
        data['frames'].append(data['frames'][-1])
    elif damage == 'reflection':
        data['frames'][-1]['T_base_camera'][2][2] = -1
    else:
        (tmp_path / 'run.json').write_text('{}')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        bundlesdf_frame(tmp_path)


def test_export_preserves_sources_uv_faces_and_rotates_normals(tmp_path):
    root = tmp_path / 'run'
    base_camera, camera_object = saved_run(root)
    before = {p: file_hash(p) for p in root.rglob('*') if p.is_file()}
    output = tmp_path / 'base_link'
    frame = export_bundlesdf(root, output)
    lines = (output / 'textured_mesh.obj').read_text().splitlines()
    vertices = np.array([[float(v) for v in s.split()[1:4]] for s in lines if s.startswith('v ')])
    transform = base_camera @ camera_object
    assert np.allclose(vertices[0], transform[:3, 3])
    assert np.isclose(np.linalg.norm(vertices[1] - vertices[0]), 1.)
    normal = next(s for s in lines if s.startswith('vn '))
    assert np.allclose([float(v) for v in normal.split()[1:]], transform[:3, 2])
    assert 'f 1/1/1 2/1/1 3/1/1' in lines
    assert 'vt 0 0' in lines
    assert (output / 'material_0.png').read_bytes() == b'retained texture bytes'
    assert frame['geometry_sha256'] == file_hash(output / 'textured_mesh.obj')
    assert before == {p: file_hash(p) for p in before}
    with pytest.raises(FileExistsError):
        export_bundlesdf(root, output)


def test_robot_view_is_right_handed_z_up_and_toward_base():
    points = np.array([[.4, -.01, .1], [.45, .01, .15]])
    layout = view_layout(points)
    pose = rigid_transform(layout['camera_pose'])
    assert pose[0, 3] < layout['center'][0]
    assert pose[2, 3] > layout['center'][2]
    assert layout['up'][2] > 0
    assert layout['arrows'][-1][1][0] < 0
    for origin, vector, _ in layout['arrows']:
        arrow = rigid_transform(arrow_transform(origin, vector))
        assert np.allclose(arrow[:3, 2], vector / np.linalg.norm(vector))
    centered = view_layout(np.array([[-.01, -.01, -.01], [.01, .01, .01]]))
    assert len(centered['arrows']) == 3
    assert 'undefined' in centered['legend']


def test_all_report_geometry_variants_get_hash_bound_frames(tmp_path):
    report = dict(scan_dir='/original/scan', frame_inputs=[dict(
        T_base_camera=np.eye(4).tolist(), base_frame='base_link')])
    for key in ('mesh_path', 'raw_mesh_path', 'input_cloud_path', 'measured_cloud_path',
                'consensus_cloud_path', 'textured_mesh_path'):
        path = tmp_path / (key + '.ply')
        path.write_bytes(key.encode())
        report[key] = str(path)
    write_report_frames(report)
    assert report_frame(report)['frame_id'] == 'base_link'
    for key, value in report.items():
        if key.endswith('_path'):
            frame = json.loads(Path(value + '.frame.json').read_text())
            assert frame['geometry_sha256'] == file_hash(value)
    with pytest.raises(ValueError, match='unverified'):
        report_frame({})
    path = tmp_path / 'quality.json'
    path.write_text(json.dumps(report))
    output = tmp_path / 'portable'
    exported = export_report(path, output)
    assert Path(exported['mesh_path']).parent == output
    assert file_hash(exported['mesh_path']) == file_hash(report['mesh_path'])
