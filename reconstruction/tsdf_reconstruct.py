#!/usr/bin/env python3
"""Offline masked RGB-D TSDF prototype for a completed PiPER scan."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def camera_extrinsic_from_metadata(metadata):
    """Convert stored T_base_camera into Open3D's T_camera_base extrinsic."""
    transform = metadata.get('camera_transform', {})
    matrix = np.asarray(transform.get('matrix_4x4'), dtype=float)
    if not transform.get('available') or matrix.shape != (4, 4) \
            or not np.all(np.isfinite(matrix)):
        raise ValueError('frame has no finite timestamped camera transform')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('camera transform is not homogeneous')
    return np.linalg.inv(matrix)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('PyYAML is required for scan metadata') from exc
    with open(path, 'r', encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError('frame metadata is not an object')
    return value


def reconstruct(scan_dir, output_path, voxel_length=0.003,
                sdf_trunc=0.015, depth_trunc=1.5):
    try:
        import cv2
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            'install reconstruction/requirements.txt in an isolated environment') from exc
    scan = Path(scan_dir).resolve()
    with open(scan / 'manifest.json', 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    if int(manifest.get('capture_count', 0)) != 13:
        raise ValueError('TSDF prototype requires exactly 13 captured views')
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    for metadata_path in sorted((scan / 'frames').glob('view_*_metadata.yaml')):
        metadata = load_metadata(metadata_path)
        info = metadata.get('camera_info', {})
        k = info.get('k', [])
        if not info.get('available') or len(k) != 9:
            raise ValueError('%s has no camera intrinsics' % metadata_path.name)
        rgb = cv2.imread(str(metadata['rgb_file_path']), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(metadata['depth_png_file_path']), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(metadata['mask_file_path']), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise ValueError('%s has missing RGB/depth/mask data' % metadata_path.name)
        valid = mask > 0
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb[~valid] = 0
        depth = np.asarray(depth, dtype=np.uint16)
        depth[~valid] = 0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(info['width']), int(info['height']),
            float(k[0]), float(k[4]), float(k[2]), float(k[5]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
            depth_scale=1000.0, depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd, intrinsic, camera_extrinsic_from_metadata(metadata))
        integrated += 1
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise OSError('Open3D failed to save %s' % output)
    report = {
        'scan_dir': str(scan),
        'input_manifest_sha256': sha256_file(scan / 'manifest.json'),
        'integrated_views': integrated,
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.triangles),
        'mesh_path': str(output),
        'mesh_sha256': sha256_file(output),
        'voxel_length_m': float(voxel_length),
        'sdf_trunc_m': float(sdf_trunc),
    }
    with open(output.with_suffix(output.suffix + '.quality.json'),
              'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--voxel-length', type=float, default=0.003)
    parser.add_argument('--sdf-trunc', type=float, default=0.015)
    parser.add_argument('--depth-trunc', type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(reconstruct(
        args.scan_dir, args.output, args.voxel_length,
        args.sdf_trunc, args.depth_trunc), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
