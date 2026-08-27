#!/usr/bin/env python3
"""Interactive Open3D validation view for one generated quality report."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def camera_axes(transform, size=0.025):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(np.asarray(transform, dtype=float))
    return frame


def expected_wireframe(report, geometry):
    check = report.get('mesh_metrics', {}).get('dimension_check') or {}
    dimensions = np.asarray(check.get('expected_m', []), dtype=float)
    if dimensions.shape != (3,):
        return None
    observed = geometry.get_oriented_bounding_box(robust=True)
    box = o3d.geometry.OrientedBoundingBox(
        observed.center, observed.R, dimensions)
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    lines.paint_uniform_color([1.0, 0.1, 0.1])
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--mesh', choices=('cleaned', 'raw', 'measured'),
                        default='cleaned')
    parser.add_argument('--show-input', action='store_true')
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    with open(report_path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    if args.mesh == 'measured':
        measured_path = Path(str(report.get(
            'measured_cloud_path', ''))).resolve()
        if not measured_path.is_file():
            raise SystemExit(
                'reported measured point cloud is missing: %s'
                % measured_path)
        primary = o3d.io.read_point_cloud(str(measured_path))
        if not primary.has_colors():
            primary.paint_uniform_color([0.1, 0.35, 1.0])
    else:
        mesh_key = 'raw_mesh_path' if args.mesh == 'raw' else 'mesh_path'
        mesh_path = Path(str(report.get(mesh_key, ''))).resolve()
        if not mesh_path.is_file():
            raise SystemExit('reported mesh is missing: %s' % mesh_path)
        primary = o3d.io.read_triangle_mesh(str(mesh_path))
        primary.compute_vertex_normals()
    geometries = [primary]
    if args.mesh != 'measured':
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    wireframe = expected_wireframe(report, primary)
    if wireframe is not None:
        geometries.append(wireframe)
    if args.mesh != 'measured':
        for frame in report.get('frame_inputs', []):
            transform = np.asarray(frame.get('T_base_camera'), dtype=float)
            if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
                geometries.append(camera_axes(transform))
    if args.show_input and args.mesh != 'measured':
        cloud_path = Path(str(report.get(
            'measured_cloud_path', report.get(
                'input_cloud_path', '')))).resolve()
        if not cloud_path.is_file():
            raise SystemExit('reported masked input cloud is missing: %s' % cloud_path)
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        cloud.paint_uniform_color([0.1, 0.35, 1.0])
        geometries.append(cloud)
    o3d.visualization.draw_geometries(
        geometries,
        window_name='PiPER Reconstruction Validation (%s)' % args.mesh,
        mesh_show_back_face=True)


if __name__ == '__main__':
    main()
