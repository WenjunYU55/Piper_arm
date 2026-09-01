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
    parser.add_argument(
        '--mesh', choices=(
            'cleaned', 'raw', 'measured', 'superposition', 'consensus',
            'textured'),
        default='cleaned')
    parser.add_argument('--show-input', action='store_true')
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    with open(report_path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    source = report
    if args.mesh in ('superposition', 'consensus', 'textured') \
            and str(report.get('registration_mode', '')) \
            != 'constrained_superposition':
        candidates = report.get('candidate_reports') or {}
        candidate = candidates.get('constrained_superposition')
        source = candidate if isinstance(candidate, dict) else {}
    point_cloud_variant = args.mesh in (
        'measured', 'superposition', 'consensus')
    if point_cloud_variant:
        cloud_key = (
            'consensus_cloud_path'
            if args.mesh == 'consensus' else 'measured_cloud_path')
        cloud_path = Path(str(source.get(cloud_key, ''))).resolve()
        if not cloud_path.is_file():
            raise SystemExit(
                'reported %s point cloud is missing: %s'
                % (args.mesh, cloud_path))
        primary = o3d.io.read_point_cloud(str(cloud_path))
        if not primary.has_colors():
            primary.paint_uniform_color([0.1, 0.35, 1.0])
    else:
        mesh_key = (
            'raw_mesh_path' if args.mesh == 'raw' else
            'textured_mesh_path' if args.mesh == 'textured' else 'mesh_path')
        mesh_path = Path(str(source.get(mesh_key, ''))).resolve()
        if not mesh_path.is_file():
            raise SystemExit('reported mesh is missing: %s' % mesh_path)
        primary = o3d.io.read_triangle_mesh(
            str(mesh_path), enable_post_processing=(args.mesh == 'textured'))
        primary.compute_vertex_normals()
    geometries = [primary]
    if not point_cloud_variant:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    wireframe = expected_wireframe(source, primary)
    if wireframe is not None:
        geometries.append(wireframe)
    if not point_cloud_variant:
        for frame in source.get('frame_inputs', []):
            transform = np.asarray(frame.get('T_base_camera'), dtype=float)
            if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
                geometries.append(camera_axes(transform))
    if args.show_input and not point_cloud_variant:
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
