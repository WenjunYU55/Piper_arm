#!/usr/bin/env python3
"""Interactive Open3D validation view for one generated quality report."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from coordinate_frames import arrow_transform, file_hash, report_frame, view_layout


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
    parser.add_argument('--show-cameras', action='store_true')
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
    orientation = None
    try:
        report_frame(source)
        asset = cloud_path if point_cloud_variant else mesh_path
        sidecar = asset.with_name(asset.name + '.frame.json')
        if sidecar.is_file():
            frame = json.loads(sidecar.read_text())
            if frame.get('geometry_sha256') != file_hash(asset) or frame.get('frame_id') != 'base_link':
                raise ValueError('Coordinate-frame sidecar does not match geometry')
        points = np.asarray(primary.points if point_cloud_variant else primary.vertices)
        orientation = view_layout(points)
        for index, (origin, vector, color) in enumerate(orientation['arrows']):
            length = np.linalg.norm(vector)
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=length * .025, cone_radius=length * .065,
                cylinder_height=length * .75, cone_height=length * .25)
            arrow.paint_uniform_color(color)
            arrow.transform(arrow_transform(origin, vector))
            geometries.append(arrow)
            if hasattr(o3d.t.geometry.TriangleMesh, 'create_text'):
                text = o3d.t.geometry.TriangleMesh.create_text(
                    ('X', 'Y', 'Z UP', 'ROBOT')[index], depth=.1).to_legacy()
                text.scale(orientation['span'] * .005, center=[0., 0., 0.])
                pose = orientation['camera_pose'].copy()
                pose[:3, 3] = origin + vector + pose[:3, 0] * orientation['span'] * .025
                text.transform(pose)
                text.paint_uniform_color(color)
                geometries.append(text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Base orientation unavailable: ' + str(exc), flush=True)
    wireframe = expected_wireframe(source, primary)
    if wireframe is not None:
        geometries.append(wireframe)
    if args.show_cameras and orientation:
        for frame in source.get('frame_inputs', []):
            geometries.append(camera_axes(frame['T_base_camera']))
    if args.show_input and not point_cloud_variant:
        cloud_path = Path(str(report.get(
            'measured_cloud_path', report.get(
                'input_cloud_path', '')))).resolve()
        if not cloud_path.is_file():
            raise SystemExit('reported masked input cloud is missing: %s' % cloud_path)
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        cloud.paint_uniform_color([0.1, 0.35, 1.0])
        geometries.append(cloud)
    view = (dict(front=orientation['front'], up=[0., 0., 1.],
                 lookat=orientation['center'], zoom=.65) if orientation else {})
    title = 'PiPER Reconstruction (%s) | %s' % (
        args.mesh, orientation['legend'] if orientation else 'ORIENTATION UNVERIFIED')
    print(title, flush=True)
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title, mesh_show_back_face=True, **view)


if __name__ == '__main__':
    main()
