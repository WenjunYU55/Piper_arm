#!/usr/bin/env python3
"""Render deterministic diagnostic projections without opening a GUI window."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import open3d as o3d


BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def expected_box_corners(mesh, dimensions):
    obb = mesh.get_oriented_bounding_box(robust=True)
    half = np.asarray(dimensions, dtype=float) / 2.0
    local = np.asarray([
        [x, y, z]
        for x in (-half[0], half[0])
        for y in (-half[1], half[1])
        for z in (-half[2], half[2])
    ])
    return local @ np.asarray(obb.R).T + np.asarray(obb.center)


def render(report_path, output_path):
    report_path = Path(report_path).resolve()
    with open(report_path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    mesh = o3d.io.read_triangle_mesh(str(Path(report['mesh_path']).resolve()))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles):
        raise ValueError('reported mesh is empty')
    faces = vertices[triangles]
    vertex_colors = np.asarray(mesh.vertex_colors)
    face_colors = (
        np.clip(vertex_colors[triangles].mean(axis=1), 0.0, 1.0)
        if len(vertex_colors) == len(vertices)
        else np.tile([[0.2, 0.72, 0.35]], (len(triangles), 1)))
    dimension_check = report.get('mesh_metrics', {}).get('dimension_check') or {}
    expected = np.asarray(dimension_check.get('expected_m', []), dtype=float)
    corners = (
        expected_box_corners(mesh, expected)
        if expected.shape == (3,) else None)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    radius = max(0.025, float(np.max(vertices.max(axis=0) - vertices.min(axis=0))) / 2.0)
    views = ((25, -45, 'isometric'), (0, 0, 'side'), (90, -90, 'top'))
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = fig.add_subplot(1, 3, index, projection='3d')
        collection = Poly3DCollection(
            faces, facecolors=face_colors, edgecolors='none', alpha=1.0)
        axis.add_collection3d(collection)
        if corners is not None:
            for first, second in BOX_EDGES:
                edge = corners[[first, second]]
                axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='red', linewidth=1.2)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_xlabel('base X (m)')
        axis.set_ylabel('base Y (m)')
        axis.set_zlabel('base Z (m)')
    observed = dimension_check.get('observed_obb_m', [])
    fig.suptitle(
        '%s | %s | OBB %s mm (red: provisional 40 mm cube)' % (
            report.get('registration_mode', 'unknown'),
            report.get('overall_quality',
                       report.get('structural_quality', 'UNKNOWN')),
            ' x '.join('%.1f' % (1000.0 * float(value)) for value in observed)),
        fontsize=12)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial' + output.suffix)
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = Path(args.report).resolve()
    output = Path(args.output).resolve() if args.output else report.with_suffix('.preview.png')
    print(render(report, output))


if __name__ == '__main__':
    main()
