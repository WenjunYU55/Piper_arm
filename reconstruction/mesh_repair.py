"""Conservative, explicitly labelled repair for bounded TSDF wall holes."""

from collections import defaultdict

import numpy as np


HOLE_REPAIR_MODES = ('none', 'measured_wall')
MEASURED_WALL_HOLE_RADIUS_M = 0.006


def boundary_diagnostics(mesh):
    """Describe open mesh-boundary components without modifying geometry."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not triangles.size:
        return {
            'boundary_edge_count': 0,
            'boundary_component_count': 0,
            'maximum_boundary_component_diagonal_m': 0.0,
        }
    edges = np.sort(np.concatenate((
        triangles[:, [0, 1]], triangles[:, [1, 2]],
        triangles[:, [2, 0]]), axis=0), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    adjacency = defaultdict(set)
    for first, second in boundary_edges:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    unseen = set(adjacency)
    diagonals = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor in component:
                    continue
                component.add(neighbor)
                unseen.discard(neighbor)
                pending.append(neighbor)
        component_vertices = vertices[list(component)]
        diagonal = np.linalg.norm(
            np.max(component_vertices, axis=0)
            - np.min(component_vertices, axis=0))
        diagonals.append(float(diagonal))
    return {
        'boundary_edge_count': int(len(boundary_edges)),
        'boundary_component_count': int(len(diagonals)),
        'maximum_boundary_component_diagonal_m': float(
            max(diagonals, default=0.0)),
    }


def repair_mesh(o3d, mesh, mode='none'):
    """Fill only small bounded wall holes; retain object-sized openings."""
    selected = str(mode)
    if selected not in HOLE_REPAIR_MODES:
        raise ValueError(
            'hole repair mode must be one of: %s'
            % ', '.join(HOLE_REPAIR_MODES))
    before = boundary_diagnostics(mesh)
    if selected == 'none':
        return mesh, {
            'mode': selected,
            'applied': False,
            'maximum_hole_radius_m': None,
            'interpolated_triangle_count': 0,
            'interpolated_vertex_count': 0,
            'before': before,
            'after': dict(before),
            'raw_tsdf_mesh_unchanged': True,
            'semantics': 'measured TSDF surface; no hole interpolation',
        }

    original_vertices = int(len(mesh.vertices))
    original_triangles = int(len(mesh.triangles))
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    repaired = tensor_mesh.fill_holes(
        MEASURED_WALL_HOLE_RADIUS_M).to_legacy()
    repaired.compute_vertex_normals()
    added_vertices = int(len(repaired.vertices)) - original_vertices
    added_triangles = int(len(repaired.triangles)) - original_triangles
    if added_vertices < 0 or added_triangles < 0:
        raise ValueError('wall-hole repair unexpectedly removed mesh geometry')
    after = boundary_diagnostics(repaired)
    largest_before = before['maximum_boundary_component_diagonal_m']
    largest_after = after['maximum_boundary_component_diagonal_m']
    object_sized_boundary_present = (
        largest_before > 2.0 * MEASURED_WALL_HOLE_RADIUS_M)
    if object_sized_boundary_present and largest_after < 0.8 * largest_before:
        raise ValueError(
            'wall-hole repair altered an object-sized open boundary')
    return repaired, {
        'mode': selected,
        'applied': True,
        'maximum_hole_radius_m': MEASURED_WALL_HOLE_RADIUS_M,
        'interpolated_triangle_count': added_triangles,
        'interpolated_vertex_count': added_vertices,
        'before': before,
        'after': after,
        'raw_tsdf_mesh_unchanged': True,
        'object_sized_open_boundaries_retained': bool(
            not object_sized_boundary_present
            or largest_after >= 0.8 * largest_before),
        'semantics': (
            'triangles inserted only across bounded TSDF wall openings no '
            'larger than the configured approximate radius; inserted '
            'triangles are interpolated rather than directly measured'),
    }
