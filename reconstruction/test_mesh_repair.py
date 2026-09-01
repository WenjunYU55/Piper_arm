import numpy as np
import pytest

from reconstruction.mesh_repair import (
    boundary_diagnostics, MEASURED_WALL_HOLE_RADIUS_M, repair_mesh)


o3d = pytest.importorskip('open3d')


def planar_wall_with_bounded_hole():
    """Return a 20 mm wall with a 4 mm internal opening."""
    count = 11
    coordinates = np.linspace(0.0, 0.020, count)
    vertices = np.asarray([
        (x, y, 0.0) for y in coordinates for x in coordinates],
        dtype=float)
    triangles = []
    for row in range(count - 1):
        for column in range(count - 1):
            if row in (4, 5) and column in (4, 5):
                continue
            lower_left = row * count + column
            lower_right = lower_left + 1
            upper_left = lower_left + count
            upper_right = upper_left + 1
            triangles.extend((
                (lower_left, lower_right, upper_right),
                (lower_left, upper_right, upper_left),
            ))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(triangles, dtype=np.int32))
    return mesh


def test_disabled_wall_repair_is_identity():
    mesh = planar_wall_with_bounded_hole()
    repaired, report = repair_mesh(o3d, mesh, 'none')
    assert repaired is mesh
    assert not report['applied']
    assert report['interpolated_triangle_count'] == 0
    assert report['before'] == report['after']


def test_bounded_wall_repair_fills_small_hole_but_retains_outer_boundary():
    mesh = planar_wall_with_bounded_hole()
    before = boundary_diagnostics(mesh)
    repaired, report = repair_mesh(o3d, mesh, 'measured_wall')
    after = boundary_diagnostics(repaired)

    assert before['boundary_component_count'] == 2
    assert after['boundary_component_count'] == 1
    assert report['interpolated_triangle_count'] > 0
    assert report['maximum_hole_radius_m'] == pytest.approx(
        MEASURED_WALL_HOLE_RADIUS_M)
    assert report['object_sized_open_boundaries_retained']
    assert after['maximum_boundary_component_diagonal_m'] == pytest.approx(
        before['maximum_boundary_component_diagonal_m'])


def test_unknown_wall_repair_mode_fails_closed():
    with pytest.raises(ValueError, match='hole repair mode'):
        repair_mesh(o3d, planar_wall_with_bounded_hole(), 'invented')
