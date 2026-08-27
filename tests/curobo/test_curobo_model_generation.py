"""Pure canonical-model conversion tests; CUDA is deliberately absent."""

import struct

import numpy as np
import pytest

from motion_planning.curobo.generate_robot_config import (
    covering_spheres,
    stl_vertices,
)


def test_ascii_stl_vertices_are_read_without_geometry_dependencies(tmp_path):
    path = tmp_path / 'triangle.stl'
    path.write_text(
        'solid test\n'
        ' facet normal 0 0 1\n'
        '  outer loop\n'
        '   vertex 0 0 0\n'
        '   vertex 1 0 0\n'
        '   vertex 0 1 0\n'
        '  endloop\n'
        ' endfacet\n'
        'endsolid test\n',
        encoding='ascii',
    )
    assert stl_vertices(path) == pytest.approx(np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))


def test_binary_stl_vertices_are_read_deterministically(tmp_path):
    path = tmp_path / 'triangle.stl'
    payload = bytearray(80)
    payload.extend(struct.pack('<I', 1))
    payload.extend(struct.pack('<fff', 0.0, 0.0, 1.0))
    for vertex in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        payload.extend(struct.pack('<fff', *vertex))
    payload.extend(struct.pack('<H', 0))
    path.write_bytes(payload)
    assert stl_vertices(path) == pytest.approx(np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))


def test_sphere_grid_conservatively_covers_a_collision_box():
    low = np.asarray([-0.04, -0.02, -0.01])
    high = np.asarray([0.04, 0.02, 0.01])
    spheres = covering_spheres(low, high, 0.04)
    assert len(spheres) == 2
    for corner in (low, high):
        assert any(
            np.linalg.norm(corner - np.asarray(sphere['center']))
            <= sphere['radius']
            for sphere in spheres)
