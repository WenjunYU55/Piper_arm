from pathlib import Path

import cv2
import numpy as np
import pytest

from reconstruction import texture_baking as TEXTURE


def simple_mesh():
    return (
        np.asarray([
            [-0.10, -0.10, 1.0],
            [0.10, -0.10, 1.0],
            [-0.10, 0.10, 1.0],
            [0.10, 0.10, 1.0],
        ]),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )


def camera_frame(color=(20, 180, 60), depth_mm=1000):
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:] = color
    return {
        'rgb': rgb,
        'depth': np.full((100, 100), depth_mm, dtype=np.uint16),
        'mask': np.full((100, 100), 255, dtype=np.uint8),
        'camera_matrix': np.asarray([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ]),
        'T_base_camera': np.eye(4),
        'frame': 'view_000_metadata.yaml',
    }


def dense_frame(color=(20, 180, 60)):
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:] = color
    return {
        'rgb': rgb,
        'depth': np.full((10, 10), 1000, dtype=np.uint16),
        'mask': np.full((10, 10), 255, dtype=np.uint8),
        'camera_matrix': np.asarray([
            [1000.0, 0.0, 4.5],
            [0.0, 1000.0, 4.5],
            [0.0, 0.0, 1.0],
        ]),
        'T_base_camera': np.eye(4),
        'frame': 'view_000_metadata.yaml',
    }


def test_consensus_support_keeps_nearby_dense_triangles_only():
    vertices, triangles = simple_mesh()
    consensus = np.asarray([[-0.03, -0.03, 1.0]])

    keep, diagnostics = TEXTURE.consensus_supported_triangles(
        vertices, triangles, consensus, support_radius_m=0.08)

    assert keep.tolist() == [True, False]
    assert diagnostics['available'] is True
    assert diagnostics['supported_triangle_count'] == 1


def test_missing_consensus_retains_single_capture_geometry():
    vertices, triangles = simple_mesh()

    keep, diagnostics = TEXTURE.consensus_supported_triangles(
        vertices, triangles, None, support_radius_m=0.01)

    assert keep.tolist() == [True, True]
    assert diagnostics['available'] is False
    assert diagnostics['single_capture_geometry_retained'] is True


def test_depth_visibility_selects_matching_camera_and_rejects_wrong_depth():
    vertices, triangles = simple_mesh()
    scores = TEXTURE.triangle_camera_scores(
        vertices, triangles,
        [camera_frame(), camera_frame(depth_mm=800)],
        depth_tolerance_m=0.01)

    assert np.all(np.isfinite(scores[:, 0]))
    assert np.all(np.isneginf(scores[:, 1]))


def test_texture_atlas_samples_source_rgb_without_cross_view_averaging():
    vertices, triangles = simple_mesh()
    atlas, uv, diagnostics = TEXTURE.bake_texture_atlas(
        vertices, triangles, [camera_frame()], depth_tolerance_m=0.01)

    assert uv.shape == (2, 3, 2)
    assert np.all((uv > 0.0) & (uv < 1.0))
    assert diagnostics['textured_triangle_count'] == 2
    assert diagnostics['selected_source_triangles_per_capture'] == [2]
    assert np.any(np.all(atlas == [20, 180, 60], axis=2))


def test_dense_superposition_mesh_uses_every_supported_depth_pixel():
    vertices, triangles, diagnostics = \
        TEXTURE.dense_superposition_mesh(
            [dense_frame()], None, voxel_length_m=0.003)

    assert diagnostics['input_measured_point_count'] == 100
    assert diagnostics['consensus_supported_measured_point_count'] == 100
    assert diagnostics['source_geometry'] == \
        'all_corrected_confidence_qualified_measured_depth_pixels'
    assert len(vertices) > 0
    assert len(triangles) > 0


def test_build_textured_mesh_writes_obj_mtl_and_png(tmp_path):
    output = tmp_path / 'target.textured.obj'

    diagnostics = TEXTURE.build_textured_mesh(
        output, [dense_frame()], None,
        voxel_length_m=0.003)

    material = Path(diagnostics['material_path'])
    texture = Path(diagnostics['texture_path'])
    assert output.is_file()
    assert material.is_file()
    assert texture.is_file()
    assert 'mtllib target.textured.mtl' in output.read_text(encoding='utf-8')
    assert 'map_Kd target.textured.texture.png' in material.read_text(
        encoding='utf-8')
    loaded = cv2.imread(str(texture), cv2.IMREAD_COLOR)
    assert loaded is not None and loaded.size > 0
    assert diagnostics['geometry']['single_capture_geometry_retained'] is True

    import open3d as o3d
    loaded_mesh = o3d.io.read_triangle_mesh(
        str(output), enable_post_processing=True)
    assert len(loaded_mesh.textures) == 1
    loaded_texture = np.asarray(loaded_mesh.textures[0])
    assert loaded_texture.ndim == 3
    assert loaded_texture.shape[2] == 3
    assert np.any(np.ptp(loaded_texture, axis=2) > 10)


def test_compact_mesh_rejects_empty_consensus_surface():
    vertices, triangles = simple_mesh()
    with pytest.raises(ValueError, match='retained no mesh triangles'):
        TEXTURE.compact_mesh(
            vertices, triangles, np.zeros(len(triangles), dtype=bool))
