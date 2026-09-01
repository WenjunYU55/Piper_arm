"""Consensus-supported dense geometry and visibility-aware texture baking."""

from pathlib import Path

import cv2
import numpy as np


DEFAULT_ATLAS_TILE_PX = 16
MAXIMUM_ATLAS_SIZE_PX = 4096
MINIMUM_VIEW_COSINE = 0.15


def _finite_vertices_and_triangles(vertices, triangles):
    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1:] != (3,) \
            or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError('texture mesh vertices must be finite XYZ points')
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) \
            or np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError('texture mesh triangles are invalid')
    return points, faces


def _minimum_squared_distance(samples, references, chunk_size=2048):
    samples = np.asarray(samples, dtype=float)
    references = np.asarray(references, dtype=float)
    result = np.full(len(samples), np.inf, dtype=float)
    for start in range(0, len(samples), int(chunk_size)):
        chunk = samples[start:start + int(chunk_size)]
        squared = np.sum(
            np.square(chunk[:, None, :] - references[None, :, :]), axis=2)
        result[start:start + len(chunk)] = np.min(squared, axis=1)
    return result


def consensus_supported_triangles(
        vertices, triangles, consensus_points, support_radius_m):
    """Select dense mesh triangles close to multi-capture consensus."""
    points, faces = _finite_vertices_and_triangles(vertices, triangles)
    consensus = (
        np.empty((0, 3), dtype=float)
        if consensus_points is None else
        np.asarray(consensus_points, dtype=float))
    radius = float(support_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError('consensus support radius must be positive')
    if consensus.size == 0:
        return np.ones(len(faces), dtype=bool), {
            'available': False,
            'reason': 'cross-capture consensus is unavailable',
            'support_radius_m': radius,
            'input_triangle_count': int(len(faces)),
            'supported_triangle_count': int(len(faces)),
            'supported_triangle_fraction': 1.0,
            'single_capture_geometry_retained': True,
        }
    if consensus.ndim != 2 or consensus.shape[1:] != (3,) \
            or not np.all(np.isfinite(consensus)):
        raise ValueError('consensus points must be finite XYZ points')
    triangle_points = points[faces]
    samples = np.concatenate((
        np.mean(triangle_points, axis=1)[:, None, :], triangle_points), axis=1)
    distances = np.sqrt(_minimum_squared_distance(
        samples.reshape(-1, 3), consensus).reshape(len(faces), 4))
    # The centroid proves surface support; two nearby vertices retain small
    # triangles whose centroid lies just across a correspondence-cell edge.
    supported = (
        (distances[:, 0] <= radius)
        | (np.count_nonzero(distances[:, 1:] <= radius, axis=1) >= 2))
    count = int(np.count_nonzero(supported))
    return supported, {
        'available': True,
        'support_radius_m': radius,
        'input_consensus_point_count': int(len(consensus)),
        'input_triangle_count': int(len(faces)),
        'supported_triangle_count': count,
        'supported_triangle_fraction': float(count) / float(len(faces)),
        'single_capture_geometry_retained': False,
    }


def compact_mesh(vertices, triangles, keep_triangles):
    """Remove rejected faces and vertices not referenced by retained faces."""
    points, faces = _finite_vertices_and_triangles(vertices, triangles)
    keep = np.asarray(keep_triangles, dtype=bool)
    if keep.shape != (len(faces),):
        raise ValueError('texture triangle selection has the wrong shape')
    retained = faces[keep]
    if not len(retained):
        raise ValueError('consensus support retained no mesh triangles')
    used = np.unique(retained.reshape(-1))
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return points[used], remap[retained]


def _grid_triangles(index_grid, points, maximum_edge_m):
    """Triangulate valid adjacent depth pixels without crossing depth gaps."""
    grid = np.asarray(index_grid, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError('depth index grid must be two-dimensional')
    top_left = grid[:-1, :-1].reshape(-1)
    top_right = grid[:-1, 1:].reshape(-1)
    bottom_left = grid[1:, :-1].reshape(-1)
    bottom_right = grid[1:, 1:].reshape(-1)
    candidates = np.concatenate((
        np.column_stack((top_left, top_right, bottom_left)),
        np.column_stack((top_right, bottom_right, bottom_left))), axis=0)
    candidates = candidates[np.all(candidates >= 0, axis=1)]
    if not len(candidates):
        return np.empty((0, 3), dtype=np.int64)
    triangle_points = np.asarray(points, dtype=float)[candidates]
    edges = np.stack((
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 1],
        triangle_points[:, 0] - triangle_points[:, 2]), axis=1)
    keep = np.max(np.linalg.norm(edges, axis=2), axis=1) \
        <= float(maximum_edge_m)
    return candidates[keep]


def dense_superposition_mesh(
        frames, consensus_points, voxel_length_m):
    """Fuse every consensus-supported corrected measured depth pixel."""
    voxel = float(voxel_length_m)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError('dense measured-surface voxel length must be positive')
    if not frames:
        raise ValueError('dense measured surface requires a capture')
    consensus = (
        np.empty((0, 3), dtype=float)
        if consensus_points is None else
        np.asarray(consensus_points, dtype=float))
    if consensus.size and (
            consensus.ndim != 2 or consensus.shape[1:] != (3,)
            or not np.all(np.isfinite(consensus))):
        raise ValueError('dense measured surface consensus is malformed')
    support_radius = max(0.0045, 2.5 * voxel)
    fusion_voxel = min(0.0015, max(0.0005, 0.5 * voxel))
    maximum_edge = max(0.0045, 2.0 * voxel)
    view_points = []
    view_triangles = []
    input_points = 0
    retained_points = 0
    offset = 0
    per_capture = []
    for frame in frames:
        depth = np.asarray(frame['depth'])
        mask = np.asarray(frame['mask']) > 0
        intrinsic = np.asarray(frame['camera_matrix'], dtype=float)
        transform = np.asarray(frame['T_base_camera'], dtype=float)
        if depth.ndim != 2 or mask.shape != depth.shape \
                or intrinsic.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError('dense measured-surface frame is malformed')
        valid = mask & (depth > 0)
        rows, columns = np.nonzero(valid)
        z = depth[rows, columns].astype(float) / 1000.0
        camera = np.column_stack((
            (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z,
            np.ones(len(z), dtype=float)))
        base = (transform @ camera.T).T[:, :3]
        input_count = int(len(base))
        input_points += input_count
        if len(consensus):
            keep = _minimum_squared_distance(base, consensus) \
                <= support_radius * support_radius
        else:
            keep = np.ones(len(base), dtype=bool)
        base = base[keep]
        kept_rows = rows[keep]
        kept_columns = columns[keep]
        retained_count = int(len(base))
        retained_points += retained_count
        index_grid = np.full(depth.shape, -1, dtype=np.int64)
        index_grid[kept_rows, kept_columns] = np.arange(
            retained_count, dtype=np.int64)
        triangles = _grid_triangles(index_grid, base, maximum_edge)
        view_points.append(base)
        view_triangles.append(triangles + offset)
        offset += retained_count
        per_capture.append({
            'frame': str(frame.get('frame', '')),
            'input_measured_points': input_count,
            'consensus_supported_points': retained_count,
            'depth_grid_triangles': int(len(triangles)),
        })
    points = np.concatenate(view_points, axis=0)
    triangles = np.concatenate(view_triangles, axis=0)
    if not len(points) or not len(triangles):
        raise ValueError(
            'measured superposition produced no triangulated surface')
    keys = np.floor(points / fusion_voxel).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    fused = np.zeros((int(np.max(inverse)) + 1, 3), dtype=float)
    counts = np.bincount(inverse)
    np.add.at(fused, inverse, points)
    fused /= counts[:, None]
    triangles = inverse[triangles]
    triangles = triangles[
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])]
    if not len(triangles):
        raise ValueError('fine-voxel fusion removed every measured triangle')
    canonical = np.sort(triangles, axis=1)
    _unique_triangles, indices = np.unique(
        canonical, axis=0, return_index=True)
    triangles = triangles[np.sort(indices)]
    support_keep, support = consensus_supported_triangles(
        fused, triangles, consensus, support_radius)
    fused, triangles = compact_mesh(fused, triangles, support_keep)
    support.update({
        'source_geometry':
            'all_corrected_confidence_qualified_measured_depth_pixels',
        'input_measured_point_count': input_points,
        'consensus_supported_measured_point_count': retained_points,
        'fine_fusion_voxel_m': fusion_voxel,
        'maximum_depth_grid_edge_m': maximum_edge,
        'fused_vertex_count': int(len(fused)),
        'fused_triangle_count': int(len(triangles)),
        'per_capture': per_capture,
    })
    return fused, triangles, support


def _project(points_base, frame):
    transform = np.asarray(frame['T_base_camera'], dtype=float)
    intrinsic = np.asarray(frame['camera_matrix'], dtype=float)
    if transform.shape != (4, 4) or intrinsic.shape != (3, 3) \
            or not np.all(np.isfinite(transform)) \
            or not np.all(np.isfinite(intrinsic)):
        raise ValueError('texture frame camera model is malformed')
    homogeneous = np.column_stack((
        np.asarray(points_base, dtype=float),
        np.ones(len(points_base), dtype=float)))
    camera = (np.linalg.inv(transform) @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    pixels = np.full((len(camera), 2), np.nan, dtype=float)
    positive = depth > 1e-9
    pixels[positive, 0] = (
        intrinsic[0, 0] * camera[positive, 0] / depth[positive]
        + intrinsic[0, 2])
    pixels[positive, 1] = (
        intrinsic[1, 1] * camera[positive, 1] / depth[positive]
        + intrinsic[1, 2])
    return pixels, depth


def _nearest_observed_depth(frame, pixels, radius_px=2):
    depth_image = np.asarray(frame['depth'])
    mask = np.asarray(frame['mask']) > 0
    height, width = depth_image.shape[:2]
    observed = np.full(len(pixels), np.nan, dtype=float)
    finite = np.all(np.isfinite(pixels), axis=1)
    rounded = np.rint(np.where(np.isfinite(pixels), pixels, 0.0)).astype(int)
    offsets = sorted(
        ((x, y) for y in range(-radius_px, radius_px + 1)
         for x in range(-radius_px, radius_px + 1)),
        key=lambda value: value[0] * value[0] + value[1] * value[1])
    for x_offset, y_offset in offsets:
        unresolved = finite & ~np.isfinite(observed)
        if not np.any(unresolved):
            break
        indices = np.flatnonzero(unresolved)
        u = rounded[indices, 0] + x_offset
        v = rounded[indices, 1] + y_offset
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = indices[inside]
        u = u[inside]
        v = v[inside]
        valid = mask[v, u] & (depth_image[v, u] > 0)
        observed[indices[valid]] = (
            depth_image[v[valid], u[valid]].astype(float) / 1000.0)
    return observed


def _projected_triangle_area(pixels):
    first, second, third = np.asarray(pixels, dtype=float)
    first_edge = second - first
    second_edge = third - first
    return 0.5 * abs(float(
        first_edge[0] * second_edge[1]
        - first_edge[1] * second_edge[0]))


def triangle_camera_scores(
        vertices, triangles, frames, depth_tolerance_m):
    """Score depth-visible source frames for every mesh triangle."""
    points, faces = _finite_vertices_and_triangles(vertices, triangles)
    tolerance = float(depth_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('texture depth tolerance must be positive')
    if not frames:
        raise ValueError('texture baking requires at least one RGB-D frame')
    triangle_points = points[faces]
    centres = np.mean(triangle_points, axis=1)
    normals = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid_faces = lengths > 1e-12
    normals[valid_faces] /= lengths[valid_faces, None]
    scores = np.full((len(faces), len(frames)), -np.inf, dtype=float)
    for frame_index, frame in enumerate(frames):
        height, width = np.asarray(frame['depth']).shape[:2]
        samples = np.concatenate((centres[:, None, :], triangle_points), axis=1)
        pixels, camera_depth = _project(samples.reshape(-1, 3), frame)
        pixels = pixels.reshape(len(faces), 4, 2)
        camera_depth = camera_depth.reshape(len(faces), 4)
        observed = _nearest_observed_depth(
            frame, pixels.reshape(-1, 2)).reshape(len(faces), 4)
        inside = (
            np.all(np.isfinite(pixels), axis=(1, 2))
            & np.all(camera_depth > 0.0, axis=1)
            & np.all(pixels[:, :, 0] >= 1.0, axis=1)
            & np.all(pixels[:, :, 0] < width - 1.0, axis=1)
            & np.all(pixels[:, :, 1] >= 1.0, axis=1)
            & np.all(pixels[:, :, 1] < height - 1.0, axis=1)
            & np.all(np.isfinite(observed), axis=1)
            & np.all(np.abs(observed - camera_depth) <= tolerance, axis=1))
        camera_origin = np.asarray(frame['T_base_camera'], dtype=float)[:3, 3]
        view = camera_origin[None, :] - centres
        view_length = np.linalg.norm(view, axis=1)
        positive_view = view_length > 1e-12
        view[positive_view] /= view_length[positive_view, None]
        cosine = np.abs(np.sum(normals * view, axis=1))
        area = np.asarray([
            _projected_triangle_area(item[1:]) for item in pixels],
            dtype=float)
        centre_u = pixels[:, 0, 0]
        centre_v = pixels[:, 0, 1]
        radial = np.sqrt(
            np.square((centre_u - 0.5 * width) / max(1.0, 0.5 * width))
            + np.square((centre_v - 0.5 * height) / max(1.0, 0.5 * height)))
        consistency = np.mean(
            np.abs(observed - camera_depth), axis=1)
        admissible = (
            valid_faces & inside & positive_view
            & (cosine >= MINIMUM_VIEW_COSINE) & (area >= 0.05))
        scores[admissible, frame_index] = (
            cosine[admissible]
            * np.sqrt(area[admissible])
            / (1.0 + radial[admissible])
            / (1.0 + consistency[admissible] / tolerance))
    return scores


def _bilinear_rgb(image, pixels):
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    pixels = np.asarray(pixels, dtype=float)
    x = np.clip(pixels[:, 0], 0.0, width - 1.0)
    y = np.clip(pixels[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    values = (
        image[y0, x0].astype(float) * ((1.0 - wx) * (1.0 - wy))[:, None]
        + image[y0, x1].astype(float) * (wx * (1.0 - wy))[:, None]
        + image[y1, x0].astype(float) * ((1.0 - wx) * wy)[:, None]
        + image[y1, x1].astype(float) * (wx * wy)[:, None])
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _atlas_layout(triangle_count):
    columns = int(np.ceil(np.sqrt(int(triangle_count))))
    tile = min(
        DEFAULT_ATLAS_TILE_PX,
        MAXIMUM_ATLAS_SIZE_PX // max(1, columns))
    if tile < 4:
        raise ValueError('textured mesh has too many triangles for the atlas')
    rows = int(np.ceil(float(triangle_count) / float(columns)))
    return columns, rows, tile


def bake_texture_atlas(vertices, triangles, frames, depth_tolerance_m):
    """Bake one best-view RGB patch for each triangle into a UV atlas."""
    points, faces = _finite_vertices_and_triangles(vertices, triangles)
    scores = triangle_camera_scores(
        points, faces, frames, depth_tolerance_m)
    selected = np.argmax(scores, axis=1)
    selected_scores = scores[np.arange(len(faces)), selected]
    textured = np.isfinite(selected_scores)
    columns, rows, tile = _atlas_layout(len(faces))
    atlas = np.full((rows * tile, columns * tile, 3), 127, dtype=np.uint8)
    texture_coordinates = np.zeros((len(faces), 3, 2), dtype=float)
    padding = 1
    span = tile - 1 - 2 * padding
    barycentric_samples = []
    local_pixels = []
    for y in range(padding, tile - padding):
        for x in range(padding, tile - padding):
            first = float(x - padding) / float(span)
            second = float(y - padding) / float(span)
            if first + second <= 1.0 + 1e-12:
                barycentric_samples.append((1.0 - first - second, first, second))
                local_pixels.append((x, y))
    barycentric = np.asarray(barycentric_samples, dtype=float)
    for face_index, face in enumerate(faces):
        column = face_index % columns
        row = face_index // columns
        x0, y0 = column * tile, row * tile
        uv_pixels = np.asarray([
            [x0 + padding, y0 + padding],
            [x0 + tile - padding - 1, y0 + padding],
            [x0 + padding, y0 + tile - padding - 1],
        ], dtype=float)
        texture_coordinates[face_index, :, 0] = (
            uv_pixels[:, 0] + 0.5) / float(atlas.shape[1])
        texture_coordinates[face_index, :, 1] = 1.0 - (
            (uv_pixels[:, 1] + 0.5) / float(atlas.shape[0]))
        if not textured[face_index]:
            continue
        surface = barycentric @ points[face]
        source_pixels, depth = _project(
            surface, frames[int(selected[face_index])])
        valid = np.all(np.isfinite(source_pixels), axis=1) & (depth > 0.0)
        colors = np.full((len(surface), 3), 127, dtype=np.uint8)
        if np.any(valid):
            colors[valid] = _bilinear_rgb(
                frames[int(selected[face_index])]['rgb'],
                source_pixels[valid])
        for (x, y), color in zip(local_pixels, colors):
            atlas[y0 + y, x0 + x] = color
        # Fill the one-pixel gutter from the three texture vertices so linear
        # filtering cannot pull unrelated neighboring tiles into an edge.
        atlas[y0:y0 + tile, x0] = atlas[y0 + padding, x0 + padding]
        atlas[y0, x0:x0 + tile] = atlas[y0 + padding, x0 + padding]
        atlas[y0:y0 + tile, x0 + tile - 1] = \
            atlas[y0 + padding, x0 + tile - padding - 1]
        atlas[y0 + tile - 1, x0:x0 + tile] = \
            atlas[y0 + tile - padding - 1, x0 + padding]
    counts = np.bincount(selected[textured], minlength=len(frames))
    return atlas, texture_coordinates, {
        'triangle_count': int(len(faces)),
        'textured_triangle_count': int(np.count_nonzero(textured)),
        'untextured_triangle_count': int(np.count_nonzero(~textured)),
        'textured_triangle_fraction': float(np.mean(textured)),
        'selected_source_triangles_per_capture': counts.astype(int).tolist(),
        'atlas_width_px': int(atlas.shape[1]),
        'atlas_height_px': int(atlas.shape[0]),
        'atlas_tile_px': int(tile),
        'source_selection': (
            'best depth-consistent front-facing projected triangle'),
        'color_sampling': 'bilinear_from_rectified_source_RGB',
        'untextured_color_rgb': [127, 127, 127],
    }


def write_textured_obj(
        output_path, vertices, triangles, texture_coordinates, atlas):
    """Atomically write OBJ, MTL and PNG assets for one textured mesh."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    points, faces = _finite_vertices_and_triangles(vertices, triangles)
    uv = np.asarray(texture_coordinates, dtype=float)
    if uv.shape != (len(faces), 3, 2) or not np.all(np.isfinite(uv)):
        raise ValueError('texture coordinates are malformed')
    texture = output.with_name(output.stem + '.texture.png')
    material = output.with_suffix('.mtl')
    temporary_texture = texture.with_name(texture.stem + '.partial.png')
    if not cv2.imwrite(
            str(temporary_texture),
            cv2.cvtColor(np.asarray(atlas, dtype=np.uint8), cv2.COLOR_RGB2BGR)):
        raise RuntimeError('failed to write texture atlas')
    temporary_texture.replace(texture)
    material_text = '\n'.join((
        'newmtl target_texture',
        'Ka 1.000000 1.000000 1.000000',
        'Kd 1.000000 1.000000 1.000000',
        'Ks 0.000000 0.000000 0.000000',
        'd 1.0',
        'illum 1',
        'map_Kd %s' % texture.name,
        '',
    ))
    temporary_material = material.with_name(material.name + '.partial')
    temporary_material.write_text(material_text, encoding='utf-8')
    temporary_material.replace(material)
    lines = ['mtllib %s' % material.name, 'o target_mesh']
    lines.extend(
        'v %.12g %.12g %.12g' % tuple(point) for point in points)
    lines.extend(
        'vt %.12g %.12g' % tuple(coordinate)
        for face_uv in uv for coordinate in face_uv)
    lines.append('usemtl target_texture')
    for face_index, face in enumerate(faces):
        texture_start = face_index * 3 + 1
        lines.append('f %d/%d %d/%d %d/%d' % (
            int(face[0]) + 1, texture_start,
            int(face[1]) + 1, texture_start + 1,
            int(face[2]) + 1, texture_start + 2))
    lines.append('')
    temporary_output = output.with_name(output.name + '.partial')
    temporary_output.write_text('\n'.join(lines), encoding='utf-8')
    temporary_output.replace(output)
    return output, material, texture


def build_textured_mesh(
        output_path, frames, consensus_points, voxel_length_m):
    """Build a dense consensus-supported mesh with source-image textures."""
    voxel = float(voxel_length_m)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError('texture voxel length must be positive')
    points, faces, support = dense_superposition_mesh(
        frames, consensus_points, voxel)
    depth_tolerance = max(0.006, 2.5 * voxel)
    atlas, uv, diagnostics = bake_texture_atlas(
        points, faces, frames, depth_tolerance)
    mesh_path, material_path, texture_path = write_textured_obj(
        output_path, points, faces, uv, atlas)
    diagnostics.update({
        'geometry': support,
        'surface_method': 'measured_triangles',
        'vertex_count': int(len(points)),
        'depth_visibility_tolerance_m': depth_tolerance,
        'mesh_path': str(mesh_path),
        'material_path': str(material_path),
        'texture_path': str(texture_path),
        'format': 'Wavefront OBJ plus MTL plus PNG texture atlas',
        'source_depth_and_masks_immutable': True,
        'source_rgb_modified': False,
    })
    return diagnostics
