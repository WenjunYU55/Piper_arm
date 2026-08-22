#!/usr/bin/env python3
"""
Secure offline object-masked RGB-D reconstruction for completed PiPER scans.

TSDF is the surface-fusion method.  Sequential bounded GICP and bounded
multi-view pose-graph GICP are optional residual corrections around the
capture-time robot poses; neither replaces calibration or robot kinematics.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


MINIMUM_CAPTURE_VIEWS = 8
MINIMUM_PARTIAL_CAPTURE_VIEWS = 1
MAXIMUM_CAPTURE_VIEWS = 24
DEFAULT_VOXEL_LENGTH_M = 0.001
DEFAULT_SDF_TRUNC_M = 0.005
DEFAULT_DEPTH_TRUNC_M = 1.5
REGISTRATION_MODES = (
    'robot_pose', 'bounded_gicp', 'multiway_gicp', 'auto')
MULTIWAY_MAX_PRIOR_NEIGHBORS = 3
MULTIWAY_MAX_VIEW_ANGLE_DEG = 75.0
MULTIWAY_MAX_CORRESPONDENCE_M = 0.015


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


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


def capture_set_provenance(manifest, allow_partial_view_set=False):
    """Classify the immutable view set without confusing count with quality."""
    count = int(manifest.get('capture_count', 0))
    minimum = (
        MINIMUM_PARTIAL_CAPTURE_VIEWS
        if allow_partial_view_set else MINIMUM_CAPTURE_VIEWS)
    if not minimum <= count <= MAXIMUM_CAPTURE_VIEWS:
        raise ValueError(
            'TSDF reconstruction requires %d-%d captured views%s'
            % (
                minimum, MAXIMUM_CAPTURE_VIEWS,
                ' when partial view sets are explicitly allowed'
                if allow_partial_view_set else ''))
    partial = count < MINIMUM_CAPTURE_VIEWS
    return {
        'classification': (
            'PARTIAL_VIEW_SET' if partial else 'FEATURE_COMPLETE_ELIGIBLE'),
        'capture_count': count,
        'ordinary_feature_minimum': MINIMUM_CAPTURE_VIEWS,
        'partial_view_set_explicitly_allowed': bool(allow_partial_view_set),
        'reason': (
            'view count is below the ordinary feature-complete floor; '
            'mesh quality and coverage remain provisional'
            if partial else
            'view count is eligible for ordinary feature-complete validation'),
    }


def validate_capture_set(
        manifest, metadata_paths, allow_partial_view_set=False):
    """Validate one bounded feature-driven capture set before integration."""
    capture_set = capture_set_provenance(
        manifest, allow_partial_view_set=allow_partial_view_set)
    count = int(capture_set['capture_count'])
    paths = list(metadata_paths)
    if len(paths) != count:
        raise ValueError(
            'manifest capture_count %d does not match %d frame metadata files'
            % (count, len(paths)))
    return paths


def validate_manifest_integrity(scan, manifest):
    """Verify the canonical manifest digest and every immutable artifact."""
    if not isinstance(manifest, dict):
        raise ValueError('manifest is not an object')
    expected = str(manifest.get('manifest_sha256', ''))
    unsigned = dict(manifest)
    unsigned.pop('manifest_sha256', None)
    actual = canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError('manifest SHA-256 does not match its canonical payload')
    root = Path(scan).resolve()
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('manifest contains no immutable artifacts')
    seen = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise ValueError('manifest file %d is invalid' % index)
        relative = Path(str(record.get('path', '')))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('manifest file escapes the dataset root')
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError('manifest file escapes the dataset root') from exc
        key = relative.as_posix()
        if not key or key in seen:
            raise ValueError('manifest contains a duplicate or empty path')
        seen.add(key)
        if not candidate.is_file():
            raise ValueError('manifest artifact is missing: %s' % candidate)
        if int(record.get('bytes', -1)) != candidate.stat().st_size:
            raise ValueError('manifest artifact size changed: %s' % candidate)
        if str(record.get('sha256', '')) != sha256_file(candidate):
            raise ValueError('manifest artifact hash changed: %s' % candidate)
    return expected


def manifest_artifact_index(scan, manifest):
    root = Path(scan).resolve()
    index = {}
    for record in manifest.get('files', []):
        relative = Path(str(record.get('path', '')))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError('manifest file escapes the dataset root') from exc
        index[relative.as_posix()] = candidate
    return index


def metadata_paths_from_manifest(
        scan, manifest, allow_partial_view_set=False):
    artifacts = manifest_artifact_index(scan, manifest)
    paths = [
        path for relative, path in artifacts.items()
        if relative.startswith('frames/view_')
        and relative.endswith('_metadata.yaml')
    ]
    return validate_capture_set(
        manifest, sorted(paths),
        allow_partial_view_set=allow_partial_view_set)


def resolve_frame_artifacts(scan, metadata_path, metadata, manifest):
    """
    Resolve a frame only through manifest-listed dataset-relative paths.

    Historical metadata contains absolute paths.  Those strings are treated as
    provenance only; their basename must agree with the immutable frame name.
    """
    root = Path(scan).resolve()
    artifacts = manifest_artifact_index(root, manifest)
    suffix = '_metadata.yaml'
    if not metadata_path.name.endswith(suffix):
        raise ValueError('frame metadata filename is invalid')
    stem = metadata_path.name[:-len(suffix)]
    requested = {
        'rgb': ('rgb_file_path', 'frames/%s_rgb.png' % stem),
        'depth': ('depth_png_file_path', 'frames/%s_depth.png' % stem),
        'mask': ('mask_file_path', 'frames/%s_mask.png' % stem),
    }
    if int(metadata.get('capture_schema_version', 1)) >= 2:
        requested.update({
            'native_depth_npy': (
                'native_depth_file_path',
                'frames/%s_native_depth.npy' % stem),
            'native_depth': (
                'native_depth_png_file_path',
                'frames/%s_native_depth.png' % stem),
            'confidence': (
                'confidence_file_path',
                'frames/%s_confidence.png' % stem),
            'target_depth': (
                'target_depth_png_file_path',
                'frames/%s_target_depth.png' % stem),
            'target_support_mask': (
                'target_support_mask_file_path',
                'frames/%s_target_support_mask.png' % stem),
        })
    resolved = {}
    for name, (metadata_key, relative) in requested.items():
        recorded = Path(str(metadata.get(metadata_key, '')))
        if recorded.name != Path(relative).name:
            raise ValueError(
                '%s does not identify the manifest-listed %s artifact'
                % (metadata_path.name, name))
        if relative not in artifacts:
            raise ValueError(
                '%s is not listed in the immutable manifest' % relative)
        candidate = artifacts[relative]
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError('frame artifact escapes the dataset root') from exc
        resolved[name] = candidate
    return resolved


def confidence_capture_provenance(metadata, manifest, artifacts, cv2):
    """Validate schema-v2 confidence-qualified geometry or identify history."""
    schema = int(metadata.get('capture_schema_version', 1))
    if schema < 2:
        return {
            'mode': 'historical_aligned_depth_mask',
            'confidence_qualified': False,
        }
    required = {
        'native_depth_npy', 'native_depth', 'confidence', 'target_depth',
        'target_support_mask'}
    if not required.issubset(artifacts):
        raise ValueError('schema-v2 capture is missing confidence artifacts')
    synchronization = metadata.get('synchronization')
    quality = metadata.get('confidence_quality')
    policy = manifest.get('confidence_policy')
    if not isinstance(synchronization, dict) or not isinstance(quality, dict) \
            or not isinstance(policy, dict):
        raise ValueError('schema-v2 confidence provenance is incomplete')
    if synchronization.get('mask_rgb_exact') is not True:
        raise ValueError('schema-v2 mask is not exactly RGB-correlated')
    try:
        threshold = int(quality['confidence_threshold'])
        policy_threshold = int(policy['minimum_grade'])
        rgb_depth_delta = float(synchronization[
            'rgb_native_depth_delta_sec'])
        depth_confidence_delta = float(synchronization[
            'native_depth_confidence_delta_sec'])
        rgb_depth_limit = float(synchronization[
            'maximum_rgb_native_depth_delta_sec'])
        depth_confidence_limit = float(synchronization[
            'maximum_native_depth_confidence_delta_sec'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('schema-v2 confidence provenance is malformed') from exc
    if threshold != policy_threshold or threshold < 8 or threshold > 15:
        raise ValueError('schema-v2 confidence threshold is not qualified')
    if not all(np.isfinite(value) and value >= 0.0 for value in (
            rgb_depth_delta, depth_confidence_delta, rgb_depth_limit,
            depth_confidence_limit)):
        raise ValueError('schema-v2 synchronization bounds are invalid')
    if rgb_depth_delta > rgb_depth_limit + 1e-9 \
            or depth_confidence_delta > depth_confidence_limit + 1e-9:
        raise ValueError('schema-v2 capture exceeds synchronization bounds')
    confidence = cv2.imread(
        str(artifacts['confidence']), cv2.IMREAD_UNCHANGED)
    if confidence is None or confidence.ndim != 2 \
            or not np.issubdtype(confidence.dtype, np.integer) \
            or np.any(confidence < 0) or np.any(confidence > 15):
        raise ValueError('schema-v2 confidence artifact is invalid')
    try:
        native_array = np.load(
            artifacts['native_depth_npy'], allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('schema-v2 native depth array is invalid') from exc
    native_png = cv2.imread(
        str(artifacts['native_depth']), cv2.IMREAD_UNCHANGED)
    target_depth = cv2.imread(
        str(artifacts['target_depth']), cv2.IMREAD_UNCHANGED)
    target_support = cv2.imread(
        str(artifacts['target_support_mask']), cv2.IMREAD_UNCHANGED)
    if native_array.ndim != 2 or native_array.dtype != np.uint16 \
            or native_png is None or native_png.dtype != np.uint16 \
            or native_png.shape != native_array.shape \
            or not np.array_equal(native_png, native_array):
        raise ValueError('schema-v2 native depth artifacts are inconsistent')
    if confidence.shape != native_array.shape:
        raise ValueError('schema-v2 confidence dimensions do not match native depth')
    if target_depth is None or target_depth.ndim != 2 \
            or target_depth.dtype != np.uint16 \
            or target_support is None or target_support.ndim != 2 \
            or target_support.shape != target_depth.shape:
        raise ValueError('schema-v2 target depth artifacts are invalid')
    unique_support = set(np.unique(target_support).astype(int).tolist())
    if not unique_support.issubset({0, 255}) \
            or not np.array_equal(target_support > 0, target_depth > 0):
        raise ValueError('schema-v2 target depth/support artifacts disagree')
    try:
        projected_points = int(quality['projected_output_points'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('schema-v2 projected point count is malformed') from exc
    if projected_points != int(np.count_nonzero(target_support)) \
            or projected_points < 20:
        raise ValueError('schema-v2 projected target point count is inconsistent')
    synchronized_target = metadata.get('synchronized_target_3d')
    if metadata.get('target_valid') is not True \
            or not isinstance(synchronized_target, dict) \
            or synchronized_target.get('valid') is not True:
        raise ValueError('schema-v2 synchronized target geometry is not valid')
    return {
        'mode': 'l515_confidence_qualified_native_projection',
        'confidence_qualified': True,
        'minimum_confidence_grade': threshold,
        'rgb_native_depth_delta_sec': rgb_depth_delta,
        'native_depth_confidence_delta_sec': depth_confidence_delta,
        'confidence_quality': quality,
    }


def calibration_provenance(manifest, metadata_values, allow_missing=False):
    identifiers = [str(manifest.get('calibration_sha256', '')).strip().lower()]
    identifiers.extend(
        str(value.get('calibration_sha256', '')).strip().lower()
        for value in metadata_values)
    present = {value for value in identifiers if value}
    if any(value and len(value) != 64 for value in identifiers):
        raise ValueError('calibration SHA-256 is malformed')
    if len(present) > 1:
        raise ValueError('capture frames do not share one calibration identity')
    if not present:
        if not allow_missing:
            raise ValueError(
                'capture has no calibration identity; use '
                '--allow-missing-calibration-id for diagnostic-only replay')
        return {
            'classification': 'DIAGNOSTIC_ONLY',
            'calibration_sha256': '',
            'reason': 'capture predates calibration identity binding',
        }
    identity = next(iter(present))
    if any(value != identity for value in identifiers):
        raise ValueError('calibration identity is missing from part of the capture set')
    return {
        'classification': 'CERTIFIED',
        'calibration_sha256': identity,
        'reason': 'manifest and every frame bind one calibration identity',
    }


def capture_schema_provenance(
        manifest, metadata_values, provenance, allow_historical=False):
    """Bind certification to one complete confidence-qualified capture schema."""
    try:
        schemas = [int(value.get('capture_schema_version', 1))
                   for value in metadata_values]
    except (TypeError, ValueError) as exc:
        raise ValueError('capture schema version is malformed') from exc
    if not schemas:
        raise ValueError('capture has no frame metadata')
    if len(set(schemas)) != 1:
        raise ValueError('capture frames mix incompatible schema versions')
    schema = schemas[0]
    result = dict(provenance)
    if schema < 2:
        if not allow_historical:
            raise ValueError(
                'capture predates confidence-qualified schema; use '
                '--allow-missing-calibration-id for diagnostic-only replay')
        prior_reason = str(result.get('reason', '')).rstrip('; ')
        result.update({
            'classification': 'DIAGNOSTIC_ONLY',
            'capture_schema_version': schema,
            'confidence_qualified': False,
            'reason': (
                prior_reason +
                ('; ' if prior_reason else '') +
                'capture predates confidence-qualified depth filtering'),
        })
        return result
    try:
        manifest_schema = int(manifest.get('capture_schema_version', 0))
    except (TypeError, ValueError) as exc:
        raise ValueError('manifest capture schema version is malformed') from exc
    if schema != 2 or manifest_schema != schema:
        raise ValueError('manifest and frames do not bind capture schema 2')
    if not isinstance(manifest.get('confidence_policy'), dict):
        raise ValueError('manifest has no confidence-qualified capture policy')
    result.update({
        'capture_schema_version': schema,
        'confidence_qualified': True,
    })
    return result


def rotation_angle_rad(matrix):
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def correction_rejection(
        transformation, fitness, inlier_rmse,
        maximum_translation_m=0.020,
        maximum_rotation_deg=5.0,
        minimum_fitness=0.15,
        maximum_rmse_m=0.010):
    transform = np.asarray(transformation, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return 'registration correction is invalid'
    translation = float(np.linalg.norm(transform[:3, 3]))
    angle_deg = float(np.degrees(rotation_angle_rad(transform)))
    if not np.isfinite(fitness) or float(fitness) < float(minimum_fitness):
        return 'registration overlap is too weak'
    if not np.isfinite(inlier_rmse) or float(inlier_rmse) > float(maximum_rmse_m):
        return 'registration residual is too large'
    if translation > float(maximum_translation_m):
        return 'registration translation correction exceeds %.0fmm' % (
            1000.0 * float(maximum_translation_m))
    if angle_deg > float(maximum_rotation_deg):
        return 'registration rotation correction exceeds %.1fdeg' % float(
            maximum_rotation_deg)
    return ''


def camera_direction_angle_deg(first, second):
    """Return the angle between two finite camera optical axes."""
    first_axis = np.asarray(first, dtype=float)[:3, 2]
    second_axis = np.asarray(second, dtype=float)[:3, 2]
    first_norm = float(np.linalg.norm(first_axis))
    second_norm = float(np.linalg.norm(second_axis))
    if first_norm <= 0.0 or second_norm <= 0.0 \
            or not np.isfinite(first_norm) or not np.isfinite(second_norm):
        raise ValueError('camera pose has no finite optical axis')
    cosine = float(np.clip(
        np.dot(first_axis, second_axis) / (first_norm * second_norm),
        -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def multiway_registration_pairs(
        nominal_poses, maximum_prior_neighbors=MULTIWAY_MAX_PRIOR_NEIGHBORS,
        maximum_view_angle_deg=MULTIWAY_MAX_VIEW_ANGLE_DEG):
    """
    Select a bounded set of overlapping pair candidates.

    Consecutive captures are always evaluated.  Each capture also considers a
    few earlier views with the closest optical direction, avoiding an O(N^2)
    registration sweep while retaining loop closures across the scan.
    """
    poses = [np.asarray(value, dtype=float) for value in nominal_poses]
    if len(poses) < 2:
        return []
    if int(maximum_prior_neighbors) < 1:
        raise ValueError('maximum prior neighbors must be positive')
    pairs = {(index - 1, index) for index in range(1, len(poses))}
    for target in range(1, len(poses)):
        ranked = sorted(
            (camera_direction_angle_deg(poses[source], poses[target]), source)
            for source in range(target))
        for angle, source in ranked[:int(maximum_prior_neighbors)]:
            if angle <= float(maximum_view_angle_deg):
                pairs.add((source, target))
    return sorted(pairs)


def registration_spanning_tree(node_count, accepted_edges):
    """Return accepted edge indices forming one deterministic spanning tree."""
    count = int(node_count)
    if count < 1:
        raise ValueError('pose graph must contain at least one node')
    parent = list(range(count))

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    selected = set()
    ranked = sorted(
        enumerate(accepted_edges),
        key=lambda item: (
            abs(int(item[1]['target']) - int(item[1]['source'])),
            float(item[1]['inlier_rmse_m']), item[0]))
    for edge_index, edge in ranked:
        source = int(edge['source'])
        target = int(edge['target'])
        if source < 0 or target < 0 or source >= count or target >= count:
            raise ValueError('pose-graph edge references an invalid node')
        source_root, target_root = root(source), root(target)
        if source_root == target_root:
            continue
        parent[target_root] = source_root
        selected.add(edge_index)
        if len(selected) == count - 1:
            break
    if len(selected) != count - 1:
        raise ValueError(
            'bounded multi-view registration graph is disconnected')
    return selected


def bounded_multiway_camera_poses(frames, o3d):
    """Refine all camera poses jointly while retaining robot-pose bounds."""
    if len(frames) < 3:
        raise ValueError(
            'bounded multi-view registration requires at least three views')
    nominal = [
        np.asarray(frame['nominal_base_camera'], dtype=float)
        for frame in frames]
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in nominal:
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(pose))
    accepted_edges = []
    attempted_edges = []
    for source, target in multiway_registration_pairs(nominal):
        initial = np.linalg.inv(nominal[target]) @ nominal[source]
        result = o3d.pipelines.registration.registration_generalized_icp(
            frames[source]['cloud_camera'], frames[target]['cloud_camera'],
            MULTIWAY_MAX_CORRESPONDENCE_M, initial,
            o3d.pipelines.registration.
            TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6,
                max_iteration=40))
        relative = np.asarray(result.transformation, dtype=float)
        # Hold the target at its robot pose to express the pairwise proposal as
        # one base-frame correction of the source pose.  The established
        # correction bounds therefore retain their original physical meaning.
        correction = (
            nominal[target] @ relative @ np.linalg.inv(nominal[source]))
        rejection = correction_rejection(
            correction, result.fitness, result.inlier_rmse)
        record = {
            'source': int(source),
            'target': int(target),
            'fitness': float(result.fitness),
            'inlier_rmse_m': float(result.inlier_rmse),
            'proposed_translation_correction_m': float(
                np.linalg.norm(correction[:3, 3])),
            'proposed_rotation_correction_deg': float(
                np.degrees(rotation_angle_rad(correction))),
            'accepted': not rejection,
            'rejection': rejection,
        }
        attempted_edges.append(record)
        if rejection:
            continue
        information = o3d.pipelines.registration.\
            get_information_matrix_from_point_clouds(
                frames[source]['cloud_camera'],
                frames[target]['cloud_camera'],
                MULTIWAY_MAX_CORRESPONDENCE_M, relative)
        record['relative_transform'] = relative
        record['information'] = information
        accepted_edges.append(record)
    tree_edges = registration_spanning_tree(len(frames), accepted_edges)
    for edge_index, edge in enumerate(accepted_edges):
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            edge['source'], edge['target'], edge['relative_transform'],
            edge['information'], uncertain=edge_index not in tree_edges))
    options = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=MULTIWAY_MAX_CORRESPONDENCE_M,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.0,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.
        GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        options)
    refined = []
    corrections = []
    for index, (node, robot_pose) in enumerate(zip(graph.nodes, nominal)):
        pose = np.asarray(node.pose, dtype=float)
        correction = pose @ np.linalg.inv(robot_pose)
        rejection = correction_rejection(
            correction, fitness=1.0, inlier_rmse=0.0)
        if rejection:
            raise ValueError(
                'multi-view pose %d rejected: %s' % (index, rejection))
        refined.append(pose)
        corrections.append({
            'frame_index': int(index),
            'translation_correction_m': float(
                np.linalg.norm(correction[:3, 3])),
            'rotation_correction_deg': float(
                np.degrees(rotation_angle_rad(correction))),
        })
    serializable_edges = []
    for edge in attempted_edges:
        serializable_edges.append({
            key: value for key, value in edge.items()
            if key not in ('relative_transform', 'information')})
    return refined, {
        'candidate_pair_count': len(attempted_edges),
        'accepted_pair_count': len(accepted_edges),
        'spanning_tree_pair_count': len(tree_edges),
        'pairwise_edges': serializable_edges,
        'pose_corrections': corrections,
        'robot_pose_bounds_retained': {
            'maximum_translation_m': 0.020,
            'maximum_rotation_deg': 5.0,
        },
    }


def camera_matrix(camera_info):
    k = np.asarray(camera_info.get('k', []), dtype=float)
    if not camera_info.get('available') or k.size != 9 \
            or not np.all(np.isfinite(k)):
        raise ValueError('frame has no finite camera intrinsics')
    width = int(camera_info.get('width', 0))
    height = int(camera_info.get('height', 0))
    if width <= 0 or height <= 0:
        raise ValueError('camera dimensions are invalid')
    return k.reshape(3, 3), width, height


def rectify_rgbd_mask(cv2, rgb_bgr, depth_mm, mask, camera_info):
    """Undistort aligned RGB, depth and mask into one pinhole image plane."""
    k, width, height = camera_matrix(camera_info)
    for name, image in (
            ('RGB', rgb_bgr), ('depth', depth_mm), ('mask', mask)):
        if image is None or tuple(image.shape[:2]) != (height, width):
            raise ValueError('%s dimensions do not match CameraInfo' % name)
    model = str(camera_info.get('distortion_model', '')).strip()
    distortion = np.asarray(camera_info.get('d', []), dtype=float)
    if distortion.size and not np.all(np.isfinite(distortion)):
        raise ValueError('camera distortion coefficients are invalid')
    if model not in ('', 'plumb_bob', 'rational_polynomial'):
        raise ValueError('unsupported camera distortion model: %s' % model)
    if distortion.size and np.any(np.abs(distortion) > 1e-12):
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            k, distortion, (width, height), 0.0, (width, height),
            centerPrincipalPoint=False)
        map_x, map_y = cv2.initUndistortRectifyMap(
            k, distortion, None, new_k, (width, height), cv2.CV_32FC1)
        rgb_bgr = cv2.remap(
            rgb_bgr, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth_mm = cv2.remap(
            depth_mm, map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.remap(
            mask, map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        k = np.asarray(new_k, dtype=float)
        rectified = True
    else:
        rectified = False
    return rgb_bgr, depth_mm, mask, k, rectified


def masked_camera_cloud(o3d, depth_mm, mask, intrinsic, depth_trunc=1.5):
    values = np.asarray(depth_mm, dtype=np.uint16).copy()
    values[np.asarray(mask) <= 0] = 0
    image = o3d.geometry.Image(values)
    cloud = o3d.geometry.PointCloud.create_from_depth_image(
        image, intrinsic, depth_scale=1000.0,
        depth_trunc=float(depth_trunc), stride=1,
        project_valid_depth_only=True)
    cloud = cloud.voxel_down_sample(DEFAULT_VOXEL_LENGTH_M)
    if len(cloud.points) >= 20:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    return cloud


def expected_dimensions_m(values):
    if values is None:
        return None
    dimensions = np.asarray(values, dtype=float)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) \
            or np.any(dimensions <= 0.0) or np.any(dimensions > 2.0):
        raise ValueError('expected dimensions must be three finite values in metres')
    return dimensions


def target_depth_support_mask(depth_mm, mask, target_depth_m,
                              expected_dimensions):
    """
    Reject masked background beyond the known target's depth envelope.

    Segmentation boundaries on aligned RGB-D can contain table pixels.  For a
    measured reference object, its maximum line-of-sight half-extent cannot
    exceed half the 3-D diagonal; 5 mm is retained for depth/calibration noise.
    """
    binary = np.asarray(mask) > 0
    depth_m = np.asarray(depth_mm, dtype=float) / 1000.0
    target = float(target_depth_m)
    if expected_dimensions is None:
        return binary, None
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError('frame has no finite target depth for dimension gating')
    half_span = 0.5 * float(np.linalg.norm(expected_dimensions)) + 0.005
    supported = binary & (depth_m > 0.0) \
        & (np.abs(depth_m - target) <= half_span)
    return supported, {
        'target_depth_m': target,
        'maximum_depth_delta_m': half_span,
        'masked_points_before': int(np.count_nonzero(binary & (depth_m > 0.0))),
        'masked_points_after': int(np.count_nonzero(supported)),
    }


def quality_classification(median_registration_rmse_m,
                           dominant_component_ratio, usable_mesh=True):
    if not usable_mesh or not np.isfinite(median_registration_rmse_m):
        return 'FAIL'
    if median_registration_rmse_m > 0.010 \
            or dominant_component_ratio < 0.80:
        return 'FAIL'
    if median_registration_rmse_m > 0.005 \
            or dominant_component_ratio < 0.95:
        return 'WARN'
    return 'PASS'


def dimension_classification(maximum_absolute_error_m):
    error = float(maximum_absolute_error_m)
    if not np.isfinite(error) or error > 0.010:
        return 'POOR'
    if error > 0.005:
        return 'WARN'
    return 'GOOD'


def mesh_metrics(o3d, mesh, input_cloud, expected_dimensions):
    triangle_count = len(mesh.triangles)
    try:
        _, component_counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(component_counts, dtype=int)
    except RuntimeError:
        counts = np.asarray([], dtype=int)
    dominant_ratio = (
        float(np.max(counts)) / float(triangle_count)
        if triangle_count and counts.size else 0.0)
    obb = mesh.get_oriented_bounding_box(robust=True)
    extents = np.sort(np.asarray(obb.extent, dtype=float))[::-1]
    dimensions = None
    if expected_dimensions is not None:
        expected_sorted = np.sort(np.asarray(expected_dimensions, dtype=float))[::-1]
        errors = np.abs(extents - expected_sorted)
        dimensions = {
            'expected_m': expected_sorted.tolist(),
            'observed_obb_m': extents.tolist(),
            'absolute_error_m': errors.tolist(),
            'mean_absolute_error_m': float(np.mean(errors)),
            'maximum_absolute_error_m': float(np.max(errors)),
            'provisional_reference': True,
        }
    points = np.asarray(input_cloud.points, dtype=np.float32)
    distances = np.asarray([], dtype=float)
    if points.size:
        try:
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            distances = scene.compute_distance(
                o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)).numpy()
        except (AttributeError, RuntimeError):
            sampled = mesh.sample_points_uniformly(
                number_of_points=max(10000, min(100000, len(points) * 2)))
            distances = np.asarray(input_cloud.compute_point_cloud_distance(sampled))
    finite = distances[np.isfinite(distances)]
    residual = {
        'sample_count': int(finite.size),
        'median_m': float(np.median(finite)) if finite.size else float('inf'),
        'p90_m': float(np.percentile(finite, 90)) if finite.size else float('inf'),
    }
    return {
        'connected_component_count': int(counts.size),
        'dominant_component_triangle_ratio': dominant_ratio,
        'oriented_bounding_box_extents_m': extents.tolist(),
        'is_watertight': bool(mesh.is_watertight()),
        'is_edge_manifold': bool(mesh.is_edge_manifold()),
        'is_vertex_manifold': bool(mesh.is_vertex_manifold()),
        'is_self_intersecting': bool(mesh.is_self_intersecting()),
        'point_to_mesh_residual': residual,
        'dimension_check': dimensions,
    }


def _load_capture(scan, metadata_path, metadata, manifest, cv2, o3d,
                  depth_trunc, expected_dimensions=None):
    artifacts = resolve_frame_artifacts(scan, metadata_path, metadata, manifest)
    rgb_bgr = cv2.imread(str(artifacts['rgb']), cv2.IMREAD_COLOR)
    input_provenance = confidence_capture_provenance(
        metadata, manifest, artifacts, cv2)
    if input_provenance['confidence_qualified']:
        depth = cv2.imread(
            str(artifacts['target_depth']), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(
            str(artifacts['target_support_mask']), cv2.IMREAD_GRAYSCALE)
        target = metadata.get('synchronized_target_3d', {})
    else:
        depth = cv2.imread(str(artifacts['depth']), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(artifacts['mask']), cv2.IMREAD_GRAYSCALE)
        target = metadata.get('target_3d', {})
    rgb_bgr, depth, mask, k, rectified = rectify_rgbd_mask(
        cv2, rgb_bgr, depth, mask, metadata.get('camera_info', {}))
    height, width = depth.shape[:2]
    valid, depth_gate = target_depth_support_mask(
        depth, mask, target.get('depth'), expected_dimensions)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 20:
        raise ValueError('%s has too few masked depth points' % metadata_path.name)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    depth = np.asarray(depth, dtype=np.uint16)
    depth[~valid] = 0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(width), int(height), float(k[0, 0]), float(k[1, 1]),
        float(k[0, 2]), float(k[1, 2]))
    nominal_base_camera = np.linalg.inv(camera_extrinsic_from_metadata(metadata))
    cloud_camera = masked_camera_cloud(
        o3d, depth, mask, intrinsic, depth_trunc=depth_trunc)
    return {
        'metadata_path': metadata_path,
        'metadata': metadata,
        'rgb': np.ascontiguousarray(rgb),
        'depth': np.ascontiguousarray(depth),
        'mask': np.ascontiguousarray(mask),
        'intrinsic': intrinsic,
        'nominal_base_camera': nominal_base_camera,
        'cloud_camera': cloud_camera,
        'valid_masked_depth_points': valid_count,
        'rectified': rectified,
        'depth_gate': depth_gate,
        'input_provenance': input_provenance,
        'width': width,
        'height': height,
    }


def _atomic_write_mesh(o3d, mesh, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial' + output.suffix)
    if not o3d.io.write_triangle_mesh(str(temporary), mesh):
        raise OSError('Open3D failed to save %s' % temporary)
    temporary.replace(output)
    return output


def _atomic_write_cloud(o3d, cloud, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial' + output.suffix)
    if not o3d.io.write_point_cloud(str(temporary), cloud):
        raise OSError('Open3D failed to save %s' % temporary)
    temporary.replace(output)
    return output


def _atomic_write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def _reconstruct_single(scan, manifest, metadata_paths, metadata_values,
                        output_path, registration_mode, voxel_length,
                        sdf_trunc, depth_trunc, expected_dimensions,
                        provenance, manifest_sha256,
                        allow_partial_view_set=False):
    try:
        import cv2
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            'install reconstruction/requirements.txt in the isolated environment') from exc
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length), sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    frames = [
        _load_capture(
            scan, metadata_path, metadata, manifest, cv2, o3d, depth_trunc,
            expected_dimensions)
        for metadata_path, metadata in zip(metadata_paths, metadata_values)]
    multiway_diagnostics = None
    refined_multiway_poses = None
    if registration_mode == 'multiway_gicp':
        refined_multiway_poses, multiway_diagnostics = \
            bounded_multiway_camera_poses(frames, o3d)
    registration = []
    frame_inputs = []
    accumulated = None
    for frame_index, (metadata_path, frame) in enumerate(
            zip(metadata_paths, frames)):
        cloud_base = None
        correction = np.eye(4)
        accepted = False
        rejection = 'first frame anchors the target model'
        fitness = 1.0
        inlier_rmse = 0.0
        proposed_translation = 0.0
        proposed_rotation = 0.0
        if refined_multiway_poses is not None:
            refined_base_camera = refined_multiway_poses[frame_index]
            correction = (
                refined_base_camera
                @ np.linalg.inv(frame['nominal_base_camera']))
            proposed_translation = float(np.linalg.norm(correction[:3, 3]))
            proposed_rotation = float(
                np.degrees(rotation_angle_rad(correction)))
            accepted = frame_index != 0 and (
                proposed_translation > 1e-9 or proposed_rotation > 1e-9)
            rejection = (
                'first frame anchors bounded multi-view registration'
                if frame_index == 0 else '')
            incident = [
                edge for edge in multiway_diagnostics['pairwise_edges']
                if edge['accepted'] and frame_index in (
                    edge['source'], edge['target'])]
            if incident:
                fitness = float(min(edge['fitness'] for edge in incident))
                inlier_rmse = float(np.median([
                    edge['inlier_rmse_m'] for edge in incident]))
        elif accumulated is not None and len(accumulated.points) >= 20:
            cloud_base = copy.deepcopy(frame['cloud_camera'])
            cloud_base.transform(frame['nominal_base_camera'].copy())
            result = o3d.pipelines.registration.registration_generalized_icp(
                cloud_base, accumulated, 0.015, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6,
                    max_iteration=40))
            fitness = float(result.fitness)
            inlier_rmse = float(result.inlier_rmse)
            proposed = np.asarray(result.transformation, dtype=float)
            proposed_translation = float(np.linalg.norm(proposed[:3, 3]))
            proposed_rotation = float(np.degrees(rotation_angle_rad(proposed)))
            rejection = correction_rejection(proposed, fitness, inlier_rmse)
            if registration_mode == 'robot_pose':
                rejection = (
                    'robot_pose mode records but does not apply local GICP'
                    if not rejection else rejection)
            elif not rejection:
                correction = proposed
                cloud_base.transform(correction)
                accepted = True
            refined_base_camera = correction @ frame['nominal_base_camera']
        else:
            refined_base_camera = frame['nominal_base_camera'].copy()
        if cloud_base is None:
            cloud_base = copy.deepcopy(frame['cloud_camera'])
            cloud_base.transform(refined_base_camera.copy())
        registration.append({
            'frame': metadata_path.name,
            'fitness': fitness,
            'inlier_rmse_m': inlier_rmse,
            'correction_accepted': accepted,
            'correction_rejection': rejection,
            'proposed_translation_correction_m': proposed_translation,
            'proposed_rotation_correction_deg': proposed_rotation,
            'applied_translation_correction_m': float(
                np.linalg.norm(correction[:3, 3])),
            'applied_rotation_correction_deg': float(
                np.degrees(rotation_angle_rad(correction))),
            'rectified': bool(frame['rectified']),
            'valid_masked_depth_points': frame['valid_masked_depth_points'],
            'target_depth_gate': frame['depth_gate'],
            'capture_input_provenance': frame['input_provenance'],
        })
        accumulated = (
            cloud_base if accumulated is None else
            (accumulated + cloud_base).voxel_down_sample(DEFAULT_VOXEL_LENGTH_M))
        frame_inputs.append({
            'frame': metadata_path.name,
            'T_base_camera': refined_base_camera.tolist(),
            'width': frame['width'], 'height': frame['height'],
        })
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame['rgb']), o3d.geometry.Image(frame['depth']),
            depth_scale=1000.0, depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd, frame['intrinsic'], np.linalg.inv(refined_base_camera))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if len(mesh.vertices) < 100 or len(mesh.triangles) < 100:
        raise ValueError(
            'reconstruction produced an unusably small mesh '
            '(%d vertices, %d triangles)'
            % (len(mesh.vertices), len(mesh.triangles)))
    output = _atomic_write_mesh(o3d, mesh, output_path)
    input_cloud_path = _atomic_write_cloud(
        o3d, accumulated,
        output.with_name(output.stem + '.input_cloud.ply'))
    metrics = mesh_metrics(
        o3d, mesh, accumulated, expected_dimensions)
    if multiway_diagnostics is not None:
        registration_rmse = np.asarray([
            item['inlier_rmse_m']
            for item in multiway_diagnostics['pairwise_edges']
            if item['accepted'] and np.isfinite(item['inlier_rmse_m'])],
            dtype=float)
        registration_fitness = [
            item['fitness']
            for item in multiway_diagnostics['pairwise_edges']
            if item['accepted']]
        evaluated_pairs = int(
            multiway_diagnostics['candidate_pair_count'])
    else:
        registration_rmse = np.asarray(
            [item['inlier_rmse_m'] for item in registration[1:]
             if np.isfinite(item['inlier_rmse_m'])], dtype=float)
        registration_fitness = [
            item['fitness'] for item in registration[1:]]
        evaluated_pairs = max(0, len(registration) - 1)
    median_rmse = (
        float(np.median(registration_rmse))
        if registration_rmse.size else float('inf'))
    structural = quality_classification(
        median_rmse, metrics['dominant_component_triangle_ratio'])
    dimension_quality = None
    overall_quality = structural
    if metrics['dimension_check'] is not None:
        dimension_quality = dimension_classification(
            metrics['dimension_check']['maximum_absolute_error_m'])
        if dimension_quality == 'POOR':
            overall_quality = 'FAIL'
        elif dimension_quality == 'WARN' and overall_quality == 'PASS':
            overall_quality = 'WARN'
    capture_set = capture_set_provenance(
        manifest, allow_partial_view_set=allow_partial_view_set)
    if (
            capture_set['classification'] == 'PARTIAL_VIEW_SET'
            and overall_quality == 'PASS'):
        overall_quality = 'WARN'
    config = {
        'registration_mode': registration_mode,
        'voxel_length_m': float(voxel_length),
        'sdf_trunc_m': float(sdf_trunc),
        'depth_trunc_m': float(depth_trunc),
        'rectification': 'CameraInfo distortion to pinhole, alpha=0',
        'expected_dimensions_m': (
            expected_dimensions.tolist()
            if expected_dimensions is not None else None),
        'allow_partial_view_set': bool(allow_partial_view_set),
    }
    if multiway_diagnostics is not None:
        config['multiway_registration'] = {
            'maximum_prior_neighbors': MULTIWAY_MAX_PRIOR_NEIGHBORS,
            'maximum_view_angle_deg': MULTIWAY_MAX_VIEW_ANGLE_DEG,
            'maximum_correspondence_m': MULTIWAY_MAX_CORRESPONDENCE_M,
            'maximum_pose_translation_correction_m': 0.020,
            'maximum_pose_rotation_correction_deg': 5.0,
        }
    report = {
        'schema_version': 3,
        'scan_dir': str(scan),
        'input_manifest_sha256': manifest_sha256,
        'provenance': provenance,
        'capture_set': capture_set,
        'integrated_views': len(frame_inputs),
        'registration_mode': registration_mode,
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.triangles),
        'mesh_path': str(output),
        'mesh_sha256': sha256_file(output),
        'input_cloud_path': str(input_cloud_path),
        'input_cloud_sha256': sha256_file(input_cloud_path),
        'configuration': config,
        'configuration_sha256': canonical_sha256(config),
        'registration': registration,
        'multiway_registration': multiway_diagnostics,
        'registration_summary': {
            'accepted_corrections': sum(
                bool(item['correction_accepted']) for item in registration),
            'evaluated_pairs': evaluated_pairs,
            'median_rmse_m': median_rmse,
            'maximum_rmse_m': (
                float(np.max(registration_rmse))
                if registration_rmse.size else float('inf')),
            'minimum_fitness': (
                float(min(registration_fitness))
                if registration_fitness else 0.0),
        },
        'mesh_metrics': metrics,
        'structural_quality': structural,
        'dimension_quality': dimension_quality,
        'overall_quality': overall_quality,
        'overall_quality_is_provisional': bool(
            metrics['dimension_check'] is not None),
        'visual_review_required': True,
        'frame_inputs': frame_inputs,
    }
    _atomic_write_json(output.with_suffix(output.suffix + '.quality.json'), report)
    return report


def _quality_rank(value):
    return {'FAIL': 0, 'WARN': 1, 'PASS': 2}.get(str(value), -1)


def select_registration_report(robot_report, gicp_report):
    robot_residual = float(robot_report['mesh_metrics'][
        'point_to_mesh_residual']['median_m'])
    gicp_residual = float(gicp_report['mesh_metrics'][
        'point_to_mesh_residual']['median_m'])
    residual_improvement = (
        (robot_residual - gicp_residual) / robot_residual
        if np.isfinite(robot_residual) and robot_residual > 0.0 else 0.0)
    robot_dimension = robot_report['mesh_metrics'].get('dimension_check')
    gicp_dimension = gicp_report['mesh_metrics'].get('dimension_check')
    dimension_ok = True
    if robot_dimension and gicp_dimension:
        dimension_ok = (
            float(gicp_dimension['mean_absolute_error_m'])
            <= float(robot_dimension['mean_absolute_error_m']) + 0.002)
    robot_component = float(robot_report['mesh_metrics'][
        'dominant_component_triangle_ratio'])
    gicp_component = float(gicp_report['mesh_metrics'][
        'dominant_component_triangle_ratio'])
    component_ok = (
        gicp_component >= 0.95 and gicp_component >= robot_component - 0.01)
    quality_ok = _quality_rank(gicp_report['structural_quality']) >= _quality_rank(
        robot_report['structural_quality'])
    choose_gicp = (
        residual_improvement >= 0.10 and dimension_ok
        and component_ok and quality_ok)
    return ('bounded_gicp' if choose_gicp else 'robot_pose'), {
        'robot_pose_point_to_mesh_median_m': robot_residual,
        'bounded_gicp_point_to_mesh_median_m': gicp_residual,
        'bounded_gicp_residual_improvement_fraction': residual_improvement,
        'dimension_not_worse_by_more_than_2mm': bool(dimension_ok),
        'component_coherence_acceptable': bool(component_ok),
        'structural_quality_not_worse': bool(quality_ok),
        'selection_rule': (
            'bounded GICP requires >=10% residual improvement, <=2mm '
            'dimension regression, coherent components, and nonworse quality'),
    }


def registration_candidate_assessment(robot_report, candidate_report):
    """Decide whether one residual refinement safely improves the baseline."""
    robot_metrics = robot_report['mesh_metrics']
    candidate_metrics = candidate_report['mesh_metrics']
    robot_residual = float(
        robot_metrics['point_to_mesh_residual']['median_m'])
    candidate_residual = float(
        candidate_metrics['point_to_mesh_residual']['median_m'])
    residual_improvement = (
        (robot_residual - candidate_residual) / robot_residual
        if np.isfinite(robot_residual) and robot_residual > 0.0 else 0.0)
    robot_component = float(
        robot_metrics['dominant_component_triangle_ratio'])
    candidate_component = float(
        candidate_metrics['dominant_component_triangle_ratio'])
    component_improvement = candidate_component - robot_component
    robot_dimension = robot_metrics.get('dimension_check')
    candidate_dimension = candidate_metrics.get('dimension_check')
    dimension_ok = True
    if robot_dimension and candidate_dimension:
        dimension_ok = (
            float(candidate_dimension['mean_absolute_error_m'])
            <= float(robot_dimension['mean_absolute_error_m']) + 0.002)
    component_ok = (
        candidate_component >= 0.80
        and (component_improvement >= 0.05 or candidate_component >= 0.95))
    quality_ok = _quality_rank(
        candidate_report['structural_quality']) >= _quality_rank(
            robot_report['structural_quality'])
    material_improvement = (
        residual_improvement >= 0.10 or component_improvement >= 0.10)
    eligible = bool(
        dimension_ok and component_ok and quality_ok and material_improvement)
    return {
        'eligible': eligible,
        'robot_pose_point_to_mesh_median_m': robot_residual,
        'candidate_point_to_mesh_median_m': candidate_residual,
        'residual_improvement_fraction': residual_improvement,
        'robot_pose_dominant_component_ratio': robot_component,
        'candidate_dominant_component_ratio': candidate_component,
        'component_improvement_fraction': component_improvement,
        'dimension_not_worse_by_more_than_2mm': bool(dimension_ok),
        'component_coherence_acceptable': bool(component_ok),
        'structural_quality_not_worse': bool(quality_ok),
        'material_improvement': bool(material_improvement),
        'selection_rule': (
            'refinement must retain dimension and structural quality, reach '
            'at least 80% dominant-component coherence with material '
            'improvement, and improve residual or coherence materially'),
    }


def select_registration_reports(reports):
    """Select the strongest safe refinement while retaining robot fallback."""
    if 'robot_pose' not in reports:
        selected = next(iter(reports))
        return selected, {
            'selection_rule': 'robot-pose baseline did not complete',
            'candidate_assessments': {},
        }
    robot = reports['robot_pose']
    assessments = {}
    eligible = []
    for mode, report in reports.items():
        if mode == 'robot_pose':
            continue
        assessment = registration_candidate_assessment(robot, report)
        assessments[mode] = assessment
        if assessment['eligible']:
            metrics = report['mesh_metrics']
            eligible.append((
                float(metrics['dominant_component_triangle_ratio']),
                -float(metrics['point_to_mesh_residual']['median_m']),
                mode))
    selected = max(eligible)[2] if eligible else 'robot_pose'
    return selected, {
        'selected_mode': selected,
        'candidate_assessments': assessments,
        'selection_rule': (
            'select the eligible refinement with greatest component '
            'coherence, then lowest point-to-mesh residual; otherwise retain '
            'the timestamped robot-pose baseline'),
    }


def reconstruct(scan_dir, output_path, voxel_length=DEFAULT_VOXEL_LENGTH_M,
                sdf_trunc=DEFAULT_SDF_TRUNC_M,
                depth_trunc=DEFAULT_DEPTH_TRUNC_M,
                registration_mode='auto', expected_dimensions=None,
                allow_missing_calibration_id=False,
                allow_partial_view_set=False):
    scan = Path(scan_dir).resolve()
    with open(scan / 'manifest.json', 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest_sha256 = validate_manifest_integrity(scan, manifest)
    metadata_paths = metadata_paths_from_manifest(
        scan, manifest,
        allow_partial_view_set=allow_partial_view_set)
    metadata_values = [load_metadata(path) for path in metadata_paths]
    provenance = calibration_provenance(
        manifest, metadata_values, allow_missing_calibration_id)
    provenance = capture_schema_provenance(
        manifest, metadata_values, provenance, allow_missing_calibration_id)
    dimensions = expected_dimensions_m(expected_dimensions)
    mode = str(registration_mode)
    if mode not in REGISTRATION_MODES:
        raise ValueError('registration mode must be one of: %s' % ', '.join(
            REGISTRATION_MODES))
    output = Path(output_path).resolve()
    if mode != 'auto':
        return _reconstruct_single(
            scan, manifest, metadata_paths, metadata_values, output, mode,
            voxel_length, sdf_trunc, depth_trunc, dimensions, provenance,
            manifest_sha256, allow_partial_view_set)
    reports = {}
    errors = {}
    for candidate_mode in (
            'robot_pose', 'bounded_gicp', 'multiway_gicp'):
        candidate_output = output.with_name(
            output.stem + '.' + candidate_mode + output.suffix)
        try:
            reports[candidate_mode] = _reconstruct_single(
                scan, manifest, metadata_paths, metadata_values,
                candidate_output, candidate_mode, voxel_length, sdf_trunc,
                depth_trunc, dimensions, provenance, manifest_sha256,
                allow_partial_view_set)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors[candidate_mode] = str(exc)
    if not reports:
        raise ValueError('all reconstruction modes failed: %s' % errors)
    selected_mode, comparison = select_registration_reports(reports)
    if errors:
        comparison['mode_errors'] = errors
    selected = dict(reports[selected_mode])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    shutil.copy2(selected['mesh_path'], temporary)
    temporary.replace(output)
    selected.update({
        'registration_mode': selected_mode,
        'mesh_path': str(output),
        'mesh_sha256': sha256_file(output),
        'auto_comparison': comparison,
        'candidate_reports': {
            key: value for key, value in reports.items()},
        'candidate_errors': errors,
    })
    _atomic_write_json(output.with_suffix(output.suffix + '.quality.json'), selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--voxel-length', type=float,
                        default=DEFAULT_VOXEL_LENGTH_M)
    parser.add_argument('--sdf-trunc', type=float, default=DEFAULT_SDF_TRUNC_M)
    parser.add_argument('--depth-trunc', type=float,
                        default=DEFAULT_DEPTH_TRUNC_M)
    parser.add_argument('--registration-mode', choices=REGISTRATION_MODES,
                        default='auto')
    parser.add_argument(
        '--expected-dimensions-mm', type=float, nargs=3,
        metavar=('X', 'Y', 'Z'))
    parser.add_argument('--allow-missing-calibration-id', action='store_true')
    parser.add_argument(
        '--allow-partial-view-set', action='store_true',
        help=(
            'Permit 1-7 immutable captures for a partial/provisional mesh; '
            'the report remains explicitly PARTIAL_VIEW_SET.'))
    args = parser.parse_args()
    dimensions = (
        np.asarray(args.expected_dimensions_mm, dtype=float) / 1000.0
        if args.expected_dimensions_mm else None)
    print(json.dumps(reconstruct(
        args.scan_dir, args.output, args.voxel_length, args.sdf_trunc,
        args.depth_trunc, args.registration_mode, dimensions,
        args.allow_missing_calibration_id,
        args.allow_partial_view_set), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
