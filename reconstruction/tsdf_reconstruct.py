#!/usr/bin/env python3
"""
Secure offline object-masked RGB-D reconstruction for completed PiPER scans.

TSDF is the surface-fusion method. Sequential target GICP, target-only
multi-view GICP and static-scene RGB-D pose-graph refinement are bounded
residual corrections around capture-time robot poses. The separately
selectable constrained superposition fixes capture zero, permits unbounded
translation and caps minimum-prior rotation at 45 degrees; it remains an
offline visual/model fit and does not replace calibration or kinematics.
Constrained output also derives a separate consensus-supported dense OBJ with
depth-visible source-image texture; this never changes TSDF quality evidence.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np


DEFAULT_VOXEL_LENGTH_M = 0.003
DEFAULT_SDF_TRUNC_M = 0.015
DEFAULT_DEPTH_TRUNC_M = 1.5
REGISTRATION_MODES = (
    'robot_pose', 'bounded_gicp', 'multiway_gicp',
    'constrained_superposition', 'scene_pose_graph', 'auto')
MULTIWAY_MAX_PRIOR_NEIGHBORS = 3
MULTIWAY_MAX_VIEW_ANGLE_DEG = 75.0
MULTIWAY_MAX_CORRESPONDENCE_M = 0.015
SUPERPOSITION_MAX_CORRESPONDENCE_M = 0.015
SUPERPOSITION_NORMAL_ANGLE_DEG = 35.0
SUPERPOSITION_MAX_POINTS_PER_EDGE = 500
SUPERPOSITION_PRIOR_SIGMA_M = 0.008
SUPERPOSITION_ROTATION_PRIOR_SIGMA_DEG = 1.0
SUPERPOSITION_DATA_SIGMA_M = 0.002
SUPERPOSITION_HUBER_DELTA_M = 0.003
SUPERPOSITION_MAX_ITERATIONS = 6
SUPERPOSITION_MAX_ROTATION_DEG = 45.0
CONSENSUS_MIN_VOXEL_M = 0.0015
CONSENSUS_NORMAL_ANGLE_DEG = 35.0
CONSENSUS_MINIMUM_VIEWS = 2
SCENE_REGISTRATION_VOXEL_M = 0.005
MEASURED_VIEW_COLORS_RGB = (
    (0.90, 0.20, 0.20), (0.20, 0.55, 0.95), (0.20, 0.75, 0.35),
    (0.95, 0.65, 0.15), (0.65, 0.30, 0.90), (0.10, 0.75, 0.75),
)
SCENE_TARGET_EXCLUSION_RADIUS_PX = 6
SCENE_MINIMUM_POINTS = 500
MINIMUM_SIGNIFICANT_COMPONENT_AREA_FRACTION = 0.05


try:
    from reconstruction import input_provenance as _inputs
    from reconstruction import texture_baking as _texture
except ModuleNotFoundError as error:
    if error.name != 'reconstruction':
        raise
    import input_provenance as _inputs
    import texture_baking as _texture

MINIMUM_CAPTURE_VIEWS = _inputs.MINIMUM_CAPTURE_VIEWS
MAXIMUM_CAPTURE_VIEWS = _inputs.MAXIMUM_CAPTURE_VIEWS
MASK_SOURCES = _inputs.MASK_SOURCES
canonical_sha256 = _inputs.canonical_sha256
camera_extrinsic_from_metadata = _inputs.camera_extrinsic_from_metadata
sha256_file = _inputs.sha256_file
load_metadata = _inputs.load_metadata
capture_set_provenance = _inputs.capture_set_provenance
validate_capture_set = _inputs.validate_capture_set
validate_manifest_integrity = _inputs.validate_manifest_integrity
manifest_artifact_index = _inputs.manifest_artifact_index
prepare_offline_mask_context = _inputs.prepare_offline_mask_context
load_offline_target_mask = _inputs.load_offline_target_mask
metadata_paths_from_manifest = _inputs.metadata_paths_from_manifest
resolve_frame_artifacts = _inputs.resolve_frame_artifacts
confidence_capture_provenance = _inputs.confidence_capture_provenance
calibration_provenance = _inputs.calibration_provenance
capture_schema_provenance = _inputs.capture_schema_provenance


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


def _bounded_pose_graph_camera_poses(
        frames, o3d, cloud_key, registration_source,
        use_rgbd_odometry=False):
    """Refine camera poses from one explicit registration data source."""
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
        seed = initial
        odometry_attempted = False
        odometry_succeeded = False
        odometry_failure = ''
        if use_rgbd_odometry:
            odometry_attempted = True
            try:
                odometry_succeeded, odometry_transform, _ = \
                    o3d.pipelines.odometry.compute_rgbd_odometry(
                        frames[source]['registration_rgbd'],
                        frames[target]['registration_rgbd'],
                        frames[source]['intrinsic'], initial,
                        o3d.pipelines.odometry.
                        RGBDOdometryJacobianFromHybridTerm(),
                        o3d.pipelines.odometry.OdometryOption())
                odometry_transform = np.asarray(
                    odometry_transform, dtype=float)
                if odometry_succeeded \
                        and odometry_transform.shape == (4, 4) \
                        and np.all(np.isfinite(odometry_transform)):
                    seed = odometry_transform
                else:
                    odometry_succeeded = False
                    odometry_failure = 'RGB-D odometry found no valid solution'
            except RuntimeError as exc:
                odometry_failure = str(exc)
        result = o3d.pipelines.registration.registration_generalized_icp(
            frames[source][cloud_key], frames[target][cloud_key],
            MULTIWAY_MAX_CORRESPONDENCE_M, seed,
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
            'registration_source': registration_source,
            'rgbd_odometry_attempted': odometry_attempted,
            'rgbd_odometry_succeeded': bool(odometry_succeeded),
            'rgbd_odometry_failure': odometry_failure,
        }
        attempted_edges.append(record)
        if rejection:
            continue
        information = o3d.pipelines.registration.\
            get_information_matrix_from_point_clouds(
                frames[source][cloud_key],
                frames[target][cloud_key],
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
        'registration_source': registration_source,
        'target_geometry_used_for_registration': bool(
            registration_source == 'target_masked_depth'),
        'robot_pose_bounds_retained': {
            'maximum_translation_m': 0.020,
            'maximum_rotation_deg': 5.0,
        },
    }


def bounded_multiway_camera_poses(frames, o3d):
    """Refine poses from target-only geometry with the legacy behaviour."""
    return _bounded_pose_graph_camera_poses(
        frames, o3d, 'cloud_camera', 'target_masked_depth')


def scene_pose_graph_camera_poses(frames, o3d):
    """Refine poses from static RGB-D scene context, never target geometry."""
    for frame in frames:
        if frame.get('registration_cloud_camera') is None \
                or frame.get('registration_rgbd') is None:
            raise ValueError('static-scene registration inputs are unavailable')
        if len(frame['registration_cloud_camera'].points) < 20:
            raise ValueError('static-scene registration cloud is too small')
    return _bounded_pose_graph_camera_poses(
        frames, o3d, 'registration_cloud_camera',
        'static_scene_rgbd_excluding_target', use_rgbd_odometry=True)


def solve_constrained_translations(
        node_count, constraints, prior_sigma_m=SUPERPOSITION_PRIOR_SIGMA_M,
        data_sigma_m=SUPERPOSITION_DATA_SIGMA_M,
        huber_delta_m=SUPERPOSITION_HUBER_DELTA_M,
        maximum_translation_m=0.020, iterations=6):
    """Solve robust global per-view translations around fixed robot rotations.

    Each constraint is a point-to-plane equation between two capture views.
    A finite zero-mean robot-pose prior regularizes unobservable tangential
    motion, while the first pose is the fixed gauge anchor. This deliberately
    cannot invent target rotations from a symmetric silhouette.
    """
    count = int(node_count)
    if count < 2:
        raise ValueError('constrained superposition needs at least two views')
    if not constraints:
        raise ValueError('constrained superposition has no overlap constraints')
    prior_sigma = float(prior_sigma_m)
    data_sigma = float(data_sigma_m)
    huber_delta = float(huber_delta_m)
    if min(prior_sigma, data_sigma, huber_delta) <= 0.0:
        raise ValueError('superposition solver scales must be positive')
    rows = []
    values = []
    base_weights = []
    for item in constraints:
        source = int(item['source'])
        target = int(item['target'])
        normal = np.asarray(item['normal_base'], dtype=float)
        offset = float(item['offset_m'])
        edge_count = int(item.get('edge_constraint_count', 1))
        if source < 0 or target < 0 or source >= count or target >= count \
                or source == target or normal.shape != (3,) \
                or not np.all(np.isfinite(normal)) \
                or not np.isfinite(offset) or edge_count < 1:
            raise ValueError('superposition constraint is malformed')
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError('superposition constraint normal is zero')
        normal = normal / norm
        row = np.zeros(3 * count, dtype=float)
        row[3 * source:3 * source + 3] = normal
        row[3 * target:3 * target + 3] = -normal
        rows.append(row)
        values.append(offset / norm)
        base_weights.append(
            1.0 / (data_sigma * np.sqrt(float(edge_count))))
    data_row_count = len(rows)
    for node in range(count):
        sigma = 1e-5 if node == 0 else prior_sigma
        for axis in range(3):
            row = np.zeros(3 * count, dtype=float)
            row[3 * node + axis] = 1.0
            rows.append(row)
            values.append(0.0)
            base_weights.append(1.0 / sigma)
    matrix = np.asarray(rows, dtype=float)
    target_values = np.asarray(values, dtype=float)
    base = np.asarray(base_weights, dtype=float)
    solution = np.zeros(3 * count, dtype=float)
    for _unused in range(max(1, int(iterations))):
        raw_residual = matrix @ solution - target_values
        robust = np.ones_like(raw_residual)
        data_residual = np.abs(raw_residual[:data_row_count])
        outside = data_residual > huber_delta
        robust[:data_row_count][outside] = np.sqrt(
            huber_delta / data_residual[outside])
        weights = base * robust
        weighted_matrix = matrix * weights[:, None]
        weighted_target = target_values * weights
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            weighted_matrix, weighted_target, rcond=None)
        if rank < 3 * count:
            raise ValueError('constrained superposition system is singular')
    translations = solution.reshape(count, 3)
    magnitudes = np.linalg.norm(translations, axis=1)
    if np.any(~np.isfinite(translations)):
        raise ValueError('constrained superposition produced non-finite poses')
    if maximum_translation_m is not None and float(np.max(magnitudes)) \
            > float(maximum_translation_m) + 1e-12:
        raise ValueError(
            'constrained superposition correction exceeds %.0fmm'
            % (1000.0 * float(maximum_translation_m)))
    return translations


def solve_constrained_rigid_corrections(
        node_count, constraints, camera_origins,
        prior_sigma_m=SUPERPOSITION_PRIOR_SIGMA_M,
        rotation_prior_sigma_rad=np.radians(
            SUPERPOSITION_ROTATION_PRIOR_SIGMA_DEG),
        data_sigma_m=SUPERPOSITION_DATA_SIGMA_M,
        huber_delta_m=SUPERPOSITION_HUBER_DELTA_M,
        maximum_translation_m=0.020, maximum_rotation_deg=5.0,
        fixed_reference_index=None,
        iterations=SUPERPOSITION_MAX_ITERATIONS):
    """Solve prior-regularized camera corrections around each origin.

    The compatibility default keeps equal finite priors and bounded residual
    corrections. ``fixed_reference_index`` instead pins that capture exactly;
    callers may pass no translation ceiling while retaining a rotation bound.
    """
    count = int(node_count)
    origins = np.asarray(camera_origins, dtype=float)
    if count < 2 or origins.shape != (count, 3) \
            or not np.all(np.isfinite(origins)):
        raise ValueError('constrained rigid superposition origins are invalid')
    if not constraints:
        raise ValueError(
            'constrained rigid superposition has no overlap constraints')
    translation_sigma = float(prior_sigma_m)
    rotation_sigma = float(rotation_prior_sigma_rad)
    data_sigma = float(data_sigma_m)
    huber_delta = float(huber_delta_m)
    if min(translation_sigma, rotation_sigma, data_sigma, huber_delta) <= 0.0:
        raise ValueError('rigid superposition solver scales must be positive')
    rows = []
    values = []
    base_weights = []
    for item in constraints:
        source = int(item['source'])
        target = int(item['target'])
        source_point = np.asarray(item['source_point_base'], dtype=float)
        target_point = np.asarray(item['target_point_base'], dtype=float)
        normal = np.asarray(item['normal_base'], dtype=float)
        edge_count = int(item.get('edge_constraint_count', 1))
        if source < 0 or target < 0 or source >= count or target >= count \
                or source == target or source_point.shape != (3,) \
                or target_point.shape != (3,) or normal.shape != (3,) \
                or not np.all(np.isfinite(source_point)) \
                or not np.all(np.isfinite(target_point)) \
                or not np.all(np.isfinite(normal)) or edge_count < 1:
            raise ValueError('rigid superposition constraint is malformed')
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-12:
            raise ValueError('rigid superposition constraint normal is zero')
        normal = normal / normal_length
        correspondence_point = 0.5 * (source_point + target_point)
        row = np.zeros(6 * count, dtype=float)
        row[6 * source:6 * source + 3] = normal
        row[6 * source + 3:6 * source + 6] = np.cross(
            correspondence_point - origins[source], normal)
        row[6 * target:6 * target + 3] = -normal
        row[6 * target + 3:6 * target + 6] = -np.cross(
            correspondence_point - origins[target], normal)
        rows.append(row)
        values.append(float(np.dot(
            normal, target_point - source_point)))
        base_weights.append(
            1.0 / (data_sigma * np.sqrt(float(edge_count))))
    data_row_count = len(rows)
    reference = (
        None if fixed_reference_index is None else int(fixed_reference_index))
    if reference is not None and (reference < 0 or reference >= count):
        raise ValueError('fixed superposition reference is invalid')
    for node in range(count):
        node_translation_sigma = (
            1e-8 if node == reference else translation_sigma)
        node_rotation_sigma = 1e-8 if node == reference else rotation_sigma
        for axis in range(3):
            translation_row = np.zeros(6 * count, dtype=float)
            translation_row[6 * node + axis] = 1.0
            rows.append(translation_row)
            values.append(0.0)
            base_weights.append(1.0 / node_translation_sigma)
            rotation_row = np.zeros(6 * count, dtype=float)
            rotation_row[6 * node + 3 + axis] = 1.0
            rows.append(rotation_row)
            values.append(0.0)
            base_weights.append(1.0 / node_rotation_sigma)
    matrix = np.asarray(rows, dtype=float)
    target_values = np.asarray(values, dtype=float)
    base = np.asarray(base_weights, dtype=float)
    solution = np.zeros(6 * count, dtype=float)
    for _unused in range(max(1, int(iterations))):
        residual = matrix @ solution - target_values
        robust = np.ones_like(residual)
        data_residual = np.abs(residual[:data_row_count])
        outside = data_residual > huber_delta
        robust[:data_row_count][outside] = np.sqrt(
            huber_delta / data_residual[outside])
        weights = base * robust
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            matrix * weights[:, None], target_values * weights, rcond=None)
        if rank < 6 * count:
            raise ValueError('constrained rigid superposition system is singular')
    corrections = solution.reshape(count, 6)
    translations = corrections[:, :3]
    rotations = corrections[:, 3:]
    if reference is not None:
        translations[reference] = 0.0
        rotations[reference] = 0.0
    translation_magnitudes = np.linalg.norm(translations, axis=1)
    rotation_degrees = np.degrees(np.linalg.norm(rotations, axis=1))
    if not np.all(np.isfinite(corrections)):
        raise ValueError(
            'constrained rigid superposition produced non-finite poses')
    if maximum_translation_m is not None and float(np.max(
            translation_magnitudes)) \
            > float(maximum_translation_m) + 1e-12:
        raise ValueError(
            'constrained rigid superposition translation exceeds %.0fmm'
            % (1000.0 * float(maximum_translation_m)))
    if float(np.max(rotation_degrees)) \
            > float(maximum_rotation_deg) + 1e-12:
        raise ValueError(
            'constrained rigid superposition rotation exceeds %.1fdeg'
            % float(maximum_rotation_deg))
    constrained_pairs = {
        (int(item['source']), int(item['target'])) for item in constraints}
    for source, target in constrained_pairs:
        relative_translation = float(np.linalg.norm(
            translations[target] - translations[source]))
        relative_rotation_deg = float(np.degrees(np.linalg.norm(
            rotations[target] - rotations[source])))
        if maximum_translation_m is not None and relative_translation \
                > float(maximum_translation_m) + 1e-12:
            raise ValueError(
                'constrained rigid superposition relative translation '
                'exceeds %.0fmm' % (1000.0 * float(maximum_translation_m)))
        if reference is None and relative_rotation_deg \
                > float(maximum_rotation_deg) + 1e-12:
            raise ValueError(
                'constrained rigid superposition rotation exceeds %.1fdeg'
                % float(maximum_rotation_deg))
    return translations, rotations


def rotation_matrix_from_vector(rotation_vector):
    """Convert one finite axis-angle vector into a 3x3 rotation matrix."""
    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError('rotation vector is invalid')
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew))


def rigid_camera_correction_matrix(origin, translation, rotation_vector):
    """Return a base-frame correction rotating around the camera origin."""
    centre = np.asarray(origin, dtype=float)
    shift = np.asarray(translation, dtype=float)
    if centre.shape != (3,) or shift.shape != (3,) \
            or not np.all(np.isfinite(centre)) \
            or not np.all(np.isfinite(shift)):
        raise ValueError('rigid camera correction is malformed')
    rotation = rotation_matrix_from_vector(rotation_vector)
    correction = np.eye(4)
    correction[:3, :3] = rotation
    correction[:3, 3] = centre + shift - rotation @ centre
    return correction


def _superposition_edge_constraints(
        o3d, base_clouds, translations, source, target):
    """Build deterministic, normal-consistent target overlap constraints."""
    source_points = np.asarray(base_clouds[source].points, dtype=float)
    source_normals = np.asarray(base_clouds[source].normals, dtype=float)
    target_points = np.asarray(base_clouds[target].points, dtype=float)
    target_normals = np.asarray(base_clouds[target].normals, dtype=float)
    if min(len(source_points), len(target_points)) < 20 \
            or source_normals.shape != source_points.shape \
            or target_normals.shape != target_points.shape:
        return [], {
            'source': int(source), 'target': int(target), 'accepted': False,
            'rejection': 'target overlap cloud has insufficient normals'}
    tree = o3d.geometry.KDTreeFlann(base_clouds[target])
    step = max(1, int(np.ceil(
        len(source_points) / float(SUPERPOSITION_MAX_POINTS_PER_EDGE))))
    cosine_limit = float(np.cos(np.radians(
        SUPERPOSITION_NORMAL_ANGLE_DEG)))
    records = []
    distances = []
    for source_index in range(0, len(source_points), step):
        query = source_points[source_index] + translations[source]
        found, indices, squared = tree.search_knn_vector_3d(
            query - translations[target], 1)
        if found != 1:
            continue
        target_index = int(indices[0])
        distance = float(np.sqrt(squared[0]))
        if distance > SUPERPOSITION_MAX_CORRESPONDENCE_M:
            continue
        source_normal = source_normals[source_index]
        target_normal = target_normals[target_index]
        if abs(float(np.dot(source_normal, target_normal))) < cosine_limit:
            continue
        if float(np.dot(source_normal, target_normal)) < 0.0:
            target_normal = -target_normal
        records.append({
            'source': int(source),
            'target': int(target),
            'normal_base': target_normal,
            'source_point_base': source_points[source_index],
            'target_point_base': target_points[target_index],
            'offset_m': float(np.dot(
                target_normal,
                target_points[target_index] - source_points[source_index])),
        })
        distances.append(distance)
    minimum = max(20, int(0.08 * min(
        len(source_points), SUPERPOSITION_MAX_POINTS_PER_EDGE)))
    accepted = len(records) >= minimum
    if accepted:
        for record in records:
            record['edge_constraint_count'] = len(records)
    return (records if accepted else []), {
        'source': int(source),
        'target': int(target),
        'fitness': float(len(records)) / float(max(
            1, min(len(source_points), SUPERPOSITION_MAX_POINTS_PER_EDGE))),
        'inlier_rmse_m': (
            float(np.sqrt(np.mean(np.square(distances))))
            if distances else float('inf')),
        'correspondence_count': int(len(records)),
        'accepted': bool(accepted),
        'rejection': (
            '' if accepted else 'target overlap is too weak for superposition'),
        'registration_source':
            'target_masked_depth_capture_zero_anchored_global_alignment',
    }


def _rigid_superposition_edge_constraints(
        o3d, base_clouds, corrections, source, target):
    """Build correspondences under current bounded rigid corrections."""
    source_original = np.asarray(base_clouds[source].points, dtype=float)
    target_original = np.asarray(base_clouds[target].points, dtype=float)
    source_cloud = copy.deepcopy(base_clouds[source])
    target_cloud = copy.deepcopy(base_clouds[target])
    source_cloud.transform(np.asarray(corrections[source], dtype=float))
    target_cloud.transform(np.asarray(corrections[target], dtype=float))
    source_points = np.asarray(source_cloud.points, dtype=float)
    target_points = np.asarray(target_cloud.points, dtype=float)
    source_normals = np.asarray(source_cloud.normals, dtype=float)
    target_normals = np.asarray(target_cloud.normals, dtype=float)
    if min(len(source_points), len(target_points)) < 20 \
            or source_normals.shape != source_points.shape \
            or target_normals.shape != target_points.shape:
        return [], {
            'source': int(source), 'target': int(target), 'accepted': False,
            'rejection': 'target overlap cloud has insufficient normals'}
    target_tree = o3d.geometry.KDTreeFlann(target_cloud)
    source_tree = o3d.geometry.KDTreeFlann(source_cloud)
    step = max(1, int(np.ceil(
        len(source_points) / float(SUPERPOSITION_MAX_POINTS_PER_EDGE))))
    cosine_limit = float(np.cos(np.radians(
        SUPERPOSITION_NORMAL_ANGLE_DEG)))
    records = []
    distances = []
    for source_index in range(0, len(source_points), step):
        found, indices, squared = target_tree.search_knn_vector_3d(
            source_points[source_index], 1)
        if found != 1:
            continue
        target_index = int(indices[0])
        distance = float(np.sqrt(squared[0]))
        if distance > SUPERPOSITION_MAX_CORRESPONDENCE_M:
            continue
        reverse_found, reverse_indices, _reverse_squared = \
            source_tree.search_knn_vector_3d(target_points[target_index], 1)
        if reverse_found != 1 or int(reverse_indices[0]) != source_index:
            continue
        source_normal = source_normals[source_index]
        target_normal = target_normals[target_index]
        if abs(float(np.dot(source_normal, target_normal))) < cosine_limit:
            continue
        if float(np.dot(source_normal, target_normal)) < 0.0:
            target_normal = -target_normal
        records.append({
            'source': int(source),
            'target': int(target),
            'normal_base': target_normal,
            'source_point_base': source_original[source_index],
            'target_point_base': target_original[target_index],
        })
        distances.append(distance)
    minimum = max(20, int(0.08 * min(
        len(source_points), SUPERPOSITION_MAX_POINTS_PER_EDGE)))
    accepted = len(records) >= minimum
    if accepted:
        for record in records:
            record['edge_constraint_count'] = len(records)
    return (records if accepted else []), {
        'source': int(source),
        'target': int(target),
        'fitness': float(len(records)) / float(max(
            1, min(len(source_points), SUPERPOSITION_MAX_POINTS_PER_EDGE))),
        'inlier_rmse_m': (
            float(np.sqrt(np.mean(np.square(distances))))
            if distances else float('inf')),
        'correspondence_count': int(len(records)),
        'accepted': bool(accepted),
        'rejection': (
            '' if accepted else
            'target overlap is too weak for rigid superposition'),
        'registration_source':
            'target_masked_depth_capture_zero_anchored_global_alignment',
        'correspondence_policy': 'mutual_nearest_normal_consistent',
    }


def constrained_superposition_camera_poses(frames, o3d):
    """Align all measured surfaces in an exactly fixed capture-zero frame."""
    if not frames:
        raise ValueError(
            'constrained superposition requires at least one view')
    nominal = [
        np.asarray(frame['nominal_base_camera'], dtype=float)
        for frame in frames]
    if len(frames) == 1:
        return [nominal[0].copy()], {
            'candidate_pair_count': 0,
            'accepted_pair_count': 0,
            'spanning_tree_pair_count': 0,
            'constraint_count': 0,
            'pairwise_edges': [],
            'pose_corrections': [{
                'frame_index': 0,
                'translation_correction_m': 0.0,
                'translation_vector_m': [0.0, 0.0, 0.0],
                'rotation_correction_deg': 0.0,
                'rotation_vector_rad': [0.0, 0.0, 0.0],
                'fixed_reference': True,
            }],
            'correction_objective': {
                'policy':
                    'single_capture_fixed_reference_no_alignment',
                'total_translation_correction_m': 0.0,
                'rms_translation_correction_m': 0.0,
                'total_rotation_correction_deg': 0.0,
                'rms_rotation_correction_deg': 0.0,
            },
            'registration_source':
                'single_capture_nominal_pose_no_alignment',
            'target_geometry_used_for_registration': False,
            'single_capture_noop': True,
            'alignment_bounds': {
                'fixed_reference_frame_index': 0,
                'maximum_translation_m': None,
                'maximum_rotation_deg': SUPERPOSITION_MAX_ROTATION_DEG,
            },
            'solver': {
                'model': 'single_capture_no_alignment_required',
                'gauge_policy': 'capture_zero_fixed_exactly',
                'correspondence_policy': 'not_applicable_single_capture',
                'translation_limit_m': None,
                'rotation_prior_sigma_deg':
                    SUPERPOSITION_ROTATION_PRIOR_SIGMA_DEG,
                'maximum_rotation_deg': SUPERPOSITION_MAX_ROTATION_DEG,
                'data_sigma_m': SUPERPOSITION_DATA_SIGMA_M,
                'huber_delta_m': SUPERPOSITION_HUBER_DELTA_M,
                'maximum_iterations': 0,
                'full_resolution_depth_fused_once': True,
            },
        }
    clouds = []
    for frame, pose in zip(frames, nominal):
        cloud = copy.deepcopy(frame['cloud_camera'])
        cloud.transform(pose.copy())
        clouds.append(cloud)
    translations = np.zeros((len(frames), 3), dtype=float)
    attempted = []
    accepted_pairs = []
    constraints = []
    for _iteration in range(SUPERPOSITION_MAX_ITERATIONS):
        attempted = []
        accepted_pairs = []
        constraints = []
        for source, target in multiway_registration_pairs(nominal):
            edge_constraints, report = _superposition_edge_constraints(
                o3d, clouds, translations, source, target)
            attempted.append(report)
            if report['accepted']:
                accepted_pairs.append(report)
                constraints.extend(edge_constraints)
        registration_spanning_tree(len(frames), accepted_pairs)
        updated = solve_constrained_translations(
            len(frames), constraints,
            prior_sigma_m=1.0e6, iterations=6,
            maximum_translation_m=None)
        updated[0] = 0.0
        if float(np.max(np.linalg.norm(updated - translations, axis=1))) \
                < 1e-5:
            translations = updated
            break
        translations = updated
    camera_origins = np.asarray(
        [pose[:3, 3] for pose in nominal], dtype=float)
    rotations = np.zeros((len(frames), 3), dtype=float)
    constraints = []
    attempted = []
    accepted_pairs = []
    for _iteration in range(SUPERPOSITION_MAX_ITERATIONS):
        current_corrections = [
            rigid_camera_correction_matrix(origin, translation, rotation)
            for origin, translation, rotation in zip(
                camera_origins, translations, rotations)]
        attempted = []
        accepted_pairs = []
        constraints = []
        for source, target in multiway_registration_pairs(nominal):
            edge_constraints, report = \
                _rigid_superposition_edge_constraints(
                    o3d, clouds, current_corrections, source, target)
            attempted.append(report)
            if report['accepted']:
                accepted_pairs.append(report)
                constraints.extend(edge_constraints)
        registration_spanning_tree(len(frames), accepted_pairs)
        updated_translations, updated_rotations = \
            solve_constrained_rigid_corrections(
                len(frames), constraints, camera_origins,
                prior_sigma_m=1.0e6,
                maximum_translation_m=None,
                maximum_rotation_deg=SUPERPOSITION_MAX_ROTATION_DEG,
                fixed_reference_index=0)
        translation_delta = float(np.max(np.linalg.norm(
            updated_translations - translations, axis=1)))
        rotation_delta_deg = float(np.degrees(np.max(np.linalg.norm(
            updated_rotations - rotations, axis=1))))
        translations = updated_translations
        rotations = updated_rotations
        if translation_delta < 1e-5 and rotation_delta_deg < 0.01:
            break
    refined = []
    correction_reports = []
    for index, (pose, origin, translation, rotation_vector) in enumerate(zip(
            nominal, camera_origins, translations, rotations)):
        correction = rigid_camera_correction_matrix(
            origin, translation, rotation_vector)
        refined_pose = correction @ pose
        camera_shift = float(np.linalg.norm(
            refined_pose[:3, 3] - pose[:3, 3]))
        rotation_degrees = float(np.degrees(
            rotation_angle_rad(correction)))
        if rotation_degrees > SUPERPOSITION_MAX_ROTATION_DEG + 1e-9:
            raise ValueError(
                'capture %d rotation exceeds %.1fdeg' % (
                    index, SUPERPOSITION_MAX_ROTATION_DEG))
        refined.append(refined_pose)
        correction_reports.append({
            'frame_index': int(index),
            'translation_correction_m': camera_shift,
            'translation_vector_m': translation.tolist(),
            'rotation_correction_deg': rotation_degrees,
            'rotation_vector_rad': rotation_vector.tolist(),
            'fixed_reference': bool(index == 0),
        })
    translation_costs = np.asarray([
        item['translation_correction_m'] for item in correction_reports],
        dtype=float)
    rotation_costs = np.asarray([
        item['rotation_correction_deg'] for item in correction_reports],
        dtype=float)
    return refined, {
        'candidate_pair_count': len(attempted),
        'accepted_pair_count': len(accepted_pairs),
        'spanning_tree_pair_count': len(frames) - 1,
        'constraint_count': len(constraints),
        'pairwise_edges': attempted,
        'pose_corrections': correction_reports,
        'correction_objective': {
            'policy':
                'fixed_capture_zero_maximum_overlap_minimum_rotation',
            'total_translation_correction_m': float(np.sum(
                translation_costs)),
            'rms_translation_correction_m': float(np.sqrt(np.mean(
                np.square(translation_costs)))),
            'total_rotation_correction_deg': float(np.sum(rotation_costs)),
            'rms_rotation_correction_deg': float(np.sqrt(np.mean(
                np.square(rotation_costs)))),
        },
        'registration_source':
            'target_masked_depth_capture_zero_anchored_global_alignment',
        'target_geometry_used_for_registration': True,
        'alignment_bounds': {
            'fixed_reference_frame_index': 0,
            'maximum_translation_m': None,
            'maximum_rotation_deg': SUPERPOSITION_MAX_ROTATION_DEG,
        },
        'solver': {
            'model': 'capture_zero_anchored_global_point_to_plane_alignment',
            'gauge_policy': 'capture_zero_fixed_exactly',
            'correspondence_policy':
                'mutual_nearest_normal_consistent',
            'translation_limit_m': None,
            'rotation_prior_sigma_deg':
                SUPERPOSITION_ROTATION_PRIOR_SIGMA_DEG,
            'maximum_rotation_deg': SUPERPOSITION_MAX_ROTATION_DEG,
            'data_sigma_m': SUPERPOSITION_DATA_SIGMA_M,
            'huber_delta_m': SUPERPOSITION_HUBER_DELTA_M,
            'maximum_iterations': SUPERPOSITION_MAX_ITERATIONS,
            'full_resolution_depth_fused_once': True,
        },
    }


def robust_cross_capture_consensus(
        view_surfaces, voxel_length_m,
        minimum_views=CONSENSUS_MINIMUM_VIEWS,
        maximum_normal_angle_deg=CONSENSUS_NORMAL_ANGLE_DEG):
    """Fuse corresponding surface samples with one vote per capture.

    Points inside one spatial cell are never averaged within a capture. The
    sample closest to the cell centre represents that capture. Across captures
    a component-wise median and MAD reject positional outliers, after which an
    equal-weight mean retains useful sub-pixel information without allowing a
    dense or noisy view to dominate.
    """
    cell = max(CONSENSUS_MIN_VOXEL_M, float(voxel_length_m))
    required = int(minimum_views)
    if not np.isfinite(cell) or cell <= 0.0:
        raise ValueError('consensus voxel length must be positive')
    if required < 2:
        raise ValueError('consensus requires at least two different captures')
    cosine_limit = float(np.cos(np.radians(
        float(maximum_normal_angle_deg))))
    cells = {}
    input_points = 0
    for view_index, surface in enumerate(view_surfaces):
        points = np.asarray(surface['points'], dtype=float)
        normals = np.asarray(surface['normals'], dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) \
                or normals.shape != points.shape \
                or not np.all(np.isfinite(points)) \
                or not np.all(np.isfinite(normals)):
            raise ValueError('consensus view surface is malformed')
        input_points += len(points)
        representatives = {}
        for point, normal in zip(points, normals):
            key = tuple(np.floor(point / cell).astype(np.int64).tolist())
            centre = (np.asarray(key, dtype=float) + 0.5) * cell
            distance = float(np.linalg.norm(point - centre))
            current = representatives.get(key)
            if current is None or distance < current[0]:
                representatives[key] = (distance, point.copy(), normal.copy())
        for key, (_distance, point, normal) in representatives.items():
            cells.setdefault(key, {})[int(view_index)] = (point, normal)

    consensus_points = []
    consensus_normals = []
    support_counts = []
    rejected_normal = 0
    rejected_position = 0
    single_view_cells = 0
    per_cell_spread = []
    for key in sorted(cells):
        by_view = cells[key]
        if len(by_view) < required:
            single_view_cells += 1
            continue
        entries = list(by_view.values())
        points = np.asarray([item[0] for item in entries], dtype=float)
        normals = np.asarray([item[1] for item in entries], dtype=float)
        normal_lengths = np.linalg.norm(normals, axis=1)
        valid_normals = normal_lengths > 1e-9
        if int(np.count_nonzero(valid_normals)) < required:
            rejected_normal += 1
            continue
        normals = normals / normal_lengths[:, None]
        compatibility = np.abs(normals @ normals.T) >= cosine_limit
        seed_index = int(np.argmax(np.sum(compatibility, axis=1)))
        compatible = compatibility[seed_index]
        if int(np.count_nonzero(compatible)) < required:
            rejected_normal += 1
            continue
        points = points[compatible]
        normals = normals[compatible]
        reference_normal = normals[0]
        normals[np.dot(normals, reference_normal) < 0.0] *= -1.0

        median = np.median(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        distance_median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - distance_median)))
        threshold = max(
            0.75 * cell,
            distance_median + 3.0 * 1.4826 * mad)
        threshold = min(threshold, np.sqrt(3.0) * cell)
        positional = distances <= threshold + 1e-12
        if int(np.count_nonzero(positional)) < required:
            rejected_position += 1
            continue
        points = points[positional]
        normals = normals[positional]
        fused_point = np.mean(points, axis=0)
        fused_normal = np.mean(normals, axis=0)
        fused_normal_length = float(np.linalg.norm(fused_normal))
        if fused_normal_length <= 1e-9:
            rejected_normal += 1
            continue
        consensus_points.append(fused_point)
        consensus_normals.append(fused_normal / fused_normal_length)
        support_counts.append(len(points))
        per_cell_spread.append(float(np.max(
            np.linalg.norm(points - fused_point, axis=1))))
    if not consensus_points:
        raise ValueError(
            'cross-capture consensus found no multi-view correspondences')
    return (
        np.asarray(consensus_points, dtype=float),
        np.asarray(consensus_normals, dtype=float),
        np.asarray(support_counts, dtype=int),
        {
            'method':
                'one_representative_per_capture_then_MAD_filtered_mean',
            'voxel_length_m': cell,
            'minimum_distinct_captures': required,
            'maximum_normal_angle_deg': float(maximum_normal_angle_deg),
            'input_measured_points': int(input_points),
            'spatial_cell_count': int(len(cells)),
            'confirmed_consensus_points': int(len(consensus_points)),
            'single_view_cells_excluded': int(single_view_cells),
            'normal_incompatible_cells_excluded': int(rejected_normal),
            'position_outlier_cells_excluded': int(rejected_position),
            'minimum_capture_support': int(np.min(support_counts)),
            'median_capture_support': float(np.median(support_counts)),
            'maximum_capture_support': int(np.max(support_counts)),
            'median_maximum_cross_capture_spread_m': float(
                np.median(per_cell_spread)),
            'p90_maximum_cross_capture_spread_m': float(
                np.percentile(per_cell_spread, 90)),
            'same_capture_points_averaged_together': False,
            'immutable_source_points_modified': False,
        })


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


def masked_camera_cloud(
        o3d, depth_mm, mask, intrinsic, depth_trunc=1.5,
        voxel_length=DEFAULT_VOXEL_LENGTH_M):
    values = np.asarray(depth_mm, dtype=np.uint16).copy()
    values[np.asarray(mask) <= 0] = 0
    image = o3d.geometry.Image(values)
    cloud = o3d.geometry.PointCloud.create_from_depth_image(
        image, intrinsic, depth_scale=1000.0,
        depth_trunc=float(depth_trunc), stride=1,
        project_valid_depth_only=True)
    if voxel_length is not None:
        cloud = cloud.voxel_down_sample(float(voxel_length))
    if len(cloud.points) >= 20:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    return cloud


def scene_registration_support_mask(
        cv2, depth_mm, target_mask, depth_trunc,
        exclusion_radius_px=SCENE_TARGET_EXCLUSION_RADIUS_PX):
    """Return static-scene support while excluding the target and bad depth."""
    depth = np.asarray(depth_mm)
    target = np.asarray(target_mask)
    if depth.ndim != 2 or target.shape != depth.shape:
        raise ValueError('scene depth and target mask dimensions disagree')
    radius = int(exclusion_radius_px)
    if radius < 0 or radius > 64:
        raise ValueError('scene target-exclusion radius is invalid')
    binary_target = np.asarray(target > 0, dtype=np.uint8)
    if radius:
        kernel_size = 2 * radius + 1
        binary_target = cv2.dilate(
            binary_target,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1)
    depth_m = depth.astype(float) / 1000.0
    return (
        np.isfinite(depth_m) & (depth_m > 0.10)
        & (depth_m <= float(depth_trunc)) & (binary_target == 0))


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


def target_component_policy(component_triangle_counts, component_areas,
                            minimum_area_fraction=(
                                MINIMUM_SIGNIFICANT_COMPONENT_AREA_FRACTION)):
    """Classify disconnected mesh surfaces without hiding substantial ones."""
    counts = np.asarray(component_triangle_counts, dtype=int)
    areas = np.asarray(component_areas, dtype=float)
    fraction = float(minimum_area_fraction)
    if counts.ndim != 1 or areas.ndim != 1 or counts.shape != areas.shape \
            or not counts.size:
        raise ValueError(
            'mesh component counts and areas must be nonempty vectors')
    if np.any(counts <= 0) or np.any(~np.isfinite(areas)) \
            or np.any(areas < 0.0):
        raise ValueError(
            'mesh component counts and areas must be finite and positive')
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            'component area fraction must be between zero and one')
    dominant = int(np.argmax(areas))
    dominant_area = float(areas[dominant])
    if dominant_area <= 0.0:
        raise ValueError('mesh components have no positive surface area')
    threshold = dominant_area * fraction
    retained = np.flatnonzero(areas >= threshold).astype(int)
    removed = np.flatnonzero(areas < threshold).astype(int)
    total_triangles = int(np.sum(counts))
    total_area = float(np.sum(areas))
    return {
        'connected_target_assumption': True,
        'minimum_relative_surface_area': fraction,
        'surface_area_threshold_m2': threshold,
        'original_component_count': int(counts.size),
        'original_triangle_count': total_triangles,
        'original_surface_area_m2': total_area,
        'original_dominant_component_triangle_ratio': (
            float(np.max(counts)) / float(total_triangles)),
        'original_dominant_component_surface_area_ratio': (
            dominant_area / total_area),
        'retained_component_indices': retained.tolist(),
        'retained_component_count': int(retained.size),
        'retained_triangle_count': int(np.sum(counts[retained])),
        'retained_surface_area_m2': float(np.sum(areas[retained])),
        'removed_fragment_component_count': int(removed.size),
        'removed_fragment_triangle_count': int(np.sum(counts[removed])),
        'removed_fragment_surface_area_m2': float(np.sum(areas[removed])),
        'connectivity_valid': bool(retained.size == 1),
        'decision': (
            'SINGLE_CONNECTED_TARGET'
            if retained.size == 1 else 'MULTIPLE_SUBSTANTIAL_COMPONENTS'),
    }


def filter_target_mesh_components(mesh):
    """Remove only tiny components and retain substantial split surfaces."""
    filtered = copy.deepcopy(mesh)
    filtered.remove_duplicated_triangles()
    filtered.remove_degenerate_triangles()
    filtered.remove_duplicated_vertices()
    filtered.remove_unreferenced_vertices()
    labels, counts, areas = filtered.cluster_connected_triangles()
    report = target_component_policy(counts, areas)
    retained = set(report['retained_component_indices'])
    remove_mask = [int(label) not in retained for label in labels]
    filtered.remove_triangles_by_mask(remove_mask)
    filtered.remove_unreferenced_vertices()
    filtered.compute_vertex_normals()
    report['output_vertex_count'] = int(len(filtered.vertices))
    report['output_triangle_count'] = int(len(filtered.triangles))
    return filtered, report


def raw_mesh_output_path(output_path):
    """Return the adjacent raw-mesh path for one cleaned mesh output."""
    output = Path(output_path)
    return output.with_name(output.stem + '.raw' + output.suffix)


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
                  depth_trunc, expected_dimensions=None,
                  include_scene_registration=False,
                  offline_mask_context=None):
    artifacts = resolve_frame_artifacts(scan, metadata_path, metadata, manifest)
    raw_rgb_bgr = cv2.imread(str(artifacts['rgb']), cv2.IMREAD_COLOR)
    scene_depth = cv2.imread(str(artifacts['depth']), cv2.IMREAD_UNCHANGED)
    scene_target_mask = cv2.imread(
        str(artifacts['mask']), cv2.IMREAD_GRAYSCALE)
    input_provenance = confidence_capture_provenance(
        metadata, manifest, artifacts, cv2)
    if input_provenance['confidence_qualified']:
        target_depth = cv2.imread(
            str(artifacts['target_depth']), cv2.IMREAD_UNCHANGED)
        target_mask = cv2.imread(
            str(artifacts['target_support_mask']), cv2.IMREAD_GRAYSCALE)
        target = metadata.get('synchronized_target_3d', {})
    else:
        target_depth = scene_depth
        target_mask = cv2.imread(
            str(artifacts['mask']), cv2.IMREAD_GRAYSCALE)
        target = metadata.get('target_3d', {})

    if offline_mask_context is not None:
        fresh_mask, fresh_record = load_offline_target_mask(
            offline_mask_context, metadata_path, artifacts['rgb'],
            raw_rgb_bgr.shape[:2], cv2)
        original_support_points = int(np.count_nonzero(target_mask))
        target_mask = (
            (np.asarray(target_mask) > 0) & (fresh_mask > 0)
        ).astype(np.uint8) * 255
        target_depth = np.asarray(target_depth, dtype=np.uint16).copy()
        target_depth[target_mask == 0] = 0
        retained_points = int(np.count_nonzero(target_mask))
        minimum_points = int(
            manifest.get('confidence_policy', {}).get(
                'minimum_target_points', 20))
        if retained_points < minimum_points:
            raise ValueError(
                '%s offline mask retains only %d confidence-qualified target '
                'points; need %d'
                % (metadata_path.name, retained_points, minimum_points))
        input_provenance = dict(input_provenance)
        input_provenance.update({
            'semantic_mask_source':
                'offline_groundingdino_sam2_intersected_with_captured_support',
            'offline_mask_sha256': str(fresh_record['mask_sha256']),
            'offline_groundingdino_confidence': float(
                fresh_record['groundingdino_confidence']),
            'offline_sam2_score': float(fresh_record['sam2_score']),
            'captured_support_points_before_offline_mask':
                original_support_points,
            'retained_support_points_after_offline_mask': retained_points,
            'offline_mask_never_expands_captured_support': True,
        })
    else:
        input_provenance = dict(input_provenance)
        input_provenance['semantic_mask_source'] = 'captured_live_sam2'

    # Target fusion and scene registration share one rectified color plane but
    # retain separate depth support.  The scene depth is never integrated into
    # the target TSDF.
    rgb_bgr, target_depth, target_mask, k, rectified = rectify_rgbd_mask(
        cv2, raw_rgb_bgr, target_depth, target_mask,
        metadata.get('camera_info', {}))
    height, width = target_depth.shape[:2]
    valid, depth_gate = target_depth_support_mask(
        target_depth, target_mask, target.get('depth'), expected_dimensions)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 20:
        raise ValueError('%s has too few masked depth points' % metadata_path.name)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    depth = np.asarray(target_depth, dtype=np.uint16)
    depth[~valid] = 0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(width), int(height), float(k[0, 0]), float(k[1, 1]),
        float(k[0, 2]), float(k[1, 2]))
    nominal_base_camera = np.linalg.inv(camera_extrinsic_from_metadata(metadata))
    cloud_camera = masked_camera_cloud(
        o3d, depth, target_mask, intrinsic, depth_trunc=depth_trunc)
    measured_cloud_camera = masked_camera_cloud(
        o3d, depth, target_mask, intrinsic, depth_trunc=depth_trunc,
        voxel_length=None)

    registration_cloud_camera = None
    registration_rgbd = None
    scene_valid_count = 0
    if include_scene_registration:
        scene_rgb_bgr, scene_depth, scene_target_mask, scene_k, \
            scene_rectified = rectify_rgbd_mask(
                cv2, raw_rgb_bgr, scene_depth, scene_target_mask,
                metadata.get('camera_info', {}))
        if rectified != scene_rectified \
                or not np.allclose(k, scene_k, atol=1e-9):
            raise ValueError('target and scene rectification disagree')
        scene_support = scene_registration_support_mask(
            cv2, scene_depth, scene_target_mask, depth_trunc)
        scene_valid_count = int(np.count_nonzero(scene_support))
        if scene_valid_count < SCENE_MINIMUM_POINTS:
            raise ValueError(
                '%s has too few static-scene depth points'
                % metadata_path.name)
        scene_rgb = cv2.cvtColor(scene_rgb_bgr, cv2.COLOR_BGR2RGB)
        scene_rgb[~scene_support] = 0
        scene_depth = np.asarray(scene_depth, dtype=np.uint16).copy()
        scene_depth[~scene_support] = 0
        registration_cloud_camera = masked_camera_cloud(
            o3d, scene_depth, scene_support, intrinsic,
            depth_trunc=depth_trunc,
            voxel_length=SCENE_REGISTRATION_VOXEL_M)
        registration_rgbd = \
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(scene_rgb)),
                o3d.geometry.Image(np.ascontiguousarray(scene_depth)),
                depth_scale=1000.0, depth_trunc=float(depth_trunc),
                convert_rgb_to_intensity=True)
    return {
        'metadata_path': metadata_path,
        'metadata': metadata,
        'rgb': np.ascontiguousarray(rgb),
        'depth': np.ascontiguousarray(depth),
        'mask': np.ascontiguousarray(target_mask),
        'camera_matrix': np.asarray(k, dtype=float).copy(),
        'intrinsic': intrinsic,
        'nominal_base_camera': nominal_base_camera,
        'cloud_camera': cloud_camera,
        'measured_cloud_camera': measured_cloud_camera,
        'registration_cloud_camera': registration_cloud_camera,
        'registration_rgbd': registration_rgbd,
        'valid_scene_depth_points': scene_valid_count,
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
                        allow_partial_view_set=False,
                        offline_mask_context=None):
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
            expected_dimensions,
            include_scene_registration=(
                registration_mode == 'scene_pose_graph'),
            offline_mask_context=offline_mask_context)
        for metadata_path, metadata in zip(metadata_paths, metadata_values)]
    multiway_diagnostics = None
    refined_multiway_poses = None
    if registration_mode == 'multiway_gicp':
        refined_multiway_poses, multiway_diagnostics = \
            bounded_multiway_camera_poses(frames, o3d)
    elif registration_mode == 'constrained_superposition':
        refined_multiway_poses, multiway_diagnostics = \
            constrained_superposition_camera_poses(frames, o3d)
    elif registration_mode == 'scene_pose_graph':
        refined_multiway_poses, multiway_diagnostics = \
            scene_pose_graph_camera_poses(frames, o3d)
    registration = []
    frame_inputs = []
    accumulated = None
    measured_cloud = None
    measured_cloud_views = []
    consensus_view_surfaces = []
    texture_frames = []
    for frame_index, (metadata_path, frame) in enumerate(
            zip(metadata_paths, frames)):
        cloud_base = None
        correction = np.eye(4)
        accepted = False
        rejection = 'robot-pose input retained'
        fitness = 1.0
        inlier_rmse = 0.0
        proposed_translation = 0.0
        proposed_rotation = 0.0
        if refined_multiway_poses is not None:
            refined_base_camera = refined_multiway_poses[frame_index]
            correction = (
                refined_base_camera
                @ np.linalg.inv(frame['nominal_base_camera']))
            proposed_translation = float(np.linalg.norm(
                refined_base_camera[:3, 3]
                - frame['nominal_base_camera'][:3, 3]))
            proposed_rotation = float(
                np.degrees(rotation_angle_rad(correction)))
            accepted = (
                proposed_translation > 1e-9 or proposed_rotation > 1e-9)
            rejection = ''
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
            proposed_base_camera = (
                proposed @ frame['nominal_base_camera'])
            proposed_translation = float(np.linalg.norm(
                proposed_base_camera[:3, 3]
                - frame['nominal_base_camera'][:3, 3]))
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
        measured_view = copy.deepcopy(frame['measured_cloud_camera'])
        measured_view.transform(refined_base_camera.copy())
        if registration_mode == 'constrained_superposition':
            consensus_view_surfaces.append({
                'points': np.asarray(
                    measured_view.points, dtype=float).copy(),
                'normals': np.asarray(
                    measured_view.normals, dtype=float).copy(),
            })
            texture_frames.append({
                'rgb': frame['rgb'],
                'depth': frame['depth'],
                'mask': frame['mask'],
                'camera_matrix': frame['camera_matrix'],
                'T_base_camera': refined_base_camera.copy(),
                'frame': metadata_path.name,
            })
        measured_color = MEASURED_VIEW_COLORS_RGB[
            frame_index % len(MEASURED_VIEW_COLORS_RGB)]
        measured_view.paint_uniform_color(measured_color)
        measured_cloud = (
            measured_view if measured_cloud is None
            else measured_cloud + measured_view)
        measured_cloud_views.append({
            'frame': metadata_path.name,
            'point_count': len(measured_view.points),
            'display_color_rgb': list(measured_color),
        })
        registration.append({
            'frame': metadata_path.name,
            'fitness': fitness,
            'inlier_rmse_m': inlier_rmse,
            'correction_accepted': accepted,
            'correction_rejection': rejection,
            'proposed_translation_correction_m': proposed_translation,
            'proposed_rotation_correction_deg': proposed_rotation,
            'applied_translation_correction_m': float(
                np.linalg.norm(
                    refined_base_camera[:3, 3]
                    - frame['nominal_base_camera'][:3, 3])),
            'applied_rotation_correction_deg': float(
                np.degrees(rotation_angle_rad(correction))),
            'rectified': bool(frame['rectified']),
            'valid_scene_depth_points': frame['valid_scene_depth_points'],
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
            'semantic_mask_source': frame['input_provenance'].get(
                'semantic_mask_source', 'unknown'),
        })
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame['rgb']), o3d.geometry.Image(frame['depth']),
            depth_scale=1000.0, depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd, frame['intrinsic'], np.linalg.inv(refined_base_camera))
    extracted_mesh = volume.extract_triangle_mesh()
    extracted_mesh.compute_vertex_normals()
    raw_mesh_size_qualified = (
        len(extracted_mesh.vertices) >= 100
        and len(extracted_mesh.triangles) >= 100)
    if not len(extracted_mesh.vertices) \
            or not len(extracted_mesh.triangles):
        raise ValueError(
            'reconstruction produced an empty mesh '
            '(%d vertices, %d triangles)'
            % (len(extracted_mesh.vertices), len(extracted_mesh.triangles)))
    raw_metrics = mesh_metrics(
        o3d, extracted_mesh, accumulated, expected_dimensions)
    mesh, component_filter = filter_target_mesh_components(extracted_mesh)
    cleaned_mesh_size_qualified = (
        len(mesh.vertices) >= 100 and len(mesh.triangles) >= 100)
    if not len(mesh.vertices) or not len(mesh.triangles):
        raise ValueError(
            'connected-target filtering produced an empty mesh '
            '(%d vertices, %d triangles)'
            % (len(mesh.vertices), len(mesh.triangles)))
    raw_output = _atomic_write_mesh(
        o3d, extracted_mesh, raw_mesh_output_path(output_path))
    output = _atomic_write_mesh(o3d, mesh, output_path)
    input_cloud_path = _atomic_write_cloud(
        o3d, accumulated,
        output.with_name(output.stem + '.input_cloud.ply'))
    measured_cloud_path = _atomic_write_cloud(
        o3d, measured_cloud,
        output.with_name(output.stem + '.measured_points.ply'))
    consensus_cloud_path = None
    consensus_cloud = None
    consensus_diagnostics = None
    consensus_points = None
    if registration_mode == 'constrained_superposition':
        if len(consensus_view_surfaces) == 1:
            consensus_diagnostics = {
                'available': False,
                'reason':
                    'cross-capture consensus requires at least two '
                    'distinct captures',
                'minimum_distinct_captures': CONSENSUS_MINIMUM_VIEWS,
                'available_distinct_captures': 1,
                'input_measured_points': int(len(
                    consensus_view_surfaces[0]['points'])),
                'same_capture_points_averaged_together': False,
                'immutable_source_points_modified': False,
            }
        else:
            consensus_points, consensus_normals, consensus_support, \
                consensus_diagnostics = robust_cross_capture_consensus(
                    consensus_view_surfaces, voxel_length_m=voxel_length)
            consensus_diagnostics['available'] = True
            consensus_cloud = o3d.geometry.PointCloud()
            consensus_cloud.points = o3d.utility.Vector3dVector(
                consensus_points)
            consensus_cloud.normals = o3d.utility.Vector3dVector(
                consensus_normals)
            maximum_support = max(
                CONSENSUS_MINIMUM_VIEWS, int(np.max(consensus_support)))
            support_fraction = (
                (consensus_support.astype(float) - CONSENSUS_MINIMUM_VIEWS)
                / max(1, maximum_support - CONSENSUS_MINIMUM_VIEWS))
            consensus_colors = np.column_stack((
                0.10 + 0.15 * support_fraction,
                0.45 + 0.45 * support_fraction,
                0.95 - 0.65 * support_fraction,
            ))
            consensus_cloud.colors = o3d.utility.Vector3dVector(
                consensus_colors)
            consensus_cloud_path = _atomic_write_cloud(
                o3d, consensus_cloud,
                output.with_name(output.stem + '.consensus_points.ply'))
    textured_mesh = None
    if registration_mode == 'constrained_superposition':
        textured_mesh = _texture.build_textured_mesh(
            output.with_name(output.stem + '.textured.obj'),
            texture_frames, consensus_points, voxel_length)
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
        median_rmse,
        raw_metrics['dominant_component_triangle_ratio'],
        usable_mesh=(
            component_filter['connectivity_valid']
            and raw_mesh_size_qualified
            and cleaned_mesh_size_qualified))
    dimension_quality = None
    overall_quality = structural
    if raw_metrics['dimension_check'] is not None:
        dimension_quality = dimension_classification(
            raw_metrics['dimension_check']['maximum_absolute_error_m'])
        if dimension_quality == 'POOR':
            overall_quality = 'FAIL'
        elif dimension_quality == 'WARN' and overall_quality == 'PASS':
            overall_quality = 'WARN'
    capture_set = capture_set_provenance(
        manifest, allow_partial_view_set=allow_partial_view_set)
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
        'semantic_mask_source': (
            'offline_resegment'
            if offline_mask_context is not None else 'captured'),
        'connected_target_component_filter': {
            'minimum_relative_surface_area':
                MINIMUM_SIGNIFICANT_COMPONENT_AREA_FRACTION,
            'substantial_components_are_never_discarded': True,
        },
    }
    if offline_mask_context is not None:
        config['offline_resegment'] = {
            'index_path': str(offline_mask_context['index_path']),
            'generation_sha256': str(
                offline_mask_context['index']['generation_sha256']),
            'source_captures_immutable': True,
            'live_mask_used_as_model_fallback': False,
            'combination_policy':
                'intersection_with_captured_confidence_qualified_support',
        }
    if multiway_diagnostics is not None:
        config['multiway_registration'] = {
            'maximum_prior_neighbors': MULTIWAY_MAX_PRIOR_NEIGHBORS,
            'maximum_view_angle_deg': MULTIWAY_MAX_VIEW_ANGLE_DEG,
            'maximum_correspondence_m': MULTIWAY_MAX_CORRESPONDENCE_M,
            'maximum_pose_translation_correction_m': 0.020,
            'maximum_pose_rotation_correction_deg': 5.0,
        }
        if registration_mode == 'scene_pose_graph':
            config['multiway_registration'].update({
                'registration_source':
                    'static_scene_rgbd_excluding_target',
                'scene_registration_voxel_m':
                    SCENE_REGISTRATION_VOXEL_M,
                'target_exclusion_radius_px':
                    SCENE_TARGET_EXCLUSION_RADIUS_PX,
                'target_tsdf_fusion_source':
                    'confidence_qualified_target_only_depth',
            })
        elif registration_mode == 'constrained_superposition':
            config['multiway_registration'].update({
                'registration_source':
                    'target_masked_depth_capture_zero_anchored_global_alignment',
                'fixed_reference_frame_index': 0,
                'maximum_pose_translation_correction_m': None,
                'translation_corrections_unbounded': True,
                'rotation_prior_sigma_deg':
                    SUPERPOSITION_ROTATION_PRIOR_SIGMA_DEG,
                'data_sigma_m': SUPERPOSITION_DATA_SIGMA_M,
                'huber_delta_m': SUPERPOSITION_HUBER_DELTA_M,
                'rotation_corrections_allowed': True,
                'gauge_policy': 'capture_zero_fixed_exactly',
                'correspondence_policy':
                    'mutual_nearest_normal_consistent',
                'maximum_pose_rotation_correction_deg':
                    SUPERPOSITION_MAX_ROTATION_DEG,
                'target_tsdf_fusion_source':
                    'confidence_qualified_target_only_depth',
                'cross_capture_consensus': {
                    'voxel_length_m': max(
                        CONSENSUS_MIN_VOXEL_M, float(voxel_length)),
                    'minimum_distinct_captures':
                        CONSENSUS_MINIMUM_VIEWS,
                    'maximum_normal_angle_deg':
                        CONSENSUS_NORMAL_ANGLE_DEG,
                    'estimator':
                        'median_MAD_outlier_rejection_then_equal_weight_mean',
                    'maximum_one_vote_per_capture_per_cell': True,
                },
            })
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
        'raw_mesh_path': str(raw_output),
        'raw_mesh_sha256': sha256_file(raw_output),
        'raw_vertex_count': len(extracted_mesh.vertices),
        'raw_triangle_count': len(extracted_mesh.triangles),
        'minimum_diagnostic_mesh_size': {
            'minimum_vertices': 100,
            'minimum_triangles': 100,
            'raw_mesh_qualified': bool(raw_mesh_size_qualified),
            'cleaned_mesh_qualified': bool(cleaned_mesh_size_qualified),
            'policy': (
                'nonempty undersized meshes are retained for diagnostic '
                'inspection but force structural quality FAIL'),
        },
        'input_cloud_path': str(input_cloud_path),
        'input_cloud_sha256': sha256_file(input_cloud_path),
        'measured_cloud_path': str(measured_cloud_path),
        'measured_cloud_sha256': sha256_file(measured_cloud_path),
        'measured_point_count': len(measured_cloud.points),
        'measured_cloud_views': measured_cloud_views,
        'measured_cloud_semantics': (
            'all accepted per-view temporally averaged target depth points '
            'after the selected camera-pose refinement; no display '
            'downsampling; colors identify capture index'),
        'consensus_cloud_path': (
            str(consensus_cloud_path) if consensus_cloud_path else ''),
        'consensus_cloud_sha256': (
            sha256_file(consensus_cloud_path)
            if consensus_cloud_path else ''),
        'consensus_point_count': (
            len(consensus_cloud.points) if consensus_cloud is not None else 0),
        'cross_capture_consensus': consensus_diagnostics,
        'consensus_cloud_semantics': (
            'constrained-superposition output only: one representative per '
            'logical viewpoint and spatial correspondence cell; '
            'normal-incompatible and median/MAD positional outliers are '
            'rejected before an equal-weight cross-view mean; colors encode '
            'distinct-view support. The TSDF mesh remains a separate surface '
            'fusion output.' if consensus_cloud is not None else ''),
        'textured_mesh_path': (
            textured_mesh['mesh_path'] if textured_mesh else ''),
        'textured_mesh_sha256': (
            sha256_file(textured_mesh['mesh_path']) if textured_mesh else ''),
        'texture_material_path': (
            textured_mesh['material_path'] if textured_mesh else ''),
        'texture_material_sha256': (
            sha256_file(textured_mesh['material_path'])
            if textured_mesh else ''),
        'texture_atlas_path': (
            textured_mesh['texture_path'] if textured_mesh else ''),
        'texture_atlas_sha256': (
            sha256_file(textured_mesh['texture_path'])
            if textured_mesh else ''),
        'texture_baking': textured_mesh,
        'textured_mesh_semantics': (
            'all corrected confidence-qualified measured depth pixels '
            'retained near cross-capture consensus where available, '
            'triangulated in their source depth grids and fine-voxel fused, '
            'with one depth-consistent front-facing rectified source RGB '
            'selected per triangle and baked into an OBJ/MTL/PNG atlas'
            if textured_mesh else ''),
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
        'component_filter': component_filter,
        'raw_mesh_metrics': raw_metrics,
        'mesh_metrics': metrics,
        'structural_quality': structural,
        'dimension_quality': dimension_quality,
        'overall_quality': overall_quality,
        'overall_quality_is_provisional': bool(
            raw_metrics['dimension_check'] is not None),
        'visual_review_required': True,
        'frame_inputs': frame_inputs,
    }
    _atomic_write_json(output.with_suffix(output.suffix + '.quality.json'), report)
    return report


def _quality_rank(value):
    return {'FAIL': 0, 'WARN': 1, 'PASS': 2}.get(str(value), -1)


def _registration_component_coherence(report):
    """Use pre-cleanup coherence so fragment removal cannot launder quality."""
    metrics = report.get('raw_mesh_metrics', report['mesh_metrics'])
    return float(metrics['dominant_component_triangle_ratio'])


def _registration_metrics(report):
    """Return authoritative pre-cleanup reconstruction evidence."""
    return report.get('raw_mesh_metrics', report['mesh_metrics'])


def select_registration_report(robot_report, gicp_report):
    robot_metrics = _registration_metrics(robot_report)
    gicp_metrics = _registration_metrics(gicp_report)
    robot_residual = float(robot_metrics[
        'point_to_mesh_residual']['median_m'])
    gicp_residual = float(gicp_metrics[
        'point_to_mesh_residual']['median_m'])
    residual_improvement = (
        (robot_residual - gicp_residual) / robot_residual
        if np.isfinite(robot_residual) and robot_residual > 0.0 else 0.0)
    robot_dimension = robot_metrics.get('dimension_check')
    gicp_dimension = gicp_metrics.get('dimension_check')
    dimension_ok = True
    if robot_dimension and gicp_dimension:
        dimension_ok = (
            float(gicp_dimension['mean_absolute_error_m'])
            <= float(robot_dimension['mean_absolute_error_m']) + 0.002)
    robot_component = _registration_component_coherence(robot_report)
    gicp_component = _registration_component_coherence(gicp_report)
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
    robot_metrics = _registration_metrics(robot_report)
    candidate_metrics = _registration_metrics(candidate_report)
    robot_residual = float(
        robot_metrics['point_to_mesh_residual']['median_m'])
    candidate_residual = float(
        candidate_metrics['point_to_mesh_residual']['median_m'])
    residual_improvement = (
        (robot_residual - candidate_residual) / robot_residual
        if np.isfinite(robot_residual) and robot_residual > 0.0 else 0.0)
    robot_component = _registration_component_coherence(robot_report)
    candidate_component = _registration_component_coherence(candidate_report)
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
            eligible.append((
                _registration_component_coherence(report),
                -float(_registration_metrics(report)[
                    'point_to_mesh_residual']['median_m']),
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
                allow_partial_view_set=False, mask_source='captured'):
    scan = Path(scan_dir).resolve()
    with open(scan / 'manifest.json', 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest_sha256 = validate_manifest_integrity(scan, manifest)
    source = str(mask_source)
    if source not in MASK_SOURCES:
        raise ValueError('mask source must be one of: %s' % ', '.join(
            MASK_SOURCES))
    offline_mask_context = (
        prepare_offline_mask_context(scan, manifest_sha256)
        if source == 'offline_resegment' else None)
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
            manifest_sha256, allow_partial_view_set,
            offline_mask_context)
    reports = {}
    errors = {}
    for candidate_mode in (
            'robot_pose', 'bounded_gicp', 'multiway_gicp',
            'constrained_superposition',
            'scene_pose_graph'):
        candidate_output = output.with_name(
            output.stem + '.' + candidate_mode + output.suffix)
        try:
            reports[candidate_mode] = _reconstruct_single(
                scan, manifest, metadata_paths, metadata_values,
                candidate_output, candidate_mode, voxel_length, sdf_trunc,
                depth_trunc, dimensions, provenance, manifest_sha256,
                allow_partial_view_set, offline_mask_context)
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
    raw_output = raw_mesh_output_path(output)
    raw_temporary = raw_output.with_name(raw_output.name + '.partial')
    shutil.copy2(selected['raw_mesh_path'], raw_temporary)
    raw_temporary.replace(raw_output)
    selected.update({
        'registration_mode': selected_mode,
        'mesh_path': str(output),
        'mesh_sha256': sha256_file(output),
        'raw_mesh_path': str(raw_output),
        'raw_mesh_sha256': sha256_file(raw_output),
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
        '--mask-source', choices=MASK_SOURCES, default='captured',
        help=(
            'Use the captured live mask, or run/reuse a fresh offline '
            'GroundingDINO/SAM2 mask that may only narrow captured support.'))
    parser.add_argument(
        '--allow-partial-view-set', action='store_true',
        help=(
            'Backward-compatible no-op; 1-24 immutable captures are admitted '
            'and judged by the normal geometric quality gates.'))
    args = parser.parse_args()
    dimensions = (
        np.asarray(args.expected_dimensions_mm, dtype=float) / 1000.0
        if args.expected_dimensions_mm else None)
    print(json.dumps(reconstruct(
        args.scan_dir, args.output, args.voxel_length, args.sdf_trunc,
        args.depth_trunc, args.registration_mode, dimensions,
        args.allow_missing_calibration_id,
        args.allow_partial_view_set, args.mask_source),
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
