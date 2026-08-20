"""Live object-centric coverage and deterministic next-best-view scoring."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
SURFACE = np.uint8(2)


@dataclass(frozen=True)
class VoxelCoverageConfig:
    """Numerical policy for the planning-only live voxel representation."""

    voxel_size_m: float = 0.005
    minimum_radius_m: float = 0.030
    maximum_radius_m: float = 0.250
    radius_scale: float = 2.0
    padding_voxels: int = 2
    surface_tolerance_m: float = 0.007
    render_width: int = 64
    render_height: int = 48
    maximum_scoring_voxels: int = 20000

    def validate(self):
        values = (
            self.voxel_size_m,
            self.minimum_radius_m,
            self.maximum_radius_m,
            self.radius_scale,
            self.surface_tolerance_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                'voxel coverage values must be finite and positive')
        if self.minimum_radius_m > self.maximum_radius_m:
            raise ValueError('minimum voxel radius exceeds maximum radius')
        if int(self.padding_voxels) < 0:
            raise ValueError('voxel padding must be nonnegative')
        if min(int(self.render_width), int(self.render_height)) < 8:
            raise ValueError('voxel render dimensions are too small')
        if int(self.maximum_scoring_voxels) < 1000:
            raise ValueError('maximum scoring voxels must be at least 1000')


@dataclass(frozen=True)
class CoverageSnapshot:
    """Immutable snapshot produced after accepted RGB-D persistence."""

    session_id: str
    generation: int
    target_center: tuple
    radius_m: float
    voxel_size_m: float
    states: np.ndarray
    surface_view_bits: np.ndarray
    voxel_centers: np.ndarray
    view_directions: tuple
    tan_half_fov_x: float
    tan_half_fov_y: float
    render_width: int
    render_height: int
    maximum_scoring_voxels: int

    def __post_init__(self):
        for value in ('states', 'surface_view_bits', 'voxel_centers'):
            array = np.array(getattr(self, value), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, value, array)

    @property
    def unknown_voxels(self):
        return int(np.count_nonzero(self.states == UNKNOWN))

    @property
    def surface_voxels(self):
        return int(np.count_nonzero(self.states == SURFACE))


def _vector3(value, label):
    if isinstance(value, dict):
        value = [value.get(axis) for axis in ('x', 'y', 'z')]
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('%s must contain three finite values' % label)
    return result


def _inside(root, value):
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if root != path and root not in path.parents:
        # Capture metadata intentionally records an absolute provenance path.
        # A whole scan directory remains replayable after being archived or
        # copied by accepting only an existing same-basename frame artifact.
        relocated = (root / 'frames' / path.name).resolve()
        if relocated.is_file() and root in relocated.parents:
            return relocated
        raise ValueError('NBV capture artifact escapes the dataset root')
    return path


def _depth_metres(array, encoding='16UC1'):
    result = np.asarray(array, dtype=np.float64)
    if '16U' in str(encoding) or str(encoding) in ('mono16', '16UC1'):
        result *= 0.001
    result[~np.isfinite(result)] = 0.0
    return result


def _camera_geometry(metadata):
    matrix = np.asarray(
        metadata['camera_transform']['matrix_4x4'], dtype=np.float64)
    intrinsic = np.asarray(metadata['camera_info']['k'], dtype=np.float64)
    width = int(metadata['camera_info']['width'])
    height = int(metadata['camera_info']['height'])
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('NBV camera transform is invalid')
    if intrinsic.shape != (9,) or not np.all(np.isfinite(intrinsic)):
        raise ValueError('NBV camera intrinsics are invalid')
    if min(intrinsic[0], intrinsic[4], width, height) <= 0:
        raise ValueError('NBV camera dimensions are invalid')
    return matrix, intrinsic, width, height


def _frame_arrays(dataset, metadata):
    target_depth_path = _inside(
        dataset, metadata['target_depth_png_file_path'])
    support_path = _inside(
        dataset, metadata['target_support_mask_file_path'])
    aligned_path = _inside(dataset, metadata['depth_file_path'])
    target_depth = cv2.imread(str(target_depth_path), cv2.IMREAD_UNCHANGED)
    support = cv2.imread(str(support_path), cv2.IMREAD_GRAYSCALE)
    aligned = np.load(str(aligned_path), allow_pickle=False)
    if target_depth is None or support is None:
        raise ValueError('NBV target depth/support artifact is unreadable')
    if target_depth.shape != support.shape or aligned.shape != support.shape:
        raise ValueError('NBV capture arrays have inconsistent dimensions')
    target_depth = _depth_metres(target_depth, '16UC1')
    aligned = _depth_metres(aligned, metadata.get('depth_encoding', '16UC1'))
    return target_depth, support > 0, aligned


def _target_points(depth, support, intrinsic, base_camera):
    rows, cols = np.nonzero(
        support & np.isfinite(depth) & (depth > 0.10) & (depth < 1.50))
    if not len(rows):
        raise ValueError('NBV capture contains no supported target depth')
    z = depth[rows, cols]
    fx, fy, cx, cy = (
        intrinsic[0], intrinsic[4], intrinsic[2], intrinsic[5])
    camera = np.column_stack((
        (cols - cx) * z / fx,
        (rows - cy) * z / fy,
        z,
        np.ones_like(z),
    ))
    return camera.dot(base_camera.T)[:, :3]


def direction_bin(direction):
    """Return one of 32 deterministic azimuth/elevation observation bins."""
    vector = _vector3(direction, 'view direction')
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError('view direction has zero length')
    vector /= norm
    azimuth = math.atan2(float(vector[1]), float(vector[0])) % (2.0 * math.pi)
    elevation = math.asin(max(-1.0, min(1.0, float(vector[2]))))
    azimuth_bin = min(7, int(azimuth / (2.0 * math.pi) * 8.0))
    elevation_bin = min(3, max(0, int(
        (elevation + 0.5 * math.pi) / math.pi * 4.0)))
    return elevation_bin * 8 + azimuth_bin


class ObjectCoverageModel:
    """Session-scoped confidence-qualified visual-hull approximation."""

    def __init__(self, config=None):
        self.config = config or VoxelCoverageConfig()
        self.config.validate()
        self.reset()

    def reset(self, session_id=''):
        self.session_id = str(session_id)
        self.generation = 0
        self.target_center = None
        self.radius_m = 0.0
        self.voxel_centers = None
        self.inside_object_envelope = None
        self.surface_hits = None
        self.free_hits = None
        self.surface_view_bits = None
        self.view_directions = []
        self.tan_half_fov_x = 1.0
        self.tan_half_fov_y = 1.0

    def _initialize(self, points, target_center, intrinsic, width, height):
        center = _vector3(target_center, 'coverage target center')
        distances = np.linalg.norm(points - center, axis=1)
        observed = float(np.percentile(distances, 95.0))
        radius = (
            self.config.radius_scale * observed
            + self.config.padding_voxels * self.config.voxel_size_m)
        radius = min(
            self.config.maximum_radius_m,
            max(self.config.minimum_radius_m, radius))
        steps = int(math.ceil(2.0 * radius / self.config.voxel_size_m)) + 1
        axis = (
            np.arange(steps, dtype=np.float64) * self.config.voxel_size_m
            - 0.5 * (steps - 1) * self.config.voxel_size_m)
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing='ij')
        offsets = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        self.target_center = center
        self.radius_m = radius
        self.voxel_centers = offsets + center
        self.inside_object_envelope = (
            np.linalg.norm(offsets, axis=1) <= radius + 1e-9)
        count = len(self.voxel_centers)
        self.surface_hits = np.zeros(count, dtype=np.uint16)
        self.free_hits = np.zeros(count, dtype=np.uint16)
        self.surface_view_bits = np.zeros(count, dtype=np.uint32)
        self.free_hits[~self.inside_object_envelope] = 1
        self.tan_half_fov_x = float(width) / (2.0 * float(intrinsic[0]))
        self.tan_half_fov_y = float(height) / (2.0 * float(intrinsic[4]))

    def integrate(
            self, target_depth_m, support_mask, aligned_depth_m, intrinsic,
            base_camera, target_center):
        """Integrate one accepted target observation into the live model."""
        target_depth = np.asarray(target_depth_m, dtype=np.float64)
        support = np.asarray(support_mask, dtype=bool)
        aligned = np.asarray(aligned_depth_m, dtype=np.float64)
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        transform = np.asarray(base_camera, dtype=np.float64)
        if target_depth.ndim != 2 or support.shape != target_depth.shape \
                or aligned.shape != target_depth.shape:
            raise ValueError('NBV integration images are inconsistent')
        if intrinsic.shape != (9,) or transform.shape != (4, 4):
            raise ValueError('NBV integration calibration is invalid')
        height, width = target_depth.shape
        points = _target_points(
            target_depth, support, intrinsic, transform)
        center = _vector3(target_center, 'coverage target center')
        if self.voxel_centers is None:
            self._initialize(points, center, intrinsic, width, height)
        elif float(np.linalg.norm(center - self.target_center)) > 1e-9:
            raise ValueError(
                'NBV coverage target center changed within session')

        camera_base = np.linalg.inv(transform)
        homogeneous = np.column_stack((
            self.voxel_centers,
            np.ones(len(self.voxel_centers), dtype=np.float64)))
        camera = homogeneous.dot(camera_base.T)[:, :3]
        z = camera[:, 2]
        valid = z > 1e-6
        indexes = np.flatnonzero(valid)
        if not len(indexes):
            raise ValueError('NBV object envelope is behind the camera')
        fx, fy, cx, cy = (
            intrinsic[0], intrinsic[4], intrinsic[2], intrinsic[5])
        u = np.rint(fx * camera[indexes, 0] / z[indexes] + cx).astype(int)
        v = np.rint(fy * camera[indexes, 1] / z[indexes] + cy).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indexes = indexes[inside]
        u = u[inside]
        v = v[inside]
        voxel_depth = z[indexes]
        pixel_support = support[v, u]
        target_measurement = target_depth[v, u]
        scene_measurement = aligned[v, u]
        tolerance = self.config.surface_tolerance_m

        target_valid = pixel_support & (target_measurement > 0.10)
        free = target_valid & (voxel_depth < target_measurement - tolerance)
        surface = target_valid & (
            np.abs(voxel_depth - target_measurement) <= tolerance)
        background_valid = (~pixel_support) & (scene_measurement > 0.10)
        free |= background_valid & (
            voxel_depth < scene_measurement - tolerance)
        np.add.at(self.free_hits, indexes[free], 1)
        np.add.at(self.surface_hits, indexes[surface], 1)

        camera_position = transform[:3, 3]
        view = camera_position - self.target_center
        view /= np.linalg.norm(view)
        bit = np.uint32(1 << direction_bin(view))
        np.bitwise_or.at(self.surface_view_bits, indexes[surface], bit)
        self.view_directions.append(tuple(float(value) for value in view))
        self.generation += 1

    def states(self):
        if self.voxel_centers is None:
            return np.empty(0, dtype=np.uint8)
        result = np.full(len(self.voxel_centers), UNKNOWN, dtype=np.uint8)
        result[self.free_hits > 0] = FREE
        result[self.surface_hits > 0] = SURFACE
        result[~self.inside_object_envelope] = FREE
        return result

    def snapshot(self):
        if self.voxel_centers is None:
            raise ValueError('NBV coverage model is empty')
        return CoverageSnapshot(
            session_id=self.session_id,
            generation=self.generation,
            target_center=tuple(float(value) for value in self.target_center),
            radius_m=float(self.radius_m),
            voxel_size_m=float(self.config.voxel_size_m),
            states=self.states(),
            surface_view_bits=self.surface_view_bits,
            voxel_centers=self.voxel_centers,
            view_directions=tuple(self.view_directions),
            tan_half_fov_x=float(self.tan_half_fov_x),
            tan_half_fov_y=float(self.tan_half_fov_y),
            render_width=int(self.config.render_width),
            render_height=int(self.config.render_height),
            maximum_scoring_voxels=int(
                self.config.maximum_scoring_voxels),
        )

    def rebuild_from_scan(
            self, scan_dir, accepted_views, target_center, session_id):
        """Rebuild exactly one accepted generation from committed artifacts."""
        expected = int(accepted_views)
        dataset = Path(str(scan_dir or '')).resolve()
        manifest_path = dataset / 'manifest.json'
        if expected <= 0:
            self.reset(session_id)
            return None
        if not manifest_path.is_file():
            raise ValueError('NBV scan manifest is unavailable')
        with manifest_path.open('r', encoding='utf-8') as stream:
            manifest = json.load(stream)
        if int(manifest.get('capture_count', -1)) < expected:
            raise ValueError('NBV capture generation is still catching up')
        records = sorted(
            (item for item in manifest.get('files', [])
             if str(item.get('path', '')).endswith('_metadata.yaml')),
            key=lambda item: str(item.get('path', '')))
        if len(records) < expected:
            raise ValueError('NBV frame metadata generation is incomplete')
        self.reset(session_id)
        for record in records[:expected]:
            metadata_path = _inside(dataset, record['path'])
            with metadata_path.open('r', encoding='utf-8') as stream:
                metadata = yaml.safe_load(stream) or {}
            transform, intrinsic, _width, _height = _camera_geometry(metadata)
            target_depth, support, aligned = _frame_arrays(dataset, metadata)
            self.integrate(
                target_depth, support, aligned, intrinsic, transform,
                target_center)
        return self.snapshot()


def _camera_basis(camera_position, target_center, look_direction=None):
    camera = _vector3(camera_position, 'candidate camera position')
    center = _vector3(target_center, 'coverage target center')
    forward = (
        center - camera
        if look_direction is None
        else _vector3(look_direction, 'candidate look direction'))
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        raise ValueError('candidate camera coincides with coverage target')
    forward /= norm
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return camera, right, down, forward


def candidate_information(
        snapshot, camera_position, current_camera=None,
        look_direction=None):
    """Predict normalized marginal target information for one camera ray."""
    camera, right, down, forward = _camera_basis(
        camera_position, snapshot.target_center, look_direction)
    surface_indexes = np.flatnonzero(snapshot.states == SURFACE)
    unknown_indexes = np.flatnonzero(snapshot.states == UNKNOWN)
    unknown_budget = max(
        0, int(snapshot.maximum_scoring_voxels) - len(surface_indexes))
    if len(unknown_indexes) > unknown_budget:
        sample = np.linspace(
            0, len(unknown_indexes) - 1, unknown_budget,
            dtype=np.int64)
        unknown_indexes = unknown_indexes[sample]
    indexes = np.concatenate((surface_indexes, unknown_indexes))
    relative = snapshot.voxel_centers[indexes] - camera
    z = relative.dot(forward)
    valid = z > 1e-6
    indexes = indexes[valid]
    relative = relative[valid]
    z = z[valid]
    normalized_x = relative.dot(right) / z
    normalized_y = relative.dot(down) / z
    valid = (
        (np.abs(normalized_x) <= snapshot.tan_half_fov_x)
        & (np.abs(normalized_y) <= snapshot.tan_half_fov_y))
    indexes = indexes[valid]
    z = z[valid]
    normalized_x = normalized_x[valid]
    normalized_y = normalized_y[valid]
    if not len(indexes):
        return {
            'predicted_unknown_pixels': 0,
            'novel_surface_pixels': 0,
            'marginal_information_pixels': 0,
            'marginal_information_fraction': 0.0,
            'projected_object_pixels': 0,
            'direction_novelty_deg': 0.0,
            'camera_travel_m': math.inf,
            'positive_information_gain': False,
        }
    width = int(snapshot.render_width)
    height = int(snapshot.render_height)
    u = np.floor(
        (normalized_x / snapshot.tan_half_fov_x + 1.0)
        * 0.5 * width).astype(int)
    v = np.floor(
        (normalized_y / snapshot.tan_half_fov_y + 1.0)
        * 0.5 * height).astype(int)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    pixels = v * width + u
    order = np.lexsort((z, pixels))
    ordered_pixels = pixels[order]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    visible_indexes = indexes[order[first]]
    visible_states = snapshot.states[visible_indexes]
    unknown = int(np.count_nonzero(visible_states == UNKNOWN))
    surface = visible_states == SURFACE

    camera_direction = camera - np.asarray(snapshot.target_center, dtype=float)
    camera_direction /= np.linalg.norm(camera_direction)
    bit = np.uint32(1 << direction_bin(camera_direction))
    novel_surface = int(np.count_nonzero(
        surface & ((snapshot.surface_view_bits[visible_indexes] & bit) == 0)))
    projected = int(len(visible_indexes))
    marginal = int(unknown + novel_surface)
    marginal_fraction = (
        float(marginal) / float(projected) if projected > 0 else 0.0)
    novelty = 180.0
    if snapshot.view_directions:
        separations = []
        for previous in snapshot.view_directions:
            cosine = float(np.dot(camera_direction, np.asarray(previous)))
            separations.append(math.degrees(math.acos(
                max(-1.0, min(1.0, cosine)))))
        novelty = min(separations)
    travel = (
        float(np.linalg.norm(camera - _vector3(
            current_camera, 'current camera position')))
        if current_camera is not None else 0.0)
    return {
        'predicted_unknown_pixels': unknown,
        'novel_surface_pixels': novel_surface,
        'marginal_information_pixels': marginal,
        'marginal_information_fraction': marginal_fraction,
        'projected_object_pixels': projected,
        'direction_novelty_deg': float(novelty),
        'camera_travel_m': float(travel),
        'positive_information_gain': bool(unknown > 0 or novel_surface > 0),
    }


def rank_next_best_views(snapshot, viewpoints, current_camera=None):
    """
    Return candidates in information-first, direction-diverse order.

    The generator deliberately provides several radii for each viewing ray so
    Tesseract can escape a local IK branch.  Voxel visibility is a directional
    question, so each direction is raycast once and its radial alternatives
    retain that result.  Round-robin flattening tries one candidate from every
    informative direction before trying second-radius fallbacks.
    """
    center = np.asarray(snapshot.target_center, dtype=np.float64)
    groups = {}
    for viewpoint in viewpoints:
        candidate = dict(viewpoint)
        position = _vector3(
            candidate.get('desired_camera_position'),
            'candidate camera position')
        direction = position - center
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-9:
            raise ValueError('candidate camera coincides with coverage target')
        direction /= distance
        key = tuple(float(value) for value in np.round(direction, 6))
        candidate['_nbv_distance_m'] = distance
        candidate['_nbv_position'] = position
        groups.setdefault(key, []).append(candidate)

    scored_groups = []
    for key, alternatives in groups.items():
        # The nearest permitted radius has the greatest projected resolution.
        representative = min(
            alternatives,
            key=lambda item: (
                float(item['_nbv_distance_m']), int(item.get('index', 0))))
        metrics = candidate_information(
            snapshot, representative['_nbv_position'], current_camera,
            representative.get('desired_look_at_direction'))
        for item in alternatives:
            travel = (
                float(np.linalg.norm(
                    item['_nbv_position'] - _vector3(
                        current_camera, 'current camera position')))
                if current_camera is not None else 0.0)
            item.update({
                'nbv_model_generation': int(snapshot.generation),
                'nbv_predicted_unknown_pixels': int(
                    metrics['predicted_unknown_pixels']),
                'nbv_novel_surface_pixels': int(
                    metrics['novel_surface_pixels']),
                'nbv_marginal_information_pixels': int(
                    metrics['marginal_information_pixels']),
                'nbv_marginal_information_fraction': float(
                    metrics['marginal_information_fraction']),
                'nbv_projected_object_pixels': int(
                    metrics['projected_object_pixels']),
                'nbv_direction_novelty_deg': float(
                    metrics['direction_novelty_deg']),
                'nbv_camera_travel_m': travel,
                'nbv_positive_information_gain': bool(
                    metrics['positive_information_gain']),
            })
        alternatives.sort(key=lambda item: (
            float(item['nbv_camera_travel_m']),
            float(item['_nbv_distance_m']),
            int(item.get('index', 0)),
        ))
        scored_groups.append((key, alternatives, metrics))
    # Compare the fraction of the projected target that is new before its raw
    # pixel count. Raw counts systematically favored steep views in the live
    # L515 replay because those rays projected more envelope voxels, even when
    # they repeated the same accepted surface sector. Accepted-view angular
    # novelty resolves equal fractions; absolute information and travel remain
    # later tie-breaks rather than a fixed movement schedule.
    scored_groups.sort(key=lambda group: (
        not bool(group[2]['positive_information_gain']),
        -float(group[2]['marginal_information_fraction']),
        -float(group[2]['direction_novelty_deg']),
        -int(group[2]['marginal_information_pixels']),
        -int(group[2]['novel_surface_pixels']),
        -int(group[2]['predicted_unknown_pixels']),
        min(float(item['nbv_camera_travel_m']) for item in group[1]),
        group[0],
    ))
    scored = []
    for alternative_index in range(max(
            (len(group[1]) for group in scored_groups), default=0)):
        for _key, alternatives, _metrics in scored_groups:
            if alternative_index < len(alternatives):
                scored.append(alternatives[alternative_index])
    total = len(scored)
    for rank, item in enumerate(scored, 1):
        item['nbv_rank'] = rank
        # The Tesseract bridge consumes one scalar. Rank is assigned only after
        # the full lexicographic information decision, so motion cost can never
        # numerically overpower information gain downstream.
        item['nbv_rank_score'] = float(total - rank + 1)
        item.pop('_nbv_distance_m', None)
        item.pop('_nbv_position', None)
    return scored
