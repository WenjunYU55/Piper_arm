"""Compact, command-free camera capability-map storage and lookup."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


CAPABILITY_MAP_SCHEMA_VERSION = 1
POSITION_INDEX_BITS = 11
POSITION_INDEX_OFFSET = 1 << (POSITION_INDEX_BITS - 1)
POSITION_INDEX_MASK = (1 << POSITION_INDEX_BITS) - 1
DIRECTION_INDEX_BITS = 8
DIRECTION_INDEX_MASK = (1 << DIRECTION_INDEX_BITS) - 1
DEFAULT_POSITION_VOXEL_M = 0.020
DEFAULT_DIRECTION_BIN_DEG = 10.0
DEFAULT_DIRECTION_TOLERANCE_DEG = 15.0
DEFAULT_SPATIAL_DILATION_CELLS = 1


def sha256_file(path):
    """Return the SHA-256 identity of one source artifact."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError('%s must contain three finite values' % label)
    return array


def _direction_bin_shape(bin_degrees):
    width = float(bin_degrees)
    if (
            not math.isfinite(width) or width <= 0.0
            or abs(round(360.0 / width) * width - 360.0) > 1e-9
            or abs(round(180.0 / width) * width - 180.0) > 1e-9):
        raise ValueError(
            'direction bin size must evenly divide 180 and 360 degrees')
    azimuth_bins = int(round(360.0 / width))
    elevation_bins = int(round(180.0 / width))
    if (
            azimuth_bins > DIRECTION_INDEX_MASK
            or elevation_bins > DIRECTION_INDEX_MASK):
        raise ValueError('direction bin count exceeds packed-key capacity')
    return azimuth_bins, elevation_bins


def position_indices(position, voxel_size_m):
    """Quantize one base-frame camera position."""
    vector = _finite_vector(position, 'camera position')
    voxel = float(voxel_size_m)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise ValueError('position voxel size must be positive and finite')
    indices = np.floor(vector / voxel).astype(np.int64)
    if np.any(indices < -POSITION_INDEX_OFFSET) or np.any(
            indices >= POSITION_INDEX_OFFSET):
        raise ValueError('camera position exceeds packed capability-map range')
    return indices


def direction_indices(direction, bin_degrees):
    """Quantize one camera optical-axis direction into azimuth/elevation."""
    vector = _finite_vector(direction, 'camera optical direction')
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError('camera optical direction is zero')
    vector /= norm
    width = float(bin_degrees)
    azimuth_bins, elevation_bins = _direction_bin_shape(width)
    azimuth = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    elevation = math.degrees(math.asin(float(np.clip(vector[2], -1.0, 1.0))))
    azimuth_index = min(
        azimuth_bins - 1, int(math.floor(azimuth / width)))
    elevation_index = min(
        elevation_bins - 1,
        max(0, int(math.floor((elevation + 90.0) / width))),
    )
    return np.asarray([azimuth_index, elevation_index], dtype=np.int64)


def direction_bin_center(indices, bin_degrees):
    """Return the unit-vector centre of one direction bin."""
    values = np.asarray(indices, dtype=np.int64)
    width = float(bin_degrees)
    azimuth_bins, elevation_bins = _direction_bin_shape(width)
    if (
            values.shape != (2,) or values[0] < 0
            or values[0] >= azimuth_bins or values[1] < 0
            or values[1] >= elevation_bins):
        raise ValueError('direction bin index is invalid')
    azimuth = math.radians((float(values[0]) + 0.5) * width)
    elevation = math.radians(-90.0 + (float(values[1]) + 0.5) * width)
    cosine = math.cos(elevation)
    return np.asarray([
        cosine * math.cos(azimuth),
        cosine * math.sin(azimuth),
        math.sin(elevation),
    ], dtype=float)


def pack_capability_key(position_index, direction_index):
    """Pack three signed position cells and two direction cells into uint64."""
    position = np.asarray(position_index, dtype=np.int64)
    direction = np.asarray(direction_index, dtype=np.int64)
    if position.shape != (3,) or direction.shape != (2,):
        raise ValueError(
            'capability key requires 3D position and 2D direction')
    if np.any(position < -POSITION_INDEX_OFFSET) or np.any(
            position >= POSITION_INDEX_OFFSET):
        raise ValueError('capability position index exceeds packed range')
    if np.any(direction < 0) or np.any(direction > DIRECTION_INDEX_MASK):
        raise ValueError('capability direction index exceeds packed range')
    values = [int(value + POSITION_INDEX_OFFSET) for value in position]
    key = values[0]
    key |= values[1] << POSITION_INDEX_BITS
    key |= values[2] << (2 * POSITION_INDEX_BITS)
    key |= int(direction[0]) << (3 * POSITION_INDEX_BITS)
    key |= int(direction[1]) << (
        3 * POSITION_INDEX_BITS + DIRECTION_INDEX_BITS)
    return np.uint64(key)


def unpack_capability_key(key):
    """Return integer XYZ and direction indices from a packed key."""
    value = int(key)
    position = []
    for index in range(3):
        encoded = (
            value >> (index * POSITION_INDEX_BITS)) & POSITION_INDEX_MASK
        position.append(encoded - POSITION_INDEX_OFFSET)
    azimuth = (
        value >> (3 * POSITION_INDEX_BITS)) & DIRECTION_INDEX_MASK
    elevation = (
        value >> (3 * POSITION_INDEX_BITS + DIRECTION_INDEX_BITS)
    ) & DIRECTION_INDEX_MASK
    return np.asarray(position, dtype=np.int64), np.asarray(
        [azimuth, elevation], dtype=np.int64)


def capability_key(position, direction, voxel_size_m, direction_bin_deg):
    """Return the packed key for one camera pose."""
    return pack_capability_key(
        position_indices(position, voxel_size_m),
        direction_indices(direction, direction_bin_deg),
    )


@dataclass(frozen=True)
class CapabilityQuery:
    """Result of one cheap capability-map lookup."""

    supported: bool
    checked_keys: int
    matching_keys: int
    elapsed_ms: float
    reason: str
    sample_support: tuple = ()
    supported_intervals_m: tuple = ()


class CapabilityMap:
    """Immutable occupancy atlas used only for cheap prequalification."""

    def __init__(self, keys, maximum_tool_minimum_z_m, metadata):
        key_array = np.asarray(keys, dtype=np.uint64)
        floor_array = np.asarray(maximum_tool_minimum_z_m, dtype=np.float32)
        if (
                key_array.ndim != 1 or floor_array.shape != key_array.shape
                or len(key_array) == 0):
            raise ValueError('capability map arrays are empty or malformed')
        if np.any(key_array[1:] <= key_array[:-1]):
            raise ValueError('capability map keys must be sorted and unique')
        if not np.all(np.isfinite(floor_array)):
            raise ValueError('capability map floor metadata is non-finite')
        record = dict(metadata)
        if (
                int(record.get('schema_version', -1))
                != CAPABILITY_MAP_SCHEMA_VERSION):
            raise ValueError('capability map schema is unsupported')
        self.keys = key_array
        self.maximum_tool_minimum_z_m = floor_array
        self.metadata = record
        self.position_voxel_m = float(record['position_voxel_m'])
        self.direction_bin_deg = float(record['direction_bin_deg'])
        self.direction_tolerance_deg = float(record.get(
            'direction_tolerance_deg', DEFAULT_DIRECTION_TOLERANCE_DEG))
        self.spatial_dilation_cells = int(record.get(
            'spatial_dilation_cells', DEFAULT_SPATIAL_DILATION_CELLS))
        _direction_bin_shape(self.direction_bin_deg)
        if (
                not math.isfinite(self.position_voxel_m)
                or self.position_voxel_m <= 0.0
                or not math.isfinite(self.direction_tolerance_deg)
                or self.direction_tolerance_deg < 0.0
                or self.direction_tolerance_deg > 45.0
                or self.spatial_dilation_cells < 0
                or self.spatial_dilation_cells > 2):
            raise ValueError('capability map lookup policy is invalid')
        self._compatible_direction_bins = self._build_direction_lookup()
        dilation = range(
            -self.spatial_dilation_cells,
            self.spatial_dilation_cells + 1)
        self._spatial_offsets = np.asarray([
            [dx, dy, dz]
            for dx in dilation for dy in dilation for dz in dilation
        ], dtype=np.int64)

    def _build_direction_lookup(self):
        azimuth_bins, elevation_bins = _direction_bin_shape(
            self.direction_bin_deg)
        centres = []
        indices = []
        for elevation in range(elevation_bins):
            for azimuth in range(azimuth_bins):
                index = np.asarray([azimuth, elevation], dtype=np.int64)
                indices.append(index)
                centres.append(direction_bin_center(
                    index, self.direction_bin_deg))
        centres = np.asarray(centres, dtype=float)
        # Inflate by the bin's maximum half-diagonal. This makes boundary
        # quantization conservative: it may retain an impossible ray, but it
        # does not discard a nearby feasible sample before Tesseract.
        tolerance = math.radians(
            self.direction_tolerance_deg
            + math.sqrt(2.0) * 0.5 * self.direction_bin_deg)
        result = {}
        for query in indices:
            vector = direction_bin_center(query, self.direction_bin_deg)
            angles = np.arccos(np.clip(centres @ vector, -1.0, 1.0))
            compatible = np.asarray([
                indices[index]
                for index in np.flatnonzero(angles <= tolerance + 1e-12)
            ], dtype=np.uint64)
            packed = compatible[:, 0]
            packed |= compatible[:, 1] << np.uint64(DIRECTION_INDEX_BITS)
            packed <<= np.uint64(3 * POSITION_INDEX_BITS)
            result[tuple(int(value) for value in query)] = packed
        return result

    def _query_indices(self, positions, direction, floor_z_m, clearance_m):
        start = time.perf_counter()
        floor = float(floor_z_m)
        clearance = float(clearance_m)
        points = np.asarray(positions, dtype=float)
        if (
                points.ndim != 2 or points.shape[1] != 3
                or not np.all(np.isfinite(points))
                or not math.isfinite(floor) or not math.isfinite(clearance)
                or clearance < 0.0):
            return CapabilityQuery(
                False, 0, 0, (time.perf_counter() - start) * 1000.0,
                'capability query geometry is invalid')
        try:
            base_cells = np.floor(
                points / self.position_voxel_m).astype(np.int64)
            direction_index = direction_indices(
                direction, self.direction_bin_deg)
        except ValueError as error:
            return CapabilityQuery(
                False, 0, 0, (time.perf_counter() - start) * 1000.0,
                str(error))
        if np.any(base_cells < -POSITION_INDEX_OFFSET) or np.any(
                base_cells >= POSITION_INDEX_OFFSET):
            return CapabilityQuery(
                False, 0, 0, (time.perf_counter() - start) * 1000.0,
                'ray lies outside capability-map coordinate range')

        spatial = (
            base_cells[:, np.newaxis, :]
            + self._spatial_offsets[np.newaxis, :, :]
        )
        valid_spatial = np.all(
            (spatial >= -POSITION_INDEX_OFFSET)
            & (spatial < POSITION_INDEX_OFFSET), axis=2)
        direction_keys = self._compatible_direction_bins[
            tuple(int(value) for value in direction_index)]
        if not np.any(valid_spatial) or len(direction_keys) == 0:
            return CapabilityQuery(
                False, 0, 0, (time.perf_counter() - start) * 1000.0,
                'capability query produced no finite lookup cells',
                tuple(False for _item in points))
        # Preserve the source point axis so a ray query can recover exactly
        # which standoff samples have atlas support. Invalid neighbour cells
        # use a harmless zero key and are masked before matching.
        encoded = np.zeros(spatial.shape, dtype=np.uint64)
        encoded[valid_spatial] = (
            spatial[valid_spatial] + POSITION_INDEX_OFFSET).astype(np.uint64)
        position_keys = encoded[:, :, 0]
        position_keys |= encoded[:, :, 1] << np.uint64(POSITION_INDEX_BITS)
        position_keys |= encoded[:, :, 2] << np.uint64(
            2 * POSITION_INDEX_BITS)
        # Duplicate neighbour keys are harmless for searchsorted and cheaper
        # than sorting/uniquing every short mission ray.
        query_array = (
            position_keys[:, :, np.newaxis]
            | direction_keys[np.newaxis, np.newaxis, :])
        valid_queries = np.broadcast_to(
            valid_spatial[:, :, np.newaxis], query_array.shape).reshape(-1)
        query_array = query_array.reshape(-1)
        locations = np.searchsorted(self.keys, query_array)
        inside = (locations < len(self.keys)) & valid_queries
        matches = np.zeros(len(query_array), dtype=bool)
        matches[inside] = self.keys[locations[inside]] == query_array[inside]
        matching_locations = locations[matches]
        floor_ok = self.maximum_tool_minimum_z_m[matching_locations] >= (
            floor + clearance - 1e-9)
        qualified_matches = np.zeros(len(query_array), dtype=bool)
        qualified_matches[np.flatnonzero(matches)] = floor_ok
        sample_support = np.any(qualified_matches.reshape(
            (len(points), len(self._spatial_offsets), len(direction_keys))),
            axis=(1, 2))
        supported = bool(np.any(sample_support))
        return CapabilityQuery(
            supported,
            int(np.count_nonzero(valid_queries)),
            int(np.count_nonzero(floor_ok)),
            (time.perf_counter() - start) * 1000.0,
            '' if supported else (
                'no collision-qualified capability cell intersects ray'),
            tuple(bool(value) for value in sample_support),
        )

    def supports_pose(self, camera_position, look_direction,
                      floor_z_m=-1e6, clearance_m=0.0):
        """Check one achieved or synthetic camera pose."""
        position = _finite_vector(camera_position, 'camera position')
        return self._query_indices(
            position.reshape((1, 3)), look_direction,
            floor_z_m, clearance_m)

    def intersects_ray(self, target_center, ray_direction,
                       minimum_standoff_m, maximum_standoff_m,
                       floor_z_m, clearance_m):
        """Check the complete bounded target-centred ray interval."""
        target = _finite_vector(target_center, 'ray target center')
        direction = _finite_vector(ray_direction, 'ray direction')
        norm = float(np.linalg.norm(direction))
        minimum = float(minimum_standoff_m)
        maximum = float(maximum_standoff_m)
        if (
                norm <= 1e-12 or not math.isfinite(minimum)
                or not math.isfinite(maximum) or minimum <= 0.0
                or maximum < minimum):
            return CapabilityQuery(False, 0, 0, 0.0, 'ray interval is invalid')
        direction /= norm
        samples = max(
            1,
            int(math.ceil(
                (maximum - minimum) / self.position_voxel_m)) + 1,
        )
        standoffs = np.linspace(minimum, maximum, samples)
        positions = target.reshape((1, 3)) + np.outer(standoffs, direction)
        result = self._query_indices(
            positions, -direction, floor_z_m, clearance_m)
        support = np.asarray(result.sample_support, dtype=bool)
        intervals = []
        if len(support) == len(standoffs) and np.any(support):
            spacing = (
                float(standoffs[1] - standoffs[0])
                if len(standoffs) > 1 else self.position_voxel_m)
            indexes = np.flatnonzero(support)
            run_start = int(indexes[0])
            run_end = run_start
            for raw_index in indexes[1:]:
                index = int(raw_index)
                if index != run_end + 1:
                    intervals.append((
                        max(minimum, float(standoffs[run_start])
                            - 0.5 * spacing),
                        min(maximum, float(standoffs[run_end])
                            + 0.5 * spacing),
                    ))
                    run_start = index
                run_end = index
            intervals.append((
                max(minimum, float(standoffs[run_start]) - 0.5 * spacing),
                min(maximum, float(standoffs[run_end]) + 0.5 * spacing),
            ))
        return CapabilityQuery(
            result.supported, result.checked_keys, result.matching_keys,
            result.elapsed_ms, result.reason, result.sample_support,
            tuple(intervals))


def write_capability_map(path, keys, maximum_tool_minimum_z_m, metadata):
    """Write one compressed, pickle-free capability artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(metadata)
    record['schema_version'] = CAPABILITY_MAP_SCHEMA_VERSION
    payload = json.dumps(
        record, sort_keys=True, separators=(',', ':')).encode('utf-8')
    np.savez_compressed(
        str(output),
        keys=np.asarray(keys, dtype=np.uint64),
        maximum_tool_minimum_z_m=np.asarray(
            maximum_tool_minimum_z_m, dtype=np.float32),
        metadata_json=np.frombuffer(payload, dtype=np.uint8),
    )
    return output


def load_capability_map(path, project_root=None, verify_sources=True):
    """Load and optionally hash-validate one capability artifact."""
    artifact = Path(path)
    with np.load(str(artifact), allow_pickle=False) as archive:
        required = {
            'keys', 'maximum_tool_minimum_z_m', 'metadata_json'}
        if not required.issubset(archive.files):
            raise ValueError(
                'capability map archive is missing required arrays')
        metadata_bytes = np.asarray(
            archive['metadata_json'], dtype=np.uint8)
        metadata = json.loads(bytes(metadata_bytes).decode('utf-8'))
        result = CapabilityMap(
            np.array(archive['keys'], copy=True),
            np.array(archive['maximum_tool_minimum_z_m'], copy=True),
            metadata,
        )
    if verify_sources:
        root = Path(project_root).resolve()
        sources = result.metadata.get('source_sha256')
        if not isinstance(sources, dict) or not sources:
            raise ValueError('capability map contains no source hashes')
        for relative, expected in sorted(sources.items()):
            source = (root / str(relative)).resolve()
            if root not in source.parents or not source.is_file():
                raise ValueError(
                    'capability map source is missing: %s' % relative)
            if sha256_file(source) != str(expected):
                raise ValueError(
                    'capability map source hash mismatch: %s' % relative)
    return result
