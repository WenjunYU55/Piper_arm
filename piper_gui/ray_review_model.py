"""ROS-free data model used by the Ray Review desktop process.

This module contains the intentionally boring, testable parts of the viewer:
schema loading, event-time ray state, read-only capability-map decoding, URDF
visual parsing/FK, and exact coverage-snapshot persistence.  It must remain
safe to import in a process with no ROS installation or daemon.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml


DIAGNOSTIC_SCHEMA_VERSION = 2
UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
SURFACE = np.uint8(2)

INITIAL_STAGES = (
    'generate', 'cull', 'prequalify', 'seed_rank', 'plan', 'capture',
    'update_target',
)
LATER_STAGES = (
    'cull_used_redundant', 'nbv_rank', 'information_cull', 'prequalify',
    'plan', 'capture', 'update_target',
)
TERMINAL_STAGES = ('completed', 'cancelled', 'failed')

STAGE_LABELS = {
    'generate': 'Generate',
    'cull': 'Cull',
    'prequalify': 'Prequalify',
    'seed_rank': 'Seed rank',
    'cull_used_redundant': 'Cull used/redundant',
    'nbv_rank': 'NBV rank',
    'information_cull': 'Information cull',
    'plan': 'Plan',
    'capture': 'Capture',
    'update_target': 'Update target',
    'completed': 'Completed',
    'cancelled': 'Cancelled',
    'failed': 'Failed',
    'legacy_snapshot': 'Historical final snapshot',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _event_sort_key(event):
    return (
        int(event.get('sequence', 0)),
        int(event.get('timestamp_ns', 0)),
        str(event.get('event_id', '')),
    )


def load_diagnostic_document(path):
    """Load schema v1 or v2 without pretending v1 contains a journal."""
    source = Path(path).resolve()
    with source.open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('ray diagnostic artifact is not an object')
    schema = int(value.get('schema_version', 1))
    if schema not in (1, DIAGNOSTIC_SCHEMA_VERSION):
        raise ValueError('ray diagnostic schema is unsupported')
    result = deepcopy(value)
    result['_source_path'] = str(source)
    result['_loaded_schema_version'] = schema
    result.setdefault('generations', [])
    if schema == 1:
        result['legacy_v1'] = True
        result['journal_complete'] = False
        result['events'] = _legacy_snapshot_events(result)
    else:
        result['legacy_v1'] = False
        result.setdefault('journal_complete', True)
        events = result.get('events', [])
        if not isinstance(events, list):
            raise ValueError('ray event journal is malformed')
        result['events'] = sorted(
            (deepcopy(item) for item in events if isinstance(item, dict)),
            key=_event_sort_key)
    return result


def _legacy_snapshot_events(document):
    """Expose v1 generations honestly as final snapshots, not fake stages."""
    events = []
    for sequence, generation in enumerate(document.get('generations', [])):
        if not isinstance(generation, dict):
            continue
        cycle = int(generation.get('generation', sequence))
        events.append({
            'event_id': 'legacy:%s:%d' % (
                generation.get('session_id', ''), cycle),
            'sequence': sequence,
            'timestamp_ns': 0,
            'accepted_view_cycle': cycle,
            'planner_revision': cycle,
            'stage': 'legacy_snapshot',
            'correlation_id': '',
            'message': (
                'Schema-v1 final generation record; intermediate event order '
                'was not recorded and is not reconstructed.'),
            'legacy_partial': True,
            'snapshot_generation': deepcopy(generation),
        })
    return events


def event_cycles(document):
    """Return contiguous cycle blocks without changing journal order."""
    cycles = []
    for event in document.get('events', []):
        cycle_id = int(event.get('accepted_view_cycle', 0))
        if not cycles or cycles[-1]['cycle'] != cycle_id:
            cycles.append({'cycle': cycle_id, 'events': []})
        cycles[-1]['events'].append(event)
    return cycles


def _ray_map(values):
    return {
        int(item['ray_id']): deepcopy(item)
        for item in values if isinstance(item, dict) and 'ray_id' in item
    }


def state_at_event(document, event_index):
    """Materialize only evidence known at one scrub position.

    Rank and score fields enter state through the current or an earlier event;
    final generation records are never consulted for schema-v2 playback.
    """
    events = document.get('events', [])
    if not events:
        return {'rays': {}, 'event': None, 'target_model': None,
                'target_center_m': None, 'target_envelope': None,
                'robot_pose': None,
                'captured_ray_ids': []}
    index = max(0, min(int(event_index), len(events) - 1))
    rays = {}
    target_model = None
    target_center = None
    target_envelope = None
    robot_pose = None
    captured = []
    for event in events[:index + 1]:
        event_token = str(event.get(
            'event_id', event.get('sequence', '')))
        legacy = event.get('snapshot_generation')
        if isinstance(legacy, dict):
            rays = _ray_map(legacy.get('rays', []))
            if legacy.get('target_center_m') is not None:
                target_center = deepcopy(legacy['target_center_m'])
        deltas = event.get('ray_deltas', {})
        if isinstance(deltas, list):
            deltas = {
                str(item['ray_id']): item for item in deltas
                if isinstance(item, dict) and 'ray_id' in item}
        if isinstance(deltas, dict):
            for raw_id, delta in deltas.items():
                if not isinstance(delta, dict):
                    continue
                ray_id = int(delta.get('ray_id', raw_id))
                current = rays.setdefault(ray_id, {'ray_id': ray_id})
                was_culled = bool(
                    current.get('culled') or current.get('status') == 'culled')
                becomes_active = (
                    delta.get('culled') is False
                    or delta.get('status') in ('surviving', 'remaining'))
                if was_culled and becomes_active:
                    current['previous_cull_stage'] = str(
                        current.get('cull_stage', ''))
                    current['previous_cull_reasons'] = deepcopy(
                        current.get('reasons', []))
                    current['_reevaluated_event_id'] = event_token
                current.update(deepcopy(delta))
        if event.get('target_model') is not None:
            target_model = deepcopy(event['target_model'])
        if event.get('target_envelope') is not None:
            target_envelope = deepcopy(event['target_envelope'])
        if event.get('target_center_m') is not None:
            target_center = deepcopy(event['target_center_m'])
        if event.get('robot_pose') is not None:
            robot_pose = deepcopy(event['robot_pose'])
        for ray_id in event.get('captured_ray_ids', []):
            value = int(ray_id)
            if value not in captured:
                captured.append(value)
    selected = {
        int(value) for value in events[index].get('selected_ray_ids', [])}
    newly_culled = {
        int(value) for value in events[index].get('newly_culled_ray_ids', [])}
    for ray_id, ray in rays.items():
        ray['selected_at_event'] = ray_id in selected
        ray['newly_culled_at_event'] = ray_id in newly_culled
        ray['captured'] = ray_id in captured
        ray['reevaluated_at_event'] = (
            ray.get('_reevaluated_event_id') == str(events[index].get(
                'event_id', events[index].get('sequence', ''))))
    return {
        'rays': rays,
        'event': events[index],
        'target_model': target_model,
        'target_center_m': target_center,
        'target_envelope': target_envelope,
        'robot_pose': robot_pose,
        'captured_ray_ids': captured,
    }


def filter_rays(state, cull_stage='', reason='', rank_min=None,
                rank_max=None, generation=None, selected_captured=False,
                ray_id=''):
    """Apply inspector filters to an event-time state."""
    result = []
    identifier = str(ray_id).strip().lower()
    for ray in state.get('rays', {}).values():
        if cull_stage and str(ray.get('cull_stage', '')) != cull_stage:
            continue
        reasons = ray.get('reasons', ray.get('planner_reasons', []))
        reason_text = json.dumps(reasons, sort_keys=True).lower()
        if reason and reason.lower() not in reason_text:
            continue
        rank = ray.get('rank')
        if rank_min is not None and (rank is None or int(rank) < int(rank_min)):
            continue
        if rank_max is not None and (rank is None or int(rank) > int(rank_max)):
            continue
        if generation is not None and int(ray.get(
                'generation', generation)) != int(generation):
            continue
        if selected_captured and not (
                ray.get('selected_at_event') or ray.get('captured')):
            continue
        if identifier and identifier not in str(ray.get('ray_id', '')).lower():
            continue
        result.append(ray)
    return sorted(result, key=lambda item: (
        int(item.get('rank', 1 << 30)), int(item.get('ray_id', -1))))


def write_coverage_snapshot(path, snapshot, capture_artifacts=None,
                            configuration_artifacts=None, dataset_root=None):
    """Persist an exact pickle-free compressed ObjectCoverageModel snapshot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(snapshot.voxel_centers, dtype=np.float64)
    states = np.asarray(snapshot.states, dtype=np.uint8)
    bits = np.asarray(snapshot.surface_view_bits, dtype=np.uint32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError('coverage voxel centres are malformed')
    if states.shape != (len(centers),) or bits.shape != states.shape:
        raise ValueError('coverage arrays are inconsistent')
    side = round(len(centers) ** (1.0 / 3.0))
    grid_shape = (
        [int(side)] * 3 if side ** 3 == len(centers)
        else [int(len(centers)), 1, 1])
    root = Path(dataset_root).resolve() if dataset_root else output.parent

    def binding(path_value):
        source = Path(path_value).resolve()
        try:
            relative = str(source.relative_to(root))
        except ValueError:
            relative = os.path.relpath(str(source), str(root))
        return {'path': relative, 'sha256': sha256_file(source)}

    metadata = {
        'schema_version': 1,
        'artifact_kind': 'CoverageSnapshot',
        'session_id': str(snapshot.session_id),
        'generation': int(snapshot.generation),
        'target_center_m': [float(value) for value in snapshot.target_center],
        'radius_m': float(snapshot.radius_m),
        'voxel_size_m': float(snapshot.voxel_size_m),
        'grid_shape': grid_shape,
        'state_encoding': {'UNKNOWN': 0, 'FREE': 1, 'SURFACE': 2},
        'capture_artifacts': [binding(value) for value in capture_artifacts or []],
        'configuration_artifacts': [
            binding(value) for value in configuration_artifacts or []],
    }
    payload = np.frombuffer(json.dumps(
        metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'),
        dtype=np.uint8)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.' + output.name + '.', suffix='.npz', dir=str(output.parent))
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary, metadata_json=payload, states=states,
            observed_direction_bits=bits, voxel_centers_m=centers,
            view_directions=np.asarray(snapshot.view_directions, dtype=np.float64))
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output


def load_coverage_snapshot(path):
    with np.load(str(path), allow_pickle=False) as archive:
        required = {
            'metadata_json', 'states', 'observed_direction_bits',
            'voxel_centers_m'}
        if not required.issubset(archive.files):
            raise ValueError('coverage snapshot is missing required arrays')
        metadata = json.loads(bytes(np.asarray(
            archive['metadata_json'], dtype=np.uint8)).decode('utf-8'))
        return {
            'metadata': metadata,
            'states': np.array(archive['states'], copy=True),
            'observed_direction_bits': np.array(
                archive['observed_direction_bits'], copy=True),
            'voxel_centers_m': np.array(
                archive['voxel_centers_m'], copy=True),
        }


@dataclass(frozen=True)
class CapabilityView:
    path: Path
    sha256: str
    mtime_ns: int
    metadata: dict
    positions_m: np.ndarray
    direction_density: np.ndarray
    maximum_floor_m: np.ndarray
    direction_histogram: np.ndarray
    occupied_pose_direction_bins: int
    keys: np.ndarray
    maximum_tool_minimum_z_m: np.ndarray


def load_capability_view(path):
    """Decode the committed map read-only; never imports its generator."""
    source = Path(path).resolve()
    before = source.stat()
    digest = sha256_file(source)
    with np.load(str(source), allow_pickle=False) as archive:
        keys = np.asarray(archive['keys'], dtype=np.uint64)
        floors = np.asarray(
            archive['maximum_tool_minimum_z_m'], dtype=np.float32)
        metadata = json.loads(bytes(np.asarray(
            archive['metadata_json'], dtype=np.uint8)).decode('utf-8'))
    project_root = source.parents[4] if len(source.parents) > 4 else source.parent
    source_hashes = metadata.get('source_sha256', {})
    source_validation = 'unavailable'
    if isinstance(source_hashes, dict) and source_hashes:
        source_validation = 'valid'
        for relative, expected in sorted(source_hashes.items()):
            candidate = (project_root / str(relative)).resolve()
            if (not candidate.is_file()
                    or sha256_file(candidate) != str(expected)):
                source_validation = 'invalid'
                break
    metadata['_viewer_source_validation'] = source_validation
    position_mask = np.uint64((1 << 33) - 1)
    packed_position = keys & position_mask
    unique, first, counts = np.unique(
        packed_position, return_index=True, return_counts=True)
    offset = 1 << 10
    mask = (1 << 11) - 1
    positions = np.column_stack([
        ((unique >> np.uint64(shift)) & np.uint64(mask)).astype(np.int64)
        - offset for shift in (0, 11, 22)])
    voxel = float(metadata['position_voxel_m'])
    positions_m = (positions.astype(np.float32) + 0.5) * voxel
    # reduceat is valid because packed position is the low key field and keys
    # are sorted, making every position cell contiguous.
    maximum_floor = np.maximum.reduceat(floors, first)
    azimuth = ((keys >> np.uint64(33)) & np.uint64(255)).astype(np.int64)
    elevation = ((keys >> np.uint64(41)) & np.uint64(255)).astype(np.int64)
    bin_deg = float(metadata['direction_bin_deg'])
    histogram = np.zeros((round(180.0 / bin_deg), round(360.0 / bin_deg)),
                         dtype=np.uint32)
    np.add.at(histogram, (elevation, azimuth), 1)
    after = source.stat()
    if before.st_mtime_ns != after.st_mtime_ns or digest != sha256_file(source):
        raise RuntimeError('capability map changed while being viewed')
    return CapabilityView(
        path=source, sha256=digest, mtime_ns=before.st_mtime_ns,
        metadata=metadata, positions_m=positions_m,
        direction_density=counts.astype(np.uint32),
        maximum_floor_m=maximum_floor,
        direction_histogram=histogram,
        occupied_pose_direction_bins=len(keys),
        keys=keys,
        maximum_tool_minimum_z_m=floors,
    )


def capability_ray_overlay(view, target_center, ray_direction,
                           minimum_standoff_m, maximum_standoff_m,
                           floor_z_m=-1e6, clearance_m=0.0):
    """Return checked/matched/unsupported map cells for one mission ray.

    The overlay is diagnostic only.  It exposes occupancy evidence and never
    presents a cell as an IK state or trajectory.
    """
    target = np.asarray(target_center, dtype=float)
    direction = np.asarray(ray_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    minimum = float(minimum_standoff_m)
    maximum = float(maximum_standoff_m)
    if (target.shape != (3,) or direction.shape != (3,) or norm <= 1e-12
            or minimum <= 0.0 or maximum < minimum):
        return {'checked': np.empty((0, 3)), 'matched': np.empty((0, 3)),
                'unsupported': np.empty((0, 3)), 'reason': 'ray interval is invalid'}
    direction /= norm
    voxel = float(view.metadata['position_voxel_m'])
    count = max(1, int(math.ceil((maximum - minimum) / voxel)) + 1)
    positions = target + np.outer(np.linspace(minimum, maximum, count), direction)
    # Decode the exact optical direction bin stored in the artifact.
    optical = -direction
    bin_deg = float(view.metadata['direction_bin_deg'])
    azimuth = math.degrees(math.atan2(optical[1], optical[0])) % 360.0
    elevation = math.degrees(math.asin(float(np.clip(optical[2], -1.0, 1.0))))
    azimuth_index = int(math.floor(azimuth / bin_deg))
    elevation_index = max(0, min(round(180.0 / bin_deg) - 1,
                                 int(math.floor((elevation + 90.0) / bin_deg))))
    position_mask = np.uint64((1 << 33) - 1)
    key_azimuth = ((view.keys >> np.uint64(33)) & np.uint64(255)).astype(np.int16)
    key_elevation = ((view.keys >> np.uint64(41)) & np.uint64(255)).astype(np.int16)
    direction_match = ((key_azimuth == azimuth_index)
                       & (key_elevation == elevation_index))
    supported_positions = view.keys[direction_match] & position_mask
    supported_floors = view.maximum_tool_minimum_z_m[direction_match]
    offset = 1 << 10
    indexes = np.floor(positions / voxel).astype(np.int64)
    inside = np.all((indexes >= -offset) & (indexes < offset), axis=1)
    packed = np.zeros(len(indexes), dtype=np.uint64)
    encoded = (indexes[inside] + offset).astype(np.uint64)
    packed[inside] = (encoded[:, 0] | (encoded[:, 1] << np.uint64(11))
                      | (encoded[:, 2] << np.uint64(22)))
    locations = np.searchsorted(supported_positions, packed)
    valid = inside & (locations < len(supported_positions))
    matched = np.zeros(len(positions), dtype=bool)
    candidate_rows = np.flatnonzero(valid)
    if len(candidate_rows):
        found = supported_positions[locations[candidate_rows]] == packed[candidate_rows]
        floor_ok = supported_floors[locations[candidate_rows]] >= (
            float(floor_z_m) + float(clearance_m) - 1e-9)
        matched[candidate_rows] = found & floor_ok
    return {
        'checked': positions,
        'matched': positions[matched],
        'unsupported': positions[~matched],
        'checked_count': int(len(positions)),
        'matching_count': int(np.count_nonzero(matched)),
        'reason': '' if np.any(matched) else (
            'no collision-qualified capability cell intersects ray'),
    }


def _rpy_matrix(values):
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _origin(element):
    result = np.eye(4, dtype=np.float64)
    if element is None:
        return result
    xyz = [float(value) for value in element.get('xyz', '0 0 0').split()]
    rpy = [float(value) for value in element.get('rpy', '0 0 0').split()]
    result = _rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.eye(4)
    x_value, y_value, z_value = axis / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    one = 1.0 - cosine
    result = np.eye(4)
    result[:3, :3] = [
        [cosine + x_value * x_value * one,
         x_value * y_value * one - z_value * sine,
         x_value * z_value * one + y_value * sine],
        [y_value * x_value * one + z_value * sine,
         cosine + y_value * y_value * one,
         y_value * z_value * one - x_value * sine],
        [z_value * x_value * one - y_value * sine,
         z_value * y_value * one + x_value * sine,
         cosine + z_value * z_value * one],
    ]
    return result


def load_optical_registration(path):
    """Load the accepted L515 visual-to-colour-optical calibration."""
    with Path(path).open('r', encoding='utf-8') as stream:
        record = yaml.safe_load(stream) or {}
    if str(record.get('status', '')).lower() != 'accepted':
        raise ValueError('camera optical registration is not accepted')
    registration = record.get('mechanical_registration', {})
    translation = np.asarray(registration.get(
        'l515_visual_to_camera_color_optical_translation_m'), dtype=float)
    quaternion = np.asarray(registration.get(
        'l515_visual_to_camera_color_optical_quaternion_xyzw'), dtype=float)
    if (translation.shape != (3,) or quaternion.shape != (4,)
            or not np.all(np.isfinite(translation))
            or not np.all(np.isfinite(quaternion))):
        raise ValueError('camera optical registration is malformed')
    quaternion /= np.linalg.norm(quaternion)
    x_value, y_value, z_value, w_value = quaternion
    rotation = np.asarray([
        [1 - 2 * (y_value ** 2 + z_value ** 2),
         2 * (x_value * y_value - z_value * w_value),
         2 * (x_value * z_value + y_value * w_value)],
        [2 * (x_value * y_value + z_value * w_value),
         1 - 2 * (x_value ** 2 + z_value ** 2),
         2 * (y_value * z_value - x_value * w_value)],
        [2 * (x_value * z_value - y_value * w_value),
         2 * (y_value * z_value + x_value * w_value),
         1 - 2 * (x_value ** 2 + y_value ** 2)],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


@dataclass(frozen=True)
class UrdfVisual:
    link: str
    mesh_path: Path
    scale: tuple
    color: tuple
    origin: np.ndarray


class UrdfAssembly:
    """Checked-in URDF visuals and exact joint-origin/axis FK."""

    def __init__(self, urdf_path):
        self.path = Path(urdf_path).resolve()
        self.package_root = self.path.parents[1]
        root = ET.parse(str(self.path)).getroot()
        self.visuals = []
        for link in root.findall('link'):
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    continue
                filename = mesh.get('filename', '')
                prefix = 'package://piper_description/'
                if not filename.startswith(prefix):
                    continue
                material = visual.find('material/color')
                rgba = tuple(float(value) for value in (
                    material.get('rgba', '0.75 0.78 0.84 1').split()
                    if material is not None else '0.75 0.78 0.84 1'.split()))
                scale = tuple(float(value) for value in
                              mesh.get('scale', '1 1 1').split())
                self.visuals.append(UrdfVisual(
                    link=link.get('name'),
                    mesh_path=self.package_root / filename[len(prefix):],
                    scale=scale, color=rgba,
                    origin=_origin(visual.find('origin')),
                ))
        self.joints = []
        self.children = {}
        for joint in root.findall('joint'):
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            axis_node = joint.find('axis')
            axis = tuple(float(value) for value in (
                axis_node.get('xyz', '1 0 0').split()
                if axis_node is not None else '1 0 0'.split()))
            record = {
                'name': joint.get('name'), 'type': joint.get('type'),
                'parent': parent, 'child': child,
                'origin': _origin(joint.find('origin')), 'axis': axis,
            }
            self.joints.append(record)
            self.children.setdefault(parent, []).append(record)

    def transforms(self, joint_positions=None):
        positions = dict(joint_positions or {})
        result = {'world': np.eye(4)}
        pending = ['world']
        while pending:
            parent = pending.pop(0)
            for joint in self.children.get(parent, []):
                transform = result[parent].dot(joint['origin'])
                if joint['type'] in ('revolute', 'continuous'):
                    transform = transform.dot(_axis_rotation(
                        joint['axis'], float(positions.get(joint['name'], 0.0))))
                result[joint['child']] = transform
                pending.append(joint['child'])
        return result

    def visual_transforms(self, joint_positions=None):
        links = self.transforms(joint_positions)
        return [(item, links.get(item.link, np.eye(4)).dot(item.origin))
                for item in self.visuals]


def stl_triangle_count(path):
    """Count binary or ASCII STL facets without changing the mesh."""
    source = Path(path)
    size = source.stat().st_size
    with source.open('rb') as stream:
        header = stream.read(84)
    if len(header) >= 84:
        count = int.from_bytes(header[80:84], 'little')
        if 84 + count * 50 == size:
            return count
    with source.open('r', encoding='utf-8', errors='ignore') as stream:
        return sum(1 for line in stream if line.lstrip().startswith('facet normal'))


def assembly_triangle_count(assembly):
    return sum(stl_triangle_count(item.mesh_path) for item in assembly.visuals)
