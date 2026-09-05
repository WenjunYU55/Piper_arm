"""Hash-bound fixed-world representation selection, independent of CUDA/ROS."""

import hashlib
import math
from pathlib import Path

import yaml


BOX_NAMES = {
    'bunker_chassis_collision_cuboid', 'sensor_station_housing',
    'sensor_station_lidar_camera', 'piper_base_collision_cuboid',
}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixed_world(path, provenance):
    """Check a reviewed cuboid source against canonical and moving geometry."""
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('invalid fixed-world source schema')
    if document.get('frame') != 'base_link':
        raise ValueError('fixed-world source must use base_link')
    expected = {x['name']: x['sha256'] for x in provenance['fixed_world_meshes']}
    if document.get('source_mesh_sha256') != expected:
        raise ValueError('fixed-world source mesh hashes do not match')
    if document.get('moving_sphere_sha256') != (
            provenance.get('curated_sphere_model') or {}).get('sha256'):
        raise ValueError('fixed-world moving sphere hash does not match')
    boxes = document.get('cuboids')
    if not isinstance(boxes, list) or len(boxes) != 4:
        raise ValueError('fixed-world source requires exactly four cuboids')
    names = set()
    for box in boxes:
        if not isinstance(box, dict) or set(box) != {'name', 'dims', 'pose'}:
            raise ValueError('invalid fixed-world cuboid fields')
        names.add(box['name'])
        for field, length in [('dims', 3), ('pose', 7)]:
            values = box[field]
            if (not isinstance(values, list) or len(values) != length
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in values)):
                raise ValueError('invalid fixed-world cuboid %s' % field)
        if min(box['dims']) <= 0 or box['pose'][3:] != [1., 0., 0., 0.]:
            raise ValueError('fixed-world cuboids must be positive and axis-aligned')
    if names != BOX_NAMES:
        raise ValueError('fixed-world cuboid names do not match')
    qualification = document.get('qualification')
    if not isinstance(qualification, dict):
        raise ValueError('fixed-world qualification is missing')
    # Both the articulated model and the replacement world must share the scope.
    moving = provenance.get('hardware_qualification', {})
    for key in ('hardware_qualified', 'scope', 'floor_profile',
                'free_motion_speed_percent', 'contact_speed_percent',
                'real_motion_requires_explicit_opt_in'):
        if key not in qualification or qualification[key] != moving.get(key):
            raise ValueError('fixed-world qualification scope mismatch: %s' % key)
    return document


def bind_fixed_world(provenance, path):
    document = load_fixed_world(path, provenance)
    provenance['fixed_world_representation'] = 'cuboids'
    provenance['fixed_world_model'] = {
        'path': str(Path(path).resolve()), 'sha256': file_digest(path),
    }
    provenance['fixed_world_cuboids'] = document['cuboids']
    provenance['hardware_qualification'] = document['qualification']
    provenance['hardware_qualified'] = document['qualification']['hardware_qualified']


def validate_fixed_world(provenance):
    """Return selection only after validating derived geometry and qualification."""
    selection = provenance.get('fixed_world_representation', 'meshes')
    if selection == 'meshes':
        if provenance.get('fixed_world_model') is not None:
            raise ValueError('mesh world cannot carry a cuboid source')
        return selection
    if selection != 'cuboids':
        raise ValueError('unknown fixed-world representation')
    record = provenance.get('fixed_world_model')
    if not isinstance(record, dict) or file_digest(record.get('path', '')) != record.get('sha256'):
        raise ValueError('fixed-world source hash does not match')
    document = load_fixed_world(record['path'], provenance)
    if document['cuboids'] != provenance.get('fixed_world_cuboids'):
        raise ValueError('fixed-world cuboids differ from reviewed source')
    if document['qualification'] != provenance.get('hardware_qualification'):
        raise ValueError('fixed-world qualification differs from reviewed source')
    if document['qualification']['hardware_qualified'] is not provenance.get('hardware_qualified'):
        raise ValueError('fixed-world qualification flag mismatch')
    return selection
