"""Deterministically split detailed binary STL meshes into short convex pieces.

Tesseract's ``make_convex`` policy converts every URDF mesh into one convex
hull.  That is appropriate for Bullet's moving geometry, but one hull around a
long concave link fills intentional folded-arm clearance.  This module divides
selected source meshes into overlapping longitudinal slabs before Tesseract
convexifies them.  Every source triangle is copied to every slab intersected by
its axis-aligned extent, so the generated surface never drops source geometry.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
TRIANGLE_STRUCT = struct.Struct('<12fH')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_binary_stl(path):
    """Return the original 50-byte triangle records from a binary STL."""
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError('%s is too short to be a binary STL' % path)
    count = struct.unpack('<I', data[80:84])[0]
    expected = 84 + count * TRIANGLE_STRUCT.size
    if len(data) != expected:
        raise ValueError(
            '%s is not a supported binary STL: expected %d bytes, found %d'
            % (path, expected, len(data)))
    return [
        data[84 + index * TRIANGLE_STRUCT.size:
             84 + (index + 1) * TRIANGLE_STRUCT.size]
        for index in range(count)
    ]


def triangle_axis_bounds(record, axis_index):
    values = TRIANGLE_STRUCT.unpack(record)
    coordinates = (
        values[3 + axis_index],
        values[6 + axis_index],
        values[9 + axis_index],
    )
    return min(coordinates), max(coordinates)


def write_binary_stl(path, records, label):
    header = ('PiPER deterministic convex slab ' + label).encode('ascii')
    header = header[:80].ljust(80, b'\0')
    output = header + struct.pack('<I', len(records)) + b''.join(records)
    Path(path).write_bytes(output)


def _rpy_transform(xyz, rpy):
    """Return the URDF fixed-axis transform for one origin element."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    rotation_y = np.asarray([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    rotation_z = np.asarray([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _origin_transform(origin):
    if origin is None:
        return np.eye(4)
    xyz = [float(value) for value in origin.get('xyz', '0 0 0').split()]
    rpy = [float(value) for value in origin.get('rpy', '0 0 0').split()]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(
            'URDF origin must contain three xyz and three rpy values')
    return _rpy_transform(xyz, rpy)


def _fixed_link_transform(root, base_link, child_link):
    joints = {}
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is None or child is None:
            continue
        joints[str(child.get('link'))] = (
            str(parent.get('link')),
            _origin_transform(joint.find('origin')),
        )
    transform = np.eye(4)
    current = str(child_link)
    visited = set()
    while current != str(base_link):
        if current in visited or current not in joints:
            raise ValueError(
                'no fixed URDF chain from %s to %s' % (base_link, child_link))
        visited.add(current)
        parent, parent_from_child = joints[current]
        transform = parent_from_child @ transform
        current = parent
    return transform


def _transformed_triangle_records(path, transform, scale):
    records = []
    rotation = np.asarray(transform, dtype=float)[:3, :3]
    translation = np.asarray(transform, dtype=float)[:3, 3]
    scale_values = np.asarray(scale, dtype=float)
    for record in read_binary_stl(path):
        values = TRIANGLE_STRUCT.unpack(record)
        vertices = np.asarray([
            values[3:6], values[6:9], values[9:12],
        ], dtype=float)
        vertices = (rotation @ (vertices * scale_values).T).T + translation
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        length = float(np.linalg.norm(normal))
        if length > 0.0:
            normal /= length
        else:
            normal = np.zeros(3)
        records.append(TRIANGLE_STRUCT.pack(
            *(normal.tolist()
              + vertices[0].tolist()
              + vertices[1].tolist()
              + vertices[2].tolist()
              + [int(values[12])]),
        ))
    return records


def transform_binary_stl(source_path, output_path, transform, scale, label):
    """Write one source STL in a declared fixed planning-frame transform."""
    transform_array = np.asarray(transform, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    if transform_array.shape != (4, 4) or not np.all(
            np.isfinite(transform_array)):
        raise ValueError('transform must be a finite 4x4 matrix')
    if scale_array.shape != (3,) or not np.all(np.isfinite(scale_array)) or \
            np.any(scale_array <= 0.0):
        raise ValueError('scale must contain three positive finite values')
    records = _transformed_triangle_records(
        source_path, transform_array, scale_array)
    write_binary_stl(output_path, records, label)
    return {
        'source_sha256': sha256_file(source_path),
        'source_triangle_count': len(records),
        'scale': scale_array.tolist(),
        'base_from_mesh': transform_array.tolist(),
        'sha256': sha256_file(output_path),
    }


def build_visual_assembly_stl(
        urdf_path, output_path, base_link, visual_links):
    """Merge installed visual meshes into one deterministic base-frame STL."""
    urdf_path = Path(urdf_path)
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    links = {
        str(link.get('name')): link for link in root.findall('link')
        if link.get('name')
    }
    package_root = urdf_path.resolve().parents[1]
    records = []
    sources = []
    for link_name in visual_links:
        if link_name not in links:
            raise ValueError(
                'assembly names unknown visual link %s' % link_name)
        visual = links[link_name].find('visual')
        mesh = None if visual is None else visual.find('geometry/mesh')
        if mesh is None:
            raise ValueError('%s has no visual mesh' % link_name)
        uri = str(mesh.get('filename', ''))
        prefix = 'package://piper_description/'
        if not uri.startswith(prefix):
            raise ValueError(
                '%s visual mesh is outside piper_description' % link_name)
        source = (package_root / uri[len(prefix):]).resolve()
        if package_root not in source.parents:
            raise ValueError(
                '%s visual mesh escapes piper_description' % link_name)
        scale = [
            float(value)
            for value in mesh.get('scale', '1 1 1').split()
        ]
        if len(scale) != 3 or not all(
                math.isfinite(value) and value > 0.0 for value in scale):
            raise ValueError('%s visual mesh scale is invalid' % link_name)
        transform = (
            _fixed_link_transform(root, base_link, link_name)
            @ _origin_transform(visual.find('origin'))
        )
        transformed = _transformed_triangle_records(source, transform, scale)
        records.extend(transformed)
        sources.append({
            'link': str(link_name),
            'mesh_uri': uri,
            'mesh_sha256': sha256_file(source),
            'triangle_count': len(transformed),
            'scale': scale,
            'base_from_mesh': np.asarray(transform, dtype=float).tolist(),
        })
    write_binary_stl(output_path, records, 'installed L515 assembly')
    return {
        'base_link': str(base_link),
        'visual_links': [str(value) for value in visual_links],
        'sources': sources,
        'triangle_count': len(records),
        'sha256': sha256_file(output_path),
    }


def split_binary_stl(source_path, output_dir, link_name, axis,
                     slab_thickness_m):
    """Split one STL and return a hash-addressed piece manifest."""
    if axis not in AXIS_INDEX:
        raise ValueError('axis must be x, y, or z')
    thickness = float(slab_thickness_m)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError('slab_thickness_m must be positive and finite')
    records = read_binary_stl(source_path)
    if not records:
        raise ValueError('%s contains no triangles' % source_path)
    axis_index = AXIS_INDEX[axis]
    bounds = [triangle_axis_bounds(record, axis_index) for record in records]
    mesh_min = min(value[0] for value in bounds)
    mesh_max = max(value[1] for value in bounds)
    slab_count = max(1, int(math.ceil((mesh_max - mesh_min) / thickness)))
    slabs = [[] for _ in range(slab_count)]
    epsilon = 1e-9
    assignments = 0
    for record, (triangle_min, triangle_max) in zip(records, bounds):
        first = int(math.floor(
            max(0.0, triangle_min - mesh_min - epsilon) / thickness))
        last = int(math.floor(
            max(0.0, triangle_max - mesh_min + epsilon) / thickness))
        first = min(max(first, 0), slab_count - 1)
        last = min(max(last, first), slab_count - 1)
        for slab_index in range(first, last + 1):
            slabs[slab_index].append(record)
            assignments += 1
    if any(not slab for slab in slabs):
        raise ValueError(
            '%s produced an empty slab; choose a coarser deterministic split'
            % source_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for index, slab in enumerate(slabs):
        name = '%s_slab_%03d.stl' % (link_name, index)
        path = output_dir / name
        write_binary_stl(path, slab, '%s %03d' % (link_name, index))
        pieces.append({
            'filename': name,
            'sha256': sha256_file(path),
            'triangle_count': len(slab),
        })
    return {
        'link': str(link_name),
        'axis': axis,
        'slab_thickness_m': thickness,
        'source_sha256': sha256_file(source_path),
        'source_triangle_count': len(records),
        'triangle_assignments': assignments,
        'pieces': pieces,
    }


def split_binary_stl_grid(
        source_path, output_dir, link_name, axes, slab_thickness_m):
    """Split a concave mesh on several axes without dropping triangles."""
    axes = tuple(str(value) for value in axes)
    thicknesses = tuple(float(value) for value in slab_thickness_m)
    if not axes or len(axes) != len(thicknesses):
        raise ValueError(
            'axes and slab_thickness_m must be nonempty and equal length')
    if len(set(axes)) != len(axes) or any(
            axis not in AXIS_INDEX for axis in axes):
        raise ValueError('axes must contain unique x, y, or z values')
    if any(
            not math.isfinite(value) or value <= 0.0
            for value in thicknesses):
        raise ValueError('slab_thickness_m values must be positive and finite')
    records = read_binary_stl(source_path)
    if not records:
        raise ValueError('%s contains no triangles' % source_path)
    per_axis_bounds = [
        [triangle_axis_bounds(record, AXIS_INDEX[axis]) for record in records]
        for axis in axes
    ]
    mesh_mins = [
        min(item[0] for item in bounds) for bounds in per_axis_bounds]
    mesh_maxs = [
        max(item[1] for item in bounds) for bounds in per_axis_bounds]
    counts = [
        max(1, int(math.ceil((maximum - minimum) / thickness)))
        for minimum, maximum, thickness in zip(
            mesh_mins, mesh_maxs, thicknesses)
    ]
    cells = {}
    epsilon = 1e-9
    assignments = 0
    for record_index, record in enumerate(records):
        ranges = []
        for axis_index, thickness in enumerate(thicknesses):
            triangle_min, triangle_max = (
                per_axis_bounds[axis_index][record_index])
            first = int(math.floor(max(
                0.0, triangle_min - mesh_mins[axis_index] - epsilon)
                / thickness))
            last = int(math.floor(max(
                0.0, triangle_max - mesh_mins[axis_index] + epsilon)
                / thickness))
            first = min(max(first, 0), counts[axis_index] - 1)
            last = min(max(last, first), counts[axis_index] - 1)
            ranges.append(range(first, last + 1))
        indices = [()]
        for values in ranges:
            indices = [
                prefix + (value,)
                for prefix in indices
                for value in values
            ]
        for index in indices:
            cells.setdefault(index, []).append(record)
            assignments += 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for index, cell_records in sorted(cells.items()):
        suffix = '_'.join('%03d' % value for value in index)
        name = '%s_cell_%s.stl' % (link_name, suffix)
        path = output_dir / name
        write_binary_stl(path, cell_records, '%s %s' % (link_name, suffix))
        pieces.append({
            'filename': name,
            'grid_index': list(index),
            'sha256': sha256_file(path),
            'triangle_count': len(cell_records),
        })
    return {
        'link': str(link_name),
        'axes': list(axes),
        'slab_thickness_m': list(thicknesses),
        'source_sha256': sha256_file(source_path),
        'source_triangle_count': len(records),
        'triangle_assignments': assignments,
        'pieces': pieces,
    }


def generate_default_assets(mesh_root, output_dir, thickness=0.02):
    """Generate the reviewed PiPER folded-arm decomposition candidates."""
    specifications = (
        ('link1', 'y'),
        ('link2', 'x'),
        ('link5', 'y'),
    )
    return [
        split_binary_stl(
            Path(mesh_root) / (link + '.STL'),
            output_dir,
            link,
            axis,
            thickness,
        )
        for link, axis in specifications
    ]


def generate_bunker_assets(mesh_root, output_dir, thickness=0.15):
    """Build the locked Bunker chassis/station collision decomposition."""
    mesh_root = Path(mesh_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    description_meshes = output_dir.parent
    specifications = (
        (
            'bunker_chassis_collision',
            mesh_root / 'bunker_pro2_base_link.STL',
            np.asarray([
                [1.0, 0.0, 0.0, -0.390],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.016],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            (1.0, 1.0, 1.0),
        ),
        (
            'bunker_sensor_station_collision',
            mesh_root / 'bunker_pro2_FullCase.STL',
            np.asarray([
                [-1.0, 0.0, 0.0, -0.33375],
                [0.0, 0.0, 1.0, -0.2335],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            (0.001, 0.001, 0.001),
        ),
    )
    result = []
    for link_name, source, transform, scale in specifications:
        transformed = description_meshes / (link_name + '.STL')
        provenance = transform_binary_stl(
            source, transformed, transform, scale, link_name)
        entry = split_binary_stl_grid(
            transformed, output_dir, link_name, ('x', 'y'),
            (thickness, thickness))
        entry['assembly'] = {
            'base_link': 'base_link',
            'source': provenance,
        }
        result.append(entry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--slab-thickness-m', type=float, default=0.02)
    parser.add_argument('--manifest-output')
    parser.add_argument('--xacro')
    parser.add_argument('--assembly-output')
    parser.add_argument(
        '--bunker-platform', action='store_true',
        help='Generate the locked Bunker chassis/station collision assets.')
    args = parser.parse_args(argv)
    if args.bunker_platform:
        result = generate_bunker_assets(
            args.mesh_root, args.output_dir, args.slab_thickness_m)
    else:
        result = generate_default_assets(
            args.mesh_root, args.output_dir, args.slab_thickness_m)
    assembly = None
    if bool(args.xacro) != bool(args.assembly_output):
        parser.error('--xacro and --assembly-output must be supplied together')
    if args.xacro:
        assembly = build_visual_assembly_stl(
            args.xacro,
            args.assembly_output,
            'gripper_base',
            ('camera_holder', 'l515_visual'),
        )
        entry = split_binary_stl_grid(
            args.assembly_output,
            args.output_dir,
            'l515_attached_assembly',
            ('x', 'y'),
            (args.slab_thickness_m, args.slab_thickness_m),
        )
        entry['assembly'] = assembly
        result.append(entry)
    if args.manifest_output:
        document = {
            'schema_version': 1,
            'generator': (
                'transformed_platform_grid_v1'
                if args.bunker_platform else
                'longitudinal_triangle_slab_v1+'
                'installed_visual_grid_v1'),
            'links': result,
        }
        Path(args.manifest_output).write_text(
            json.dumps(document, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
