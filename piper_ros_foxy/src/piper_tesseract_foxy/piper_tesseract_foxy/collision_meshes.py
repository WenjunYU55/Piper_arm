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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--slab-thickness-m', type=float, default=0.02)
    parser.add_argument('--manifest-output')
    args = parser.parse_args(argv)
    result = generate_default_assets(
        args.mesh_root, args.output_dir, args.slab_thickness_m)
    if args.manifest_output:
        document = {
            'schema_version': 1,
            'generator': 'longitudinal_triangle_slab_v1',
            'links': result,
        }
        Path(args.manifest_output).write_text(
            json.dumps(document, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
