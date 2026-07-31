import hashlib
from pathlib import Path
import struct

from piper_tesseract_foxy.collision_meshes import (
    read_binary_stl,
    split_binary_stl,
)


TRIANGLE_STRUCT = struct.Struct('<12fH')


def triangle(x0, x1, x2):
    return TRIANGLE_STRUCT.pack(
        0.0, 0.0, 1.0,
        x0, 0.0, 0.0,
        x1, 0.01, 0.0,
        x2, 0.0, 0.01,
        0,
    )


def write_stl(path, records):
    path.write_bytes(
        b'synthetic'.ljust(80, b'\0')
        + struct.pack('<I', len(records))
        + b''.join(records))


def directory_hashes(path):
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(Path(path).glob('*.stl'))
    }


def test_splitter_is_deterministic_and_never_drops_source_triangles(tmp_path):
    source = tmp_path / 'link.STL'
    records = [
        triangle(0.00, 0.01, 0.02),
        triangle(0.02, 0.04, 0.06),
        triangle(0.06, 0.08, 0.09),
    ]
    write_stl(source, records)
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    manifest_a = split_binary_stl(source, first, 'link', 'x', 0.03)
    manifest_b = split_binary_stl(source, second, 'link', 'x', 0.03)

    assert manifest_a == manifest_b
    assert manifest_a['triangle_assignments'] >= len(records)
    assert directory_hashes(first) == directory_hashes(second)
    emitted = [
        record
        for piece in sorted(first.glob('*.stl'))
        for record in read_binary_stl(piece)
    ]
    assert all(record in emitted for record in records)
