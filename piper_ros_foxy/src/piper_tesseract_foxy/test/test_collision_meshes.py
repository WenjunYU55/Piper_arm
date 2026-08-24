import hashlib
from pathlib import Path
import struct

import pytest

from piper_tesseract_foxy.collision_meshes import (
    build_visual_assembly_stl,
    read_binary_stl,
    split_binary_stl,
    split_binary_stl_grid,
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


def test_visual_assembly_uses_fixed_joint_and_visual_transforms(tmp_path):
    package = tmp_path / 'piper_description'
    meshes = package / 'meshes'
    urdf = package / 'urdf'
    meshes.mkdir(parents=True)
    urdf.mkdir()
    first = meshes / 'first.STL'
    second = meshes / 'second.STL'
    write_stl(first, [triangle(0.0, 1.0, 2.0)])
    write_stl(second, [triangle(0.0, 1.0, 2.0)])
    model = urdf / 'model.urdf'
    model.write_text('''
<robot name="test">
  <link name="base"/>
  <link name="holder">
    <visual>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry><mesh
        filename="package://piper_description/meshes/first.STL"
        scale="0.001 0.001 0.001"/></geometry>
    </visual>
  </link>
  <joint name="holder_joint" type="fixed">
    <origin xyz="1 2 3" rpy="0 0 0"/>
    <parent link="base"/><child link="holder"/>
  </joint>
  <link name="camera">
    <visual>
      <geometry><mesh
        filename="package://piper_description/meshes/second.STL"
        scale="0.001 0.001 0.001"/></geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <parent link="holder"/><child link="camera"/>
  </joint>
</robot>
''', encoding='utf-8')
    output = meshes / 'assembly.STL'
    provenance = build_visual_assembly_stl(
        model, output, 'base', ('holder', 'camera'))

    assert provenance['triangle_count'] == 2
    records = read_binary_stl(output)
    first_values = TRIANGLE_STRUCT.unpack(records[0])
    second_values = TRIANGLE_STRUCT.unpack(records[1])
    assert first_values[3:6] == pytest.approx((1.1, 2.0, 3.0))
    assert second_values[3:6] == pytest.approx((1.0, 2.2, 3.0))


def test_grid_splitter_is_deterministic_and_preserves_every_triangle(tmp_path):
    source = tmp_path / 'assembly.STL'
    records = [
        triangle(0.00, 0.01, 0.02),
        triangle(0.02, 0.04, 0.06),
        triangle(0.06, 0.08, 0.09),
    ]
    write_stl(source, records)
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    manifest_a = split_binary_stl_grid(
        source, first, 'assembly', ('x', 'y'), (0.03, 0.005))
    manifest_b = split_binary_stl_grid(
        source, second, 'assembly', ('x', 'y'), (0.03, 0.005))

    assert manifest_a == manifest_b
    assert manifest_a['triangle_assignments'] >= len(records)
    assert directory_hashes(first) == directory_hashes(second)
    emitted = [
        record
        for piece in sorted(first.glob('*.stl'))
        for record in read_binary_stl(piece)
    ]
    assert all(record in emitted for record in records)


def test_committed_l515_assembly_matches_locked_visual_poses(tmp_path):
    root = Path(__file__).resolve().parents[4]
    description = root / 'piper_ros_foxy/src/piper_description'
    generated = tmp_path / 'l515_attached_assembly.STL'
    provenance = build_visual_assembly_stl(
        description / 'urdf/piper_description.xacro',
        generated,
        'gripper_base',
        ('camera_holder', 'l515_visual'),
    )
    committed = description / 'meshes/l515_attached_assembly.STL'

    assert generated.read_bytes() == committed.read_bytes()
    assert provenance['triangle_count'] == 198122
    vertices = []
    for record in read_binary_stl(generated):
        values = TRIANGLE_STRUCT.unpack(record)
        vertices.extend((values[3:6], values[6:9], values[9:12]))
    minimum = tuple(
        min(point[axis] for point in vertices) for axis in range(3))
    maximum = tuple(
        max(point[axis] for point in vertices) for axis in range(3))
    assert minimum == pytest.approx((-0.04, -0.05286335, 0.011), abs=1e-7)
    assert maximum == pytest.approx((0.0995, 0.05286335, 0.064), abs=1e-7)
