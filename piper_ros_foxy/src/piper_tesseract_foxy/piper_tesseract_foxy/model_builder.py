"""Deterministically expand the simple PiPER Xacro and attach planning envelopes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


TESSERACT_NAMESPACE = 'https://github.com/tesseract-robotics/tesseract'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_piper_description_uri(uri, package_root):
    prefix = 'package://piper_description/'
    if not str(uri).startswith(prefix):
        raise ValueError(
            'collision decomposition resource must use package://piper_description/')
    relative = str(uri)[len(prefix):]
    path = (Path(package_root) / relative).resolve()
    root = Path(package_root).resolve()
    if root not in path.parents:
        raise ValueError('collision decomposition resource escapes piper_description')
    return path


def apply_collision_decomposition(root, package_root, policy):
    """Replace selected one-hull link collisions with verified convex pieces."""
    if not policy:
        return
    manifest_uri = str(policy.get('manifest_uri', ''))
    expected_manifest_hash = str(policy.get('manifest_sha256', ''))
    mesh_uri_prefix = str(policy.get(
        'mesh_uri_prefix',
        'package://piper_description/meshes/planning')).rstrip('/')
    manifest_path = resolve_piper_description_uri(manifest_uri, package_root)
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError('collision decomposition manifest hash mismatch')
    with open(manifest_path, 'r', encoding='utf-8') as stream:
        decomposition = json.load(stream)
    if decomposition.get('schema_version') != 1:
        raise ValueError('collision decomposition schema is unsupported')
    links = {
        str(link.get('name')): link for link in root.findall('link')
        if link.get('name')
    }
    for entry in decomposition.get('links', []):
        name = str(entry.get('link', ''))
        if name not in links:
            assembly = entry.get('assembly')
            parent = '' if not isinstance(assembly, dict) else str(
                assembly.get('base_link', ''))
            if not parent or parent not in links:
                raise ValueError(
                    'collision decomposition names unknown link %s' % name)
            link = ET.SubElement(root, 'link', {'name': name})
            collision = ET.SubElement(link, 'collision')
            geometry = ET.SubElement(collision, 'geometry')
            ET.SubElement(geometry, 'mesh', {
                'filename': (
                    'package://piper_description/meshes/%s.STL' % name),
            })
            joint = ET.SubElement(root, 'joint', {
                'name': '%s_to_%s' % (parent, name),
                'type': 'fixed',
            })
            origin_element(joint, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            ET.SubElement(joint, 'parent', {'link': parent})
            ET.SubElement(joint, 'child', {'link': name})
            links[name] = link
        source = Path(package_root) / 'meshes' / (name + '.STL')
        if sha256_file(source) != str(entry.get('source_sha256', '')):
            raise ValueError('%s source collision mesh hash mismatch' % name)
        pieces = entry.get('pieces', [])
        if not pieces:
            raise ValueError('%s collision decomposition contains no pieces' % name)
        link = links[name]
        for collision in list(link.findall('collision')):
            link.remove(collision)
        for piece in pieces:
            filename = str(piece.get('filename', ''))
            path = resolve_piper_description_uri(
                mesh_uri_prefix + '/' + filename, package_root)
            if sha256_file(path) != str(piece.get('sha256', '')):
                raise ValueError('%s collision piece hash mismatch' % filename)
            collision = ET.SubElement(link, 'collision', {
                'name': '%s_%s' % (name, Path(filename).stem),
            })
            geometry = ET.SubElement(collision, 'geometry')
            ET.SubElement(geometry, 'mesh', {
                'filename': mesh_uri_prefix + '/' + filename,
            })


def matrix_to_rpy(matrix):
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def origin_element(parent, xyz, rpy):
    return ET.SubElement(parent, 'origin', {
        'xyz': ' '.join('%.12g' % float(value) for value in xyz),
        'rpy': ' '.join('%.12g' % float(value) for value in rpy),
    })


def add_fixed_box(root, name, parent_link, xyz, rpy, size):
    link = ET.SubElement(root, 'link', {'name': name})
    collision = ET.SubElement(link, 'collision')
    geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(geometry, 'box', {
        'size': ' '.join('%.12g' % float(value) for value in size),
    })
    joint = ET.SubElement(root, 'joint', {'name': name + '_joint', 'type': 'fixed'})
    origin_element(joint, xyz, rpy)
    ET.SubElement(joint, 'parent', {'link': parent_link})
    ET.SubElement(joint, 'child', {'link': name})


def finite_vector(value, length, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError('%s must contain %d finite values' % (label, length))
    return array


def validate_envelopes(envelopes, existing_links):
    validated = []
    names = set(existing_links)
    for index, envelope in enumerate(envelopes):
        if not isinstance(envelope, dict):
            raise ValueError('attached_envelopes[%d] must be a mapping' % index)
        name = str(envelope.get('name', '')).strip()
        parent = str(envelope.get('parent_link', 'link6')).strip()
        if not name or name in names:
            raise ValueError('attached_envelopes[%d] has a missing or duplicate name' % index)
        if parent not in names:
            raise ValueError(
                'attached_envelopes[%d] has unknown parent_link %s' % (index, parent))
        xyz = finite_vector(
            envelope.get('origin_xyz_m'), 3,
            'attached_envelopes[%d].origin_xyz_m' % index)
        rpy = finite_vector(
            envelope.get('origin_rpy_rad', [0.0, 0.0, 0.0]), 3,
            'attached_envelopes[%d].origin_rpy_rad' % index)
        size = finite_vector(
            envelope.get('size_m'), 3,
            'attached_envelopes[%d].size_m' % index)
        if np.any(size <= 0.0):
            raise ValueError('attached_envelopes[%d].size_m must be positive' % index)
        validated.append((name, parent, xyz, rpy, size))
        names.add(name)
    return validated


def build_planning_urdf(xacro_path, calibration_path, manifest_path, output_path):
    tree = ET.parse(str(xacro_path))
    root = tree.getroot()
    ET.register_namespace('tesseract', TESSERACT_NAMESPACE)
    root.set('name', 'piper_planning')
    # Tesseract 0.35 requires this policy to be explicit. Convex conversion is
    # conservative for the current proposal-only detailed collision meshes.
    root.set('{%s}make_convex' % TESSERACT_NAMESPACE, 'true')
    with open(calibration_path, 'r', encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream)
    if calibration.get('status') != 'accepted':
        raise ValueError('hand-eye calibration is not accepted')
    transform = np.asarray(calibration['camera_to_link6']['matrix'], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('camera_to_link6 must be a finite 4x4 matrix')
    with open(manifest_path, 'r', encoding='utf-8') as stream:
        manifest = yaml.safe_load(stream)
    package_root = Path(xacro_path).resolve().parents[1]
    policies = manifest.get('collision_mesh_decompositions')
    if policies is None:
        policies = [manifest.get('collision_mesh_decomposition')]
    if not isinstance(policies, list):
        raise ValueError('collision_mesh_decompositions must be a list')
    for policy in policies:
        apply_collision_decomposition(root, package_root, policy)

    camera = ET.SubElement(root, 'link', {'name': 'camera_optical_frame'})
    camera_joint = ET.SubElement(
        root, 'joint', {'name': 'link6_to_camera_optical', 'type': 'fixed'})
    origin_element(camera_joint, transform[:3, 3], matrix_to_rpy(transform))
    ET.SubElement(camera_joint, 'parent', {'link': 'link6'})
    ET.SubElement(camera_joint, 'child', {'link': camera.get('name')})

    existing_links = {
        str(link.get('name')) for link in root.findall('link')
        if link.get('name')
    }
    for name, parent, xyz, rpy, size in validate_envelopes(
            manifest.get('attached_envelopes', []), existing_links):
        add_fixed_box(
            root,
            name,
            parent,
            xyz,
            rpy,
            size,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # xml.etree.ElementTree.indent was added in Python 3.9, while ROS 2 Foxy
    # uses Python 3.8 on Ubuntu 20.04. Formatting is not part of the model hash,
    # so simply omit pretty-printing on Foxy.
    if hasattr(ET, 'indent'):
        ET.indent(tree, space='  ')
    tree.write(str(output), encoding='utf-8', xml_declaration=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--xacro', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    build_planning_urdf(args.xacro, args.calibration, args.manifest, args.output)


if __name__ == '__main__':
    main()
