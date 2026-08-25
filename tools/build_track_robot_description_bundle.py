#!/usr/bin/env python3
"""Build the tracked-robot-rooted PiPER description replacement bundle."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


TRACKED_REPOSITORY = 'https://github.com/ZZY5825/Track-robot-workspace'
TRACKED_COMMIT = 'c8bf4db35c7a196aa26c0add0f2549fa1c973980'
TRACKED_URDF_URL = (
    TRACKED_REPOSITORY + '/raw/' + TRACKED_COMMIT
    + '/src/bunker_pro2/urdf/bunker_pro2.urdf')

TRACKED_LINKS = {
    'robot_bottom',
    'base_link',
    'sensor_station_link',
    'camera_mount_link',
    'zed_camera_link',
    'lidar_link',
}
TRACKED_JOINTS = {
    'robot_bottom_to_base_link',
    'sensor_station_joint',
    'sensor_station_camera_mount_joint',
    'camera_mount_to_zed_camera_joint',
    'sensor_station_lidar_joint',
}
LOCAL_PLATFORM_LINKS = {
    'bunker_chassis_collision',
    'bunker_sensor_station_collision',
}
LOCAL_PLATFORM_JOINTS = {
    'base_link_to_bunker_chassis_collision',
    'base_link_to_bunker_sensor_station_collision',
}


def _indent(element, level=0):
    """Apply deterministic two-space XML indentation on Python 3.8."""
    whitespace = '\n' + level * '  '
    child_whitespace = '\n' + (level + 1) * '  '
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_whitespace
        for child in element:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_whitespace
        element[-1].tail = whitespace
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _load_tracked_urdf(path):
    if path:
        return Path(path).read_bytes()
    with urllib.request.urlopen(TRACKED_URDF_URL, timeout=30) as response:
        return response.read()


def _add_gateway_frame(arm_root):
    if arm_root.find("./link[@name='piper_base_link']") is not None:
        raise ValueError('source arm description already defines piper_base_link')
    alias_link = ET.Element('link', {'name': 'piper_base_link'})
    alias_joint = ET.Element(
        'joint', {'name': 'arm_base_to_piper_base_frame', 'type': 'fixed'})
    ET.SubElement(alias_joint, 'origin', {
        'xyz': '0 0 0',
        'rpy': '0 0 0',
    })
    ET.SubElement(alias_joint, 'parent', {'link': 'base_link'})
    ET.SubElement(alias_joint, 'child', {'link': 'piper_base_link'})
    arm_root.append(alias_link)
    arm_root.append(alias_joint)


def _arm_only_description(source_root):
    arm_root = copy.deepcopy(source_root)
    for link in list(arm_root.findall('link')):
        if link.attrib.get('name') in LOCAL_PLATFORM_LINKS:
            arm_root.remove(link)
    for joint in list(arm_root.findall('joint')):
        if joint.attrib.get('name') in LOCAL_PLATFORM_JOINTS:
            arm_root.remove(joint)
    _add_gateway_frame(arm_root)
    return arm_root


def _map_arm_element(source, tag):
    element = copy.deepcopy(source)
    if tag == 'link' and element.attrib['name'] == 'base_link':
        element.attrib['name'] = 'arm_base_link'
    for reference in element.findall('parent') + element.findall('child'):
        if reference.attrib.get('link') == 'base_link':
            reference.attrib['link'] = 'arm_base_link'
    return element


def _mount_joint():
    joint = ET.Element(
        'joint', {'name': 'base_to_arm_base_joint', 'type': 'fixed'})
    ET.SubElement(joint, 'origin', {
        'xyz': '0.39 0 0.016',
        'rpy': '0 0 0',
    })
    ET.SubElement(joint, 'parent', {'link': 'base_link'})
    ET.SubElement(joint, 'child', {'link': 'arm_base_link'})
    return joint


def _combined_description(tracked_root, arm_root):
    combined = ET.Element('robot', {'name': 'bunker_pro2'})
    for element in tracked_root:
        if element.tag == 'link' and element.attrib.get('name') in TRACKED_LINKS:
            combined.append(copy.deepcopy(element))
        elif (
                element.tag == 'joint'
                and element.attrib.get('name') in TRACKED_JOINTS):
            combined.append(copy.deepcopy(element))
    combined.append(_mount_joint())
    for link in arm_root.findall('link'):
        if link.attrib['name'] != 'world':
            combined.append(_map_arm_element(link, 'link'))
    for joint in arm_root.findall('joint'):
        if joint.attrib['name'] != 'fixed_base_joint':
            combined.append(_map_arm_element(joint, 'joint'))
    return combined


def _parent_map(root):
    return {
        joint.find('child').attrib['link']:
        joint.find('parent').attrib['link']
        for joint in root.findall('joint')
    }


def _validate(arm_root, combined):
    for root in (arm_root, combined):
        links = [item.attrib['name'] for item in root.findall('link')]
        joints = [item.attrib['name'] for item in root.findall('joint')]
        if len(links) != len(set(links)) or len(joints) != len(set(joints)):
            raise ValueError('generated description contains duplicate names')
    combined_links = {
        item.attrib['name'] for item in combined.findall('link')}
    if LOCAL_PLATFORM_LINKS & combined_links:
        raise ValueError('generated tracked description duplicates platform geometry')
    parents = _parent_map(combined)
    expected = {
        'base_link': 'robot_bottom',
        'arm_base_link': 'base_link',
        'piper_base_link': 'arm_base_link',
        'link1': 'arm_base_link',
    }
    for child, parent in expected.items():
        if parents.get(child) != parent:
            raise ValueError(
                'expected %s -> %s, found %s'
                % (parent, child, parents.get(child)))
    joint6 = combined.find("./joint[@name='joint6']/limit")
    if joint6 is None or joint6.attrib.get('lower') != '-3.141592653589793' \
            or joint6.attrib.get('upper') != '3.141592653589793':
        raise ValueError('generated description did not retain current J6 limits')


def _xml_bytes(root):
    root = copy.deepcopy(root)
    _indent(root)
    return (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(root, encoding='utf-8') + b'\n')


def build_bundle(project_root, tracked_urdf, output_root):
    project_root = Path(project_root).resolve()
    source_path = (
        project_root / 'piper_ros_foxy/src/piper_description/urdf'
        / 'piper_description.xacro')
    tracked_data = _load_tracked_urdf(tracked_urdf)
    source_data = source_path.read_bytes()
    arm_root = _arm_only_description(ET.fromstring(source_data))
    combined = _combined_description(ET.fromstring(tracked_data), arm_root)
    _validate(arm_root, combined)

    output_root = Path(output_root).resolve()
    arm_output = (
        output_root / 'src/piper_description/urdf/piper_description.xacro')
    tracked_output = (
        output_root / 'src/bunker_pro2/urdf/bunker_pro2.urdf')
    arm_output.parent.mkdir(parents=True, exist_ok=True)
    tracked_output.parent.mkdir(parents=True, exist_ok=True)
    arm_bytes = _xml_bytes(arm_root)
    tracked_bytes = _xml_bytes(combined)
    arm_output.write_bytes(arm_bytes)
    tracked_output.write_bytes(tracked_bytes)

    manifest = {
        'schema_version': 1,
        'tracked_source_repository': TRACKED_REPOSITORY,
        'tracked_source_commit': TRACKED_COMMIT,
        'tracked_source_sha256': _sha256(tracked_data),
        'local_arm_source': str(source_path.relative_to(project_root)),
        'local_arm_source_sha256': _sha256(source_data),
        'outputs': {
            str(arm_output.relative_to(output_root)): _sha256(arm_bytes),
            str(tracked_output.relative_to(output_root)): _sha256(tracked_bytes),
        },
        'tf_contract': {
            'tracked_root': 'robot_bottom',
            'tracked_base': 'base_link',
            'arm_model_root': 'arm_base_link',
            'gateway_arm_frame': 'piper_base_link',
            'base_to_arm_xyz_m': [0.39, 0.0, 0.016],
            'base_to_arm_rpy_rad': [0.0, 0.0, 0.0],
        },
    }
    manifest_path = output_root / 'description_bundle_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    return arm_output, tracked_output, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument('--project-root', default=str(default_root))
    parser.add_argument(
        '--tracked-urdf', default='',
        help='optional local upstream bunker_pro2.urdf; otherwise use pinned GitHub commit')
    parser.add_argument(
        '--output-root',
        default=str(
            default_root / 'integration/track_robot_description/drop_in'))
    args = parser.parse_args()
    outputs = build_bundle(
        args.project_root, args.tracked_urdf or None, args.output_root)
    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
