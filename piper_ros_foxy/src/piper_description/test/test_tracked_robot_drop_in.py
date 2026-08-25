import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


DESCRIPTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DESCRIPTION_ROOT.parents[2]
BUNDLE_ROOT = (
    PROJECT_ROOT / 'integration/track_robot_description/drop_in')
ARM_PATH = (
    BUNDLE_ROOT / 'src/piper_description/urdf/piper_description.xacro')
TRACKED_PATH = (
    BUNDLE_ROOT / 'src/bunker_pro2/urdf/bunker_pro2.urdf')

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
    'base_to_arm_base_joint',
}


def _signature(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or '').strip(),
        tuple(_signature(child) for child in element),
    )


def _mapped_arm_element(source, tag):
    element = copy.deepcopy(source)
    if tag == 'link' and element.attrib['name'] == 'base_link':
        element.attrib['name'] = 'arm_base_link'
    for reference in element.findall('parent') + element.findall('child'):
        if reference.attrib.get('link') == 'base_link':
            reference.attrib['link'] = 'arm_base_link'
    return element


def test_drop_in_files_and_manifest_hashes_are_current():
    assert ARM_PATH.is_file()
    assert TRACKED_PATH.is_file()
    manifest = json.loads(
        (BUNDLE_ROOT / 'description_bundle_manifest.json').read_text())
    for relative_path, expected in manifest['outputs'].items():
        actual = hashlib.sha256(
            (BUNDLE_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected


def test_tracked_description_has_one_root_and_gateway_arm_frame():
    root = ET.parse(str(TRACKED_PATH)).getroot()
    links = [link.attrib['name'] for link in root.findall('link')]
    joints = [joint.attrib['name'] for joint in root.findall('joint')]
    assert len(links) == len(set(links))
    assert len(joints) == len(set(joints))
    children = {
        joint.find('child').attrib['link'] for joint in root.findall('joint')}
    assert set(links) - children == {'robot_bottom'}
    parents = {
        joint.find('child').attrib['link']:
        joint.find('parent').attrib['link']
        for joint in root.findall('joint')
    }
    assert parents['base_link'] == 'robot_bottom'
    assert parents['arm_base_link'] == 'base_link'
    assert parents['piper_base_link'] == 'arm_base_link'
    assert parents['link1'] == 'arm_base_link'


def test_tracked_description_does_not_duplicate_platform_geometry():
    root = ET.parse(str(TRACKED_PATH)).getroot()
    links = {link.attrib['name'] for link in root.findall('link')}
    assert 'bunker_chassis_collision' not in links
    assert 'bunker_sensor_station_collision' not in links
    assert 'piper_base_link' in links
    alias = root.find("./link[@name='piper_base_link']")
    assert alias.find('visual') is None
    assert alias.find('collision') is None


def test_combined_arm_subtree_exactly_matches_drop_in_arm_source():
    arm = ET.parse(str(ARM_PATH)).getroot()
    tracked = ET.parse(str(TRACKED_PATH)).getroot()
    expected_links = {}
    expected_joints = {}
    for link in arm.findall('link'):
        if link.attrib['name'] == 'world':
            continue
        mapped = _mapped_arm_element(link, 'link')
        expected_links[mapped.attrib['name']] = _signature(mapped)
    for joint in arm.findall('joint'):
        if joint.attrib['name'] == 'fixed_base_joint':
            continue
        mapped = _mapped_arm_element(joint, 'joint')
        expected_joints[mapped.attrib['name']] = _signature(mapped)
    actual_links = {
        link.attrib['name']: _signature(link)
        for link in tracked.findall('link')
        if link.attrib['name'] not in TRACKED_LINKS
    }
    actual_joints = {
        joint.attrib['name']: _signature(joint)
        for joint in tracked.findall('joint')
        if joint.attrib['name'] not in TRACKED_JOINTS
    }
    assert actual_links == expected_links
    assert actual_joints == expected_joints


def test_mount_and_ground_geometry_match_arm_planning_profile():
    root = ET.parse(str(TRACKED_PATH)).getroot()
    mount = root.find("./joint[@name='base_to_arm_base_joint']/origin")
    assert mount.attrib['xyz'] == '0.39 0 0.016'
    assert mount.attrib['rpy'] == '0 0 0'
    base_height = root.find(
        "./joint[@name='robot_bottom_to_base_link']/origin")
    assert float(base_height.attrib['xyz'].split()[2]) == 0.45
    assert 0.45 + 0.016 == 0.466
    ground_manifest = PROJECT_ROOT / (
        'piper_ros_foxy/src/piper_tesseract_foxy/model/'
        'collision_model_ground.yaml')
    assert 'floor_z_m: -0.466' in ground_manifest.read_text()


def test_gateway_default_matches_drop_in_integration_frame():
    gateway = PROJECT_ROOT / (
        'piper_ros_foxy/src/piper_mobile_manipulation/'
        'piper_mobile_manipulation/target_scan_gateway_node.py')
    source = gateway.read_text()
    assert "self.declare_parameter('piper_base_frame', 'piper_base_link')" in source
    assert "self.declare_parameter('local_base_frame', 'base_link')" in source
    assert "lookup_transform(\n                piper_base, source," in source
