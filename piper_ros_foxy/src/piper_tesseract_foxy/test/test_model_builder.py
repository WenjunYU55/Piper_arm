from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from piper_tesseract_foxy.model_builder import build_planning_urdf


def test_builder_loads_production_robot_model_and_adds_camera_frame(tmp_path):
    root = Path(__file__).resolve().parents[4]
    xacro = root / 'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro'
    calibration = (
        root / 'L515_camera/calibration/hand_eye/session_20260701_local/'
        'calibration_result.yaml')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({
        'qualified_for_hardware': False,
        'attached_envelopes': [],
    }), encoding='utf-8')
    output = tmp_path / 'piper.urdf'
    build_planning_urdf(xacro, calibration, manifest, output)
    tree = ET.parse(str(output))
    names = {link.get('name') for link in tree.getroot().findall('link')}
    assert 'camera_optical_frame' in names
    assert 'link6' in names
    assert tree.getroot().get(
        '{https://github.com/tesseract-robotics/tesseract}make_convex') == 'true'
    joints = {
        joint.get('name'): joint for joint in tree.getroot().findall('joint')
    }
    assert joints['joint2'].find('limit').get('lower') == '-0.044796192'
    output_text = output.read_text(encoding='utf-8')
    assert '<transmission' not in output_text
    assert '<gazebo' not in output_text


def test_builder_adds_validated_attached_envelope(tmp_path):
    root = Path(__file__).resolve().parents[4]
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({
        'qualified_for_hardware': False,
        'attached_envelopes': [{
            'name': 'camera_envelope',
            'parent_link': 'camera_optical_frame',
            'origin_xyz_m': [0.0, 0.0, -0.02],
            'size_m': [0.08, 0.08, 0.05],
        }],
    }), encoding='utf-8')
    output = tmp_path / 'piper.urdf'
    build_planning_urdf(
        root / 'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro',
        root / 'L515_camera/calibration/hand_eye/session_20260701_local/'
        'calibration_result.yaml',
        manifest,
        output,
    )
    tree = ET.parse(str(output))
    links = {item.get('name'): item for item in tree.getroot().findall('link')}
    assert 'camera_envelope' in links
    assert links['camera_envelope'].find('collision/geometry/box').get('size') == \
        '0.08 0.08 0.05'


@pytest.mark.parametrize('size', ([0.0, 0.1, 0.1], [0.1, -0.1, 0.1]))
def test_builder_rejects_nonpositive_envelope_size(tmp_path, size):
    root = Path(__file__).resolve().parents[4]
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({
        'attached_envelopes': [{
            'name': 'bad_envelope',
            'parent_link': 'link6',
            'origin_xyz_m': [0.0, 0.0, 0.0],
            'size_m': size,
        }],
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='size_m must be positive'):
        build_planning_urdf(
            root / 'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro',
            root / 'L515_camera/calibration/hand_eye/session_20260701_local/'
            'calibration_result.yaml',
            manifest,
            tmp_path / 'piper.urdf',
        )
