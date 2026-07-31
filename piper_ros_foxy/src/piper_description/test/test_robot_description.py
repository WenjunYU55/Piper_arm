import math
from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET

import numpy as np


DESCRIPTION_ROOT = Path(__file__).resolve().parents[1]


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _origin_transform(origin):
    xyz = np.asarray([float(value) for value in origin.attrib["xyz"].split()])
    roll, pitch, yaw = [float(value) for value in origin.attrib["rpy"].split()]
    result = np.eye(4)
    result[:3, :3] = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    result[:3, 3] = xyz
    return result


def _urdf_fk(positions):
    root = ET.parse(DESCRIPTION_ROOT / "urdf" / "piper_description.xacro").getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    result = np.eye(4)
    for index, position in enumerate(positions, start=1):
        result = result @ _origin_transform(joints[f"joint{index}"].find("origin"))
        rotation = np.eye(4)
        rotation[:3, :3] = _rotation_z(position)
        result = result @ rotation
    return result


def _sdk_mode_0_fk(positions):
    a = np.asarray([0, 0, 285.03, -21.98, 0, 0]) / 1000.0
    alpha = np.asarray([0, -math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2])
    theta_offset = np.asarray([0, -math.radians(174.22), -math.radians(100.78), 0, 0, 0])
    d = np.asarray([123, 0, 0, 250.75, 0, 91]) / 1000.0
    result = np.eye(4)
    for index, position in enumerate(positions):
        ca, sa = math.cos(alpha[index]), math.sin(alpha[index])
        angle = position + theta_offset[index]
        ct, st = math.cos(angle), math.sin(angle)
        result = result @ np.asarray([
            [ct, -st, 0, a[index]],
            [st * ca, ct * ca, -sa, -sa * d[index]],
            [st * sa, ct * sa, ca, ca * d[index]],
            [0, 0, 0, 1],
        ])
    return result


def test_urdf_chain_matches_controller_mode_0_fk():
    poses = [
        np.zeros(6),
        np.asarray([0.4, 0.8, -0.7, 0.3, -0.4, 0.5]),
        np.asarray([-0.8, 1.5, -1.2, -0.5, 0.6, -1.0]),
    ]
    for pose in poses:
        assert np.allclose(_urdf_fk(pose), _sdk_mode_0_fk(pose), atol=1e-9)


def test_live_launch_is_feedback_only():
    launch_text = (DESCRIPTION_ROOT / "launch" / "display_live_robot.launch.py").read_text()
    assert 'remappings=[("joint_states", "/joint_states_single")]' in launch_text
    assert "joint_state_publisher_gui" not in launch_text
    assert "joint_ctrl_single" not in launch_text


def test_legacy_control_and_simulation_surfaces_are_absent():
    assert not (DESCRIPTION_ROOT / "launch" / "display_xacro.launch.py").exists()
    assert not (
        DESCRIPTION_ROOT / "config" / "piper_gazebo_control.yaml"
    ).exists()
    assert not (
        DESCRIPTION_ROOT / "config" / "joint_names_agx_arm_description.yaml"
    ).exists()
    package_text = (DESCRIPTION_ROOT / "package.xml").read_text()
    model_text = (
        DESCRIPTION_ROOT / "urdf" / "piper_description.xacro"
    ).read_text()
    assert "joint_state_publisher_gui" not in package_text
    assert "xmlns:xacro" not in model_text
    assert "<transmission" not in model_text
    assert "<gazebo" not in model_text


def _preview_module():
    path = DESCRIPTION_ROOT / "scripts" / "piper_joint_preview_node.py"
    spec = importlib.util.spec_from_file_location("piper_joint_preview_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preview_joint_angle_round_trip_and_limits():
    module = _preview_module()
    fixed = module.rpy_quaternion(0.4, -0.2, 1.1)
    for angle in (-2.0, -0.3, 0.0, 0.8, 2.7):
        marker = module.quaternion_multiply(fixed, module.z_rotation_quaternion(angle))
        assert math.isclose(module.relative_z_angle(fixed, marker), angle, abs_tol=1e-9)
    assert module.clamp(4.0, -1.0, 2.0) == 2.0


def test_joint_preview_launch_never_names_real_command_topic():
    launch_text = (DESCRIPTION_ROOT / "launch" / "joint_preview.launch.py").read_text()
    node_text = (DESCRIPTION_ROOT / "scripts" / "piper_joint_preview_node.py").read_text()
    assert "/piper_gui/preview_joint_states" in launch_text
    assert "/piper_gui/preview_set" in node_text
    assert "joint_ctrl_single" not in launch_text
    assert "joint_ctrl_single" not in node_text


def test_joint_preview_has_large_visible_rotation_grab_ring():
    node_text = (DESCRIPTION_ROOT / "scripts" / "piper_joint_preview_node.py").read_text()
    assert "marker.scale = 0.30" in node_text
    assert "Marker.LINE_STRIP" in node_text
    assert "ring_radius = 0.115" in node_text
