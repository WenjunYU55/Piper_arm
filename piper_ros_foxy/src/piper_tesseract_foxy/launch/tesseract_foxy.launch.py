import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description():
    root = os.environ.get('PIPER_ARM_ROOT', '/home/prl/Piper_arm')
    package_share = get_package_share_directory('piper_tesseract_foxy')
    runtime_root = os.environ.get('XDG_RUNTIME_DIR', '/tmp')
    profile = LaunchConfiguration('floor_profile')
    profile_is_ground = ["'", profile, "' == 'ground'"]
    manifest_name = PythonExpression([
        "'collision_model_ground.yaml' if ", *profile_is_ground,
        " else 'collision_model.yaml'",
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'floor_profile',
            default_value=os.environ.get(
                'PIPER_FLOOR_PROFILE', 'tabletop'),
            choices=['tabletop', 'ground'],
            description=(
                'Startup-only floor; combined platform geometry is invariant.'),
        ),
        Node(
            package='piper_tesseract_foxy',
            executable='tesseract_plan_bridge',
            name='tesseract_plan_bridge',
            output='screen',
            parameters=[
                os.path.join(package_share, 'config', 'tesseract_bridge_params.yaml'),
                {
                    'spool_root': os.path.join(runtime_root, 'piper_tesseract_plans'),
                    'hand_eye_calibration_path': os.path.join(
                        root,
                        'L515_camera/calibration/hand_eye/session_20260808_straight_mount/'
                        'calibration_result.yaml',
                    ),
                    'joint_bounds_path': os.path.join(root, 'piper_joint_bounds.json'),
                    'robot_xacro_path': os.path.join(
                        root,
                        'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro',
                    ),
                    'srdf_path': PathJoinSubstitution([
                        package_share, 'model', 'piper_bunker.srdf']),
                    'collision_manifest_path': PathJoinSubstitution([
                        package_share, 'model', manifest_name]),
                },
            ],
        ),
    ])
