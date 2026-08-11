import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    root = os.environ.get('PIPER_ARM_ROOT', '/home/prl/Piper_arm')
    package_share = get_package_share_directory('piper_tesseract_foxy')
    runtime_root = os.environ.get('XDG_RUNTIME_DIR', '/tmp')
    return LaunchDescription([
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
                    'srdf_path': os.path.join(package_share, 'model', 'piper.srdf'),
                    'collision_manifest_path': os.path.join(
                        package_share, 'model', 'collision_model.yaml'),
                },
            ],
        ),
    ])
