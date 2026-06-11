from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def cfg(name):
    return os.path.join(get_package_share_directory('piper_mobile_manipulation'), 'config', name)


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='piper_mobile_manipulation',
            executable='vlm_detector_node.py',
            name='vlm_detector',
            output='screen',
            parameters=[cfg('vlm_detection_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='depth_to_3d_node.py',
            name='depth_to_3d',
            output='screen',
            parameters=[cfg('camera_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='target_tracker_node.py',
            name='target_tracker',
            output='screen',
            parameters=[cfg('tracking_params.yaml'), cfg('frames.yaml')],
        ),
    ])
