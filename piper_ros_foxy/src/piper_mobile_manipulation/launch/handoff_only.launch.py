from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('piper_mobile_manipulation')
    frames = os.path.join(pkg, 'config', 'frames.yaml')
    return LaunchDescription([
        Node(
            package='piper_mobile_manipulation',
            executable='target_handoff_node.py',
            name='target_handoff',
            output='screen',
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='tf_target_transform_node.py',
            name='tf_target_transform',
            output='screen',
            parameters=[frames],
        ),
    ])
