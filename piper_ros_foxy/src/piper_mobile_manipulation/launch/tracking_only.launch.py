from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('piper_mobile_manipulation')
    return LaunchDescription([
        Node(
            package='piper_mobile_manipulation',
            executable='target_tracker_node.py',
            name='target_tracker',
            output='screen',
            parameters=[
                os.path.join(pkg, 'config', 'tracking_params.yaml'),
                os.path.join(pkg, 'config', 'frames.yaml'),
            ],
        ),
    ])
