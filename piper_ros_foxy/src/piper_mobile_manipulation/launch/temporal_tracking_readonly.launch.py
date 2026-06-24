from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('piper_mobile_manipulation'),
        'config',
        'temporal_tracking_params.yaml',
    )
    return LaunchDescription([
        Node(
            package='piper_mobile_manipulation',
            executable='temporal_mask_tracker_node.py',
            name='temporal_mask_tracker',
            output='screen',
            parameters=[params],
        ),
    ])
