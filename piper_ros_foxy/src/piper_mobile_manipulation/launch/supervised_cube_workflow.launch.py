from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory('piper_mobile_manipulation'),
                          'config', 'supervised_cube_workflow_params.yaml')
    return LaunchDescription([
        Node(package='piper_mobile_manipulation',
             executable='supervised_cube_workflow_node.py',
             name='supervised_cube_workflow', output='screen', parameters=[config])
    ])
