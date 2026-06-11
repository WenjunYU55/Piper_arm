from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def cfg(name):
    return os.path.join(get_package_share_directory('piper_mobile_manipulation'), 'config', name)


def generate_launch_description():
    scan_params = cfg('scan_planning_params.yaml')
    return LaunchDescription([
        Node(
            package='piper_mobile_manipulation',
            executable='scan_viewpoint_planner_node.py',
            name='scan_viewpoint_planner',
            output='screen',
            parameters=[scan_params],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='viewpoint_reachability_filter_node.py',
            name='viewpoint_reachability_filter',
            output='screen',
            parameters=[scan_params],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='active_scan_debug_overlay_node.py',
            name='active_scan_debug_overlay',
            output='screen',
            parameters=[scan_params],
        ),
    ])
