from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('piper_base_frame', default_value='piper_base_link'),
        DeclareLaunchArgument(
            'mission_spool_root',
            default_value='/tmp/piper_target_scan_missions'),
        Node(
            package='piper_mobile_manipulation',
            executable='target_scan_gateway_node.py',
            name='target_scan_gateway',
            output='screen',
            parameters=[{
                'piper_base_frame': LaunchConfiguration('piper_base_frame'),
                'local_base_frame': 'base_link',
                'mission_spool_root': LaunchConfiguration('mission_spool_root'),
            }],
        ),
    ])
