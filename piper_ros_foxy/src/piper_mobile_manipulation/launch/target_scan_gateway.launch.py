from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('piper_base_frame', default_value='piper_base_link'),
        DeclareLaunchArgument(
            'mission_spool_root',
            default_value='/tmp/piper_target_scan_missions'),
        DeclareLaunchArgument('project_root', default_value='/home/prl/Piper_arm'),
        DeclareLaunchArgument('reconstruction_python', default_value=''),
        DeclareLaunchArgument('max_pending_missions', default_value='8'),
        Node(
            package='piper_mobile_manipulation',
            executable='target_scan_gateway_node.py',
            name='target_scan_gateway',
            output='screen',
            parameters=[{
                'piper_base_frame': LaunchConfiguration('piper_base_frame'),
                'local_base_frame': 'base_link',
                'mission_spool_root': LaunchConfiguration('mission_spool_root'),
                'project_root': LaunchConfiguration('project_root'),
                'reconstruction_python': LaunchConfiguration(
                    'reconstruction_python'),
                'max_pending_missions': ParameterValue(
                    LaunchConfiguration('max_pending_missions'), value_type=int),
            }],
        ),
    ])
