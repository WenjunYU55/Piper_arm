from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('project_root', default_value='/home/prl/Piper_arm'),
        DeclareLaunchArgument('manage_processes', default_value='true'),
        DeclareLaunchArgument('enable_real_arm_motion', default_value='false'),
        DeclareLaunchArgument(
            'motion_speed_profile_qualified', default_value='false'),
        DeclareLaunchArgument('require_gateway_heartbeat', default_value='false'),
        DeclareLaunchArgument(
            'mission_spool_root',
            default_value='/tmp/piper_target_scan_missions'),
        DeclareLaunchArgument(
            'free_motion_speed_percent', default_value='30.0'),
        DeclareLaunchArgument(
            'contact_speed_percent', default_value='10.0'),
        # Apply the same loopback-only UDP transport as the supported shell
        # entry point even when an operator invokes ``ros2 launch`` directly.
        # This keeps foreign domain-42 participants and stale Fast DDS shared
        # memory state out of the arm-control graph.  Child driver, camera,
        # perception and scan processes inherit these settings.
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            PathJoinSubstitution([
                LaunchConfiguration('project_root'),
                'fastdds_gui_udp_only.xml',
            ])),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '0'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        Node(
            package='piper_mobile_manipulation',
            executable='target_scan_mission_node.py',
            name='target_scan_mission',
            output='screen',
            parameters=[{
                'project_root': LaunchConfiguration('project_root'),
                'manage_processes': ParameterValue(
                    LaunchConfiguration('manage_processes'), value_type=bool),
                'enable_real_arm_motion': ParameterValue(
                    LaunchConfiguration('enable_real_arm_motion'), value_type=bool),
                'motion_speed_profile_qualified': ParameterValue(
                    LaunchConfiguration('motion_speed_profile_qualified'),
                    value_type=bool),
                'require_gateway_heartbeat': ParameterValue(
                    LaunchConfiguration('require_gateway_heartbeat'), value_type=bool),
                'mission_spool_root': LaunchConfiguration('mission_spool_root'),
                'free_motion_speed_percent': ParameterValue(
                    LaunchConfiguration('free_motion_speed_percent'),
                    value_type=float),
                'contact_speed_percent': ParameterValue(
                    LaunchConfiguration('contact_speed_percent'),
                    value_type=float),
                'required_captures': 13,
            }],
        ),
    ])
