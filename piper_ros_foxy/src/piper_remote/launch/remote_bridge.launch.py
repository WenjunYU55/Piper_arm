from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    host_arg = DeclareLaunchArgument(
        'host',
        default_value='0.0.0.0',
        description='HTTP bind address. Use 0.0.0.0 to allow other devices on the network.',
    )
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='8080',
        description='HTTP port for remote commands.',
    )

    bridge_node = Node(
        package='piper_remote',
        executable='remote_bridge',
        name='piper_remote_bridge',
        output='screen',
        parameters=[{
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
        }],
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        bridge_node,
    ])
