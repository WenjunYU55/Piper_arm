from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare the launch arguments
    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='CAN port to be used by the Piper node.'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='false',
        description='Explicit motor-enable opt-in; the supported system leaves this false.'
    )

    joint_ctrl_topic_arg = DeclareLaunchArgument(
        'joint_ctrl_topic',
        default_value='/joint_ctrl_single',
        description='JointState command topic consumed by the Piper node.'
    )

    gripper_exist_arg = DeclareLaunchArgument(
        'gripper_exist',
        default_value='true',
        description='gripper'
    )
    enable_timeout_arg = DeclareLaunchArgument(
        'enable_timeout',
        default_value='15.0',
        description='Seconds to wait for all motors to report enabled.'
    )
    joint_bounds_path_arg = DeclareLaunchArgument(
        'joint_bounds_path',
        default_value='',
        description='JSON file containing hard joint command bounds.'
    )
    # Define the node
    piper_node = Node(
        package='piper',
        executable='piper_single_ctrl',
        name='piper_ctrl_single_node',
        output='screen',
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_exist': LaunchConfiguration('gripper_exist'),
            'enable_timeout': LaunchConfiguration('enable_timeout'),
            'joint_bounds_path': LaunchConfiguration('joint_bounds_path'),
        }],
        remappings=[
            ('joint_ctrl_single', LaunchConfiguration('joint_ctrl_topic')),
        ]
    )

    # Return the LaunchDescription
    return LaunchDescription([
        can_port_arg,
        auto_enable_arg,
        joint_ctrl_topic_arg,
        gripper_exist_arg,
        enable_timeout_arg,
        joint_bounds_path_arg,
        piper_node
    ])
