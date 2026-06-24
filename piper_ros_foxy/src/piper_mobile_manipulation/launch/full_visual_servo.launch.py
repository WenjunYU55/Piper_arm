from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def cfg(name):
    return os.path.join(get_package_share_directory('piper_mobile_manipulation'), 'config', name)


def generate_launch_description():
    enable_real_arm_motion = LaunchConfiguration('enable_real_arm_motion')

    return LaunchDescription([
        DeclareLaunchArgument('enable_real_arm_motion', default_value='false'),
        Node(
            package='piper_mobile_manipulation',
            executable='l515_object_detector_node.py',
            name='l515_object_detector',
            output='screen',
            parameters=[cfg('detection_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='depth_to_3d_node.py',
            name='depth_to_3d',
            output='screen',
            parameters=[cfg('camera_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='target_tracker_node.py',
            name='target_tracker',
            output='screen',
            parameters=[cfg('tracking_params.yaml'), cfg('frames.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='manipulation_target_node.py',
            name='manipulation_target',
            output='screen',
            parameters=[cfg('manipulation_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='safe_servo_node.py',
            name='safe_servo',
            output='screen',
            parameters=[
                cfg('manipulation_params.yaml'),
                {'enable_real_arm_motion': enable_real_arm_motion},
            ],
        ),
    ])
