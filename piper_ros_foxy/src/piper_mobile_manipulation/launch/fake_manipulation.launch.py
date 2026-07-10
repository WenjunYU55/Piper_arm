from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def cfg(name):
    return os.path.join(get_package_share_directory('piper_mobile_manipulation'), 'config', name)


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='piper_mobile_manipulation',
            executable='target_handoff_node.py',
            name='target_handoff',
            output='screen',
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='tf_target_transform_node.py',
            name='tf_target_transform',
            output='screen',
            parameters=[cfg('frames.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='mask_to_detection_node.py',
            name='sam2_mask_to_detection',
            output='screen',
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='depth_to_3d_node.py',
            name='depth_to_3d',
            output='screen',
            parameters=[cfg('camera_params.yaml'), {
                'detection_topic': '/piper/sam2_detection_2d',
                'mask_topic': '/piper/sam2_target_mask',
            }],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='target_tracker_node.py',
            name='target_tracker',
            output='screen',
            parameters=[cfg('frames.yaml'), cfg('tracking_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='target_error_node.py',
            name='target_error',
            output='screen',
            parameters=[cfg('target_error_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='fake_visual_servo_node.py',
            name='fake_visual_servo',
            output='screen',
            parameters=[cfg('fake_visual_servo_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='manipulation_state_machine_node.py',
            name='manipulation_state_machine',
            output='screen',
            parameters=[
                cfg('manipulation_params.yaml'),
                cfg('safety_params.yaml'),
            ],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='fake_arm_interface_node.py',
            name='fake_arm_interface',
            output='screen',
        ),
    ])
