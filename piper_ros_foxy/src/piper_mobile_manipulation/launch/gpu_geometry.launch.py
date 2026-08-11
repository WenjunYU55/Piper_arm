from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def cfg(name):
    return os.path.join(get_package_share_directory('piper_mobile_manipulation'), 'config', name)


def generate_launch_description():
    geometry_nodes = [
        Node(
            package='piper_mobile_manipulation',
            executable='motion_compensated_prompt_node.py',
            name='motion_compensated_prompt',
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
            name='sam2_depth_to_3d',
            output='screen',
            parameters=[cfg('camera_params.yaml'), {
                'detection_topic': '/piper/sam2_detection_2d',
                'mask_topic': '/piper/sam2_target_mask',
            }],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='target_tracker_node.py',
            name='sam2_target_tracker',
            output='screen',
            parameters=[cfg('frames.yaml'), cfg('tracking_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='object_frame_broadcaster_node.py',
            name='object_frame_broadcaster',
            output='screen',
            parameters=[cfg('tracking_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='scan_quality_node.py',
            name='sam2_scan_quality',
            output='screen',
            parameters=[cfg('scan_quality_params.yaml'), {
                'mask_topic': '/piper/sam2_target_mask',
                'detection_topic': '/piper/sam2_detection_2d',
                'stale_timeout_sec': 2.5,
            }],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='occlusion_checker_node.py',
            name='sam2_occlusion_checker',
            output='screen',
            parameters=[cfg('occlusion_checker_params.yaml'), {
                'mask_topic': '/piper/sam2_target_mask',
                'detection_topic': '/piper/sam2_detection_2d',
                'stale_timeout_sec': 2.5,
            }],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='obstacle_instance_3d_node.py',
            name='obstacle_instance_3d',
            output='screen',
            parameters=[cfg('obstacle_instance_3d_params.yaml')],
        ),
        Node(
            package='piper_mobile_manipulation',
            executable='target_landmark_node.py',
            name='target_landmark',
            output='screen',
            parameters=[cfg('frames.yaml')],
        ),
    ]
    shutdown_handlers = [
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(
                reason='critical GPU geometry component exited'))],
        ))
        for node in geometry_nodes
    ]
    return LaunchDescription([*shutdown_handlers, *geometry_nodes])
