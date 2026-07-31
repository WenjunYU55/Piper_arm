"""Read-only visualization of the real PiPER feedback.

This launch deliberately contains no joint command publisher and no fake joint
state publisher.  robot_state_publisher consumes the driver's feedback topic
and publishes the URDF link chain to TF.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare(package="piper_description").find("piper_description")
    urdf_path = os.path.join(package_share, "urdf", "piper_description.xacro")
    rviz_path = os.path.join(package_share, "rviz", "rviz.rviz")
    # The current .xacro is plain URDF XML (it contains no xacro macros), so
    # loading it directly avoids requiring the optional xacro CLI at runtime.
    with open(urdf_path, encoding="utf-8") as description_file:
        robot_description = description_file.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start the read-only RViz view.",
        ),
        DeclareLaunchArgument(
            "rvizconfig",
            default_value=rviz_path,
            description="Absolute path to the RViz configuration.",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="piper_live_robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", "/joint_states_single")],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="piper_live_rviz",
            output="screen",
            arguments=["-d", LaunchConfiguration("rvizconfig")],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ])
