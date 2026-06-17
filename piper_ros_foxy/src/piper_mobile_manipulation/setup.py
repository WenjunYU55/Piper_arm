from setuptools import find_packages, setup

package_name = 'piper_mobile_manipulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Safe mobile manipulation stack for PiPER arm perception and fake commands.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_handoff = piper_mobile_manipulation.target_handoff_node:main',
            'tf_target_transform = piper_mobile_manipulation.tf_target_transform_node:main',
            'l515_object_detector = piper_mobile_manipulation.l515_object_detector_node:main',
            'depth_to_3d = piper_mobile_manipulation.depth_to_3d_node:main',
            'target_tracker = piper_mobile_manipulation.target_tracker_node:main',
            'vlm_detector = piper_mobile_manipulation.vlm_detector_node:main',
            'yolo_seg_detector = piper_mobile_manipulation.yolo_seg_detector_node:main',
            'manipulation_target = piper_mobile_manipulation.manipulation_target_node:main',
            'safe_servo = piper_mobile_manipulation.safe_servo_node:main',
            'target_error = piper_mobile_manipulation.target_error_node:main',
            'fake_visual_servo = piper_mobile_manipulation.fake_visual_servo_node:main',
            'manipulation_state_machine = piper_mobile_manipulation.manipulation_state_machine_node:main',
            'fake_arm_interface = piper_mobile_manipulation.fake_arm_interface_node:main',
            'scan_viewpoint_planner = piper_mobile_manipulation.scan_viewpoint_planner_node:main',
            'viewpoint_reachability_filter = piper_mobile_manipulation.viewpoint_reachability_filter_node:main',
            'active_scan_debug_overlay = piper_mobile_manipulation.active_scan_debug_overlay_node:main',
            'scan_capture = piper_mobile_manipulation.scan_capture_node:main',
            'scan_quality = piper_mobile_manipulation.scan_quality_node:main',
            'occlusion_checker = piper_mobile_manipulation.occlusion_checker_node:main',
        ],
    },
)
