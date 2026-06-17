cmake_minimum_required(VERSION 3.5)
project(piper_mobile_manipulation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection2D.msg"
  "msg/Target3D.msg"
  "msg/TrackedTarget.msg"
  "msg/TargetError.msg"
  "msg/ServoCommand.msg"
  "msg/HandoffTarget.msg"
  "msg/ManipulationCommand.msg"
  "msg/ManipulationState.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  piper_mobile_manipulation/target_handoff_node.py
  piper_mobile_manipulation/tf_target_transform_node.py
  piper_mobile_manipulation/l515_object_detector_node.py
  piper_mobile_manipulation/depth_to_3d_node.py
  piper_mobile_manipulation/target_tracker_node.py
  piper_mobile_manipulation/vlm_detector_node.py
  piper_mobile_manipulation/yolo_seg_detector_node.py
  piper_mobile_manipulation/manipulation_target_node.py
  piper_mobile_manipulation/safe_servo_node.py
  piper_mobile_manipulation/target_error_node.py
  piper_mobile_manipulation/fake_visual_servo_node.py
  piper_mobile_manipulation/manipulation_state_machine_node.py
  piper_mobile_manipulation/fake_arm_interface_node.py
  piper_mobile_manipulation/scan_viewpoint_planner_node.py
  piper_mobile_manipulation/viewpoint_reachability_filter_node.py
  piper_mobile_manipulation/active_scan_debug_overlay_node.py
  piper_mobile_manipulation/scan_capture_node.py
  piper_mobile_manipulation/scan_quality_node.py
  piper_mobile_manipulation/occlusion_checker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
