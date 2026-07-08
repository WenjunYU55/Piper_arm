cmake_minimum_required(VERSION 3.5)
project(piper_mobile_manipulation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection2D.msg"
  "msg/Target3D.msg"
  "msg/TrackedTarget.msg"
  "msg/TargetError.msg"
  "msg/ServoCommand.msg"
  "msg/HandoffTarget.msg"
  "msg/ManipulationCommand.msg"
  "msg/ManipulationState.msg"
  "msg/ObstacleInstance3D.msg"
  "msg/ObstacleInstance3DArray.msg"
  "msg/TargetEstimate.msg"
  "msg/ScanViewpoint.msg"
  "msg/ScanViewpointArray.msg"
  "msg/ScanStatus.msg"
  "msg/SceneObject.msg"
  "msg/SceneObjectArray.msg"
  "msg/RemovalPlan.msg"
  "srv/ScanCommand.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  piper_mobile_manipulation/target_handoff_node.py
  piper_mobile_manipulation/tf_target_transform_node.py
  piper_mobile_manipulation/depth_to_3d_node.py
  piper_mobile_manipulation/target_tracker_node.py
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
  piper_mobile_manipulation/heavy_refresh_bridge_node.py
  piper_mobile_manipulation/sam2_live_bridge_node.py
  piper_mobile_manipulation/target_cloud_node.py
  piper_mobile_manipulation/mask_to_detection_node.py
  piper_mobile_manipulation/obstacle_instance_3d_node.py
  piper_mobile_manipulation/obstacle_repeatability_validator.py
  piper_mobile_manipulation/target_landmark_node.py
  piper_mobile_manipulation/supervised_cube_workflow_node.py
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
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_obstacle_geometry test/test_obstacle_geometry.py)
  ament_add_pytest_test(test_target_landmark_geometry test/test_target_landmark_geometry.py)
  ament_add_pytest_test(test_supervised_workflow test/test_supervised_workflow.py)
  ament_add_pytest_test(test_icp test/test_icp.py)
  ament_add_pytest_test(test_piper_kinematics test/test_piper_kinematics.py)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
