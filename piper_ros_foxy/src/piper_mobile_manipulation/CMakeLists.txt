cmake_minimum_required(VERSION 3.5)
project(piper_mobile_manipulation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(piper_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/RunTargetScan.action"
  "msg/Detection2D.msg"
  "msg/Target3D.msg"
  "msg/TrackedTarget.msg"
  "msg/TrackingHealth.msg"
  "msg/CameraTimestampHealth.msg"
  "msg/TargetError.msg"
  "msg/ServoCommand.msg"
  "msg/HandoffTarget.msg"
  "msg/ManipulationCommand.msg"
  "msg/ManipulationState.msg"
  "msg/ObstacleInstance3D.msg"
  "msg/ObstacleInstance3DArray.msg"
  "msg/ScanExecutionPlan.msg"
  "msg/ScanExecutionStatus.msg"
  "msg/MotionPlan.msg"
  "msg/MotionPlanStatus.msg"
  "msg/PlannerReadiness.msg"
  "msg/TesseractPlan.msg"
  "msg/TesseractPlanStatus.msg"
  "msg/TesseractReadiness.msg"
  "msg/OccluderAction.msg"
  "msg/MeshJobStatus.msg"
  "srv/AuthorizeMission.srv"
  "srv/GetTargetScanResult.srv"
  "srv/ApproveScanExecution.srv"
  "srv/ExecuteHomeStage.srv"
  "srv/PrepareAcquisition.srv"
  "srv/RequestMotionPlan.srv"
  "srv/RequestTesseractPlan.srv"
  "srv/ReportTrackedRobotHomed.srv"
  "srv/GetMeshJobResult.srv"
  DEPENDENCIES action_msgs builtin_interfaces geometry_msgs std_msgs trajectory_msgs
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
  piper_mobile_manipulation/scan_target_acquisition_node.py
  piper_mobile_manipulation/viewpoint_reachability_filter_node.py
  piper_mobile_manipulation/active_scan_debug_overlay_node.py
  piper_mobile_manipulation/scan_capture_node.py
  piper_mobile_manipulation/scan_quality_node.py
  piper_mobile_manipulation/occlusion_checker_node.py
  piper_mobile_manipulation/heavy_refresh_bridge_node.py
  piper_mobile_manipulation/sam2_live_bridge_node.py
  piper_mobile_manipulation/motion_compensated_prompt_node.py
  piper_mobile_manipulation/target_cloud_node.py
  piper_mobile_manipulation/mask_to_detection_node.py
  piper_mobile_manipulation/obstacle_instance_3d_node.py
  piper_mobile_manipulation/obstacle_repeatability_validator.py
  piper_mobile_manipulation/target_landmark_node.py
  piper_mobile_manipulation/object_frame_broadcaster_node.py
  piper_mobile_manipulation/supervised_cube_workflow_node.py
  piper_mobile_manipulation/scan_viewpoint_executor_node.py
  piper_mobile_manipulation/camera_timestamp_watchdog_node.py
  piper_mobile_manipulation/target_scan_mission_node.py
  piper_mobile_manipulation/target_scan_gateway_node.py
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
  ament_add_pytest_test(
    test_collision_environment test/test_collision_environment.py)
  ament_add_pytest_test(test_target_landmark_geometry test/test_target_landmark_geometry.py)
  ament_add_pytest_test(test_supervised_workflow test/test_supervised_workflow.py)
  ament_add_pytest_test(
    test_heavy_refresh_bridge test/test_heavy_refresh_bridge.py)
  ament_add_pytest_test(test_tracking_recovery test/test_tracking_recovery.py)
  ament_add_pytest_test(test_depth_to_3d test/test_depth_to_3d.py)
  ament_add_pytest_test(test_mask_reprojection test/test_mask_reprojection.py)
  ament_add_pytest_test(test_scan_motion test/test_scan_motion.py)
  ament_add_pytest_test(
    test_scan_session_memory test/test_scan_session_memory.py)
  ament_add_pytest_test(test_scan_capture test/test_scan_capture.py)
  ament_add_pytest_test(
    test_scan_target_acquisition test/test_scan_target_acquisition.py)
  ament_add_pytest_test(
    test_camera_timestamp_health test/test_camera_timestamp_health.py)
  ament_add_pytest_test(test_mission_core test/test_mission_core.py)
  ament_add_pytest_test(test_home_pose test/test_home_pose.py)
  ament_add_pytest_test(test_startup_gates test/test_startup_gates.py)
  ament_add_pytest_test(test_occlusion_policy test/test_occlusion_policy.py)
  ament_add_pytest_test(
    test_reconstruction_jobs test/test_reconstruction_jobs.py)
  ament_add_pytest_test(
    test_phase1_mission_characterization
    test/test_phase1_mission_characterization.py)
  ament_add_pytest_test(
    test_phase1_executor_characterization
    test/test_phase1_executor_characterization.py)
  ament_add_pytest_test(
    test_phase1_external_contract_characterization
    test/test_phase1_external_contract_characterization.py)
  ament_add_pytest_test(
    test_failure_model test/test_failure_model.py)
  ament_add_pytest_test(
    test_telemetry_store test/test_telemetry_store.py)
  ament_add_pytest_test(
    test_process_supervisor test/test_process_supervisor.py)
  ament_add_pytest_test(
    test_safety_evaluator test/test_safety_evaluator.py)
  ament_add_pytest_test(
    test_mission_engine test/test_mission_engine.py)
  ament_add_pytest_test(
    test_phase7_viewpoint_components
    test/test_phase7_viewpoint_components.py)
  ament_add_pytest_test(test_nbv_coverage test/test_nbv_coverage.py)
  ament_add_pytest_test(
    test_nbv_regression_replay test/test_nbv_regression_replay.py)
  ament_add_pytest_test(test_view_generation test/test_view_generation.py)
  ament_add_pytest_test(
    test_capability_map test/test_capability_map.py)
  ament_add_pytest_test(
    test_target_envelope test/test_target_envelope.py)
  ament_add_pytest_test(
    test_scan_viewpoint_angles test/test_scan_viewpoint_angles.py)
  ament_add_pytest_test(
    test_ray_mission_diagnostics test/test_ray_mission_diagnostics.py)
  ament_add_pytest_test(test_configuration test/test_configuration.py)
  ament_add_pytest_test(
    test_planner_backend test/test_planner_backend.py)
  ament_add_pytest_test(
    test_architecture_boundaries test/test_architecture_boundaries.py)
  ament_add_pytest_test(
    test_mission_resources test/test_mission_resources.py)
  ament_add_pytest_test(
    test_gpu_geometry_launch test/test_gpu_geometry_launch.py)
  ament_add_pytest_test(
    test_reintegrated_features test/test_reintegrated_features.py)
  ament_add_pytest_test(
    test_surface_coverage test/test_surface_coverage.py)
  ament_add_pytest_test(
    test_target_cloud_refined_cache test/test_target_cloud_refined_cache.py)
  ament_add_pytest_test(
    test_viewpoint_feedback_freshness
    test/test_viewpoint_feedback_freshness.py)
  ament_add_pytest_test(
    test_viewpoint_rays test/test_viewpoint_rays.py)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
