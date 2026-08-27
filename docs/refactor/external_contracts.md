# External contracts baseline

This inventory is backward-compatibility authority for Phase 1. Names, message
fields, type constants, QoS compatibility, JSON keys, frame meanings, file
formats, environment variables and process entry points listed here must remain
compatible unless a later phase explicitly versions and migrates them.

The 2026-08-20 previous-generation reconciliation is internal and additive:
no ROS action, topic, service, QoS, parameter, TF, motion, timeout, or result
schema changed. A processing-only stale generation is cleaned through exact
stored handles; a live prior driver still blocks unless fresh valid feedback
proves all six motors disabled.

## ROS actions

| Name | Type | Server(s) | Contract |
|---|---|---|---|
| `/piper/run_target_scan` | `piper_mobile_manipulation/action/RunTargetScan` | `target_scan_mission`; `target_scan_gateway` in gateway mode | Goal carries task identity/type, target label/profile, rough `PoseWithCovarianceStamped`, confidence and deadline. Feedback carries phase/reason/timing/acquisition/occlusion/capture/process health/shutdown. Result carries outcome, failure code, retryability, safe-shutdown proof, dataset/manifest/capture/mesh identity and JSON summary. |

Outcome constants are fixed at: succeeded 0, failed 1, cancelled 2, busy 3,
unsupported profile 4, needs operator 5, reposition required 6.

## ROS services

| Name | Type | Owner | Purpose |
|---|---|---|---|
| `/enable_srv` | `piper_msgs/srv/Enable` | PiPER driver | Boolean all-axis enable/disable request and confirmation |
| `/piper/get_target_scan_result` | `GetTargetScanResult` | mission/gateway | Idempotent task result JSON lookup |
| `/piper/report_tracked_robot_homed` | `ReportTrackedRobotHomed` | gateway | Authorizes deferred reconstruction after base home using task/job/manifest identity |
| `/piper/get_mesh_job_result` | `GetMeshJobResult` | gateway | Mesh result JSON lookup by job ID |
| `/scan_target_acquisition/prepare` | `PrepareAcquisition` | acquisition | Atomically binds session ID, look index and rough `PointStamped` |
| `/tesseract_plan_bridge/request_plan` | `RequestTesseractPlan` | Tesseract bridge | Requests one multiview plan snapshot |
| `/tesseract_plan_bridge/request_acquisition_plan` | `RequestTesseractPlan` | Tesseract bridge | Requests rough-acquisition planning |
| `/tesseract_plan_bridge/request_return_home_plan` | `RequestTesseractPlan` | Tesseract bridge | Requests a staged direct home target |
| `/tesseract_plan_bridge/request_startup_home_plan` | `RequestTesseractPlan` | Tesseract bridge | Requests startup staged home target |
| `/scan_viewpoint_executor/approve` | `ApproveScanExecution` | executor | Binds plan ID, trajectory hash and exact confirmation/mission authority |
| `/scan_viewpoint_executor/authorize_mission` | `AuthorizeMission` | executor | Grants/revokes task/hash/deadline authority |
| `/scan_viewpoint_executor/execute_home_stage` | `ExecuteHomeStage` | executor | Executes one mission-authorized configured STARTUP_WRIST, PRE_HOME, ROUGH_HOME, or STORAGE_WRIST endpoint without Tesseract |
| `/scan_viewpoint_executor/hold` | `std_srvs/Trigger` | executor | Commands current-feedback hold |
| `/scan_viewpoint_executor/cancel` | `std_srvs/Trigger` | executor | Stops motion and commands hold |
| `/scan_viewpoint_executor/refresh_plan` | `std_srvs/Trigger` | executor | Invalidates/refreshes proposal state |
| `/scan_viewpoint_executor/diagnostic_state` | `std_srvs/Trigger` | executor | JSON diagnostic snapshot |
| `/scan_capture/capture_view` | `std_srvs/Trigger` | capture | Service-mode synchronized RGB-D capture |
| `/supervised_cube_workflow/start` | `std_srvs/Trigger` | workflow | Begins measured-lock/occlusion assessment |
| `/supervised_cube_workflow/approve_plan` | `std_srvs/Trigger` | workflow | Manual proposal approval path |
| `/supervised_cube_workflow/confirm_action_complete` | `std_srvs/Trigger` | workflow | Manual contact-action completion confirmation |
| `/supervised_cube_workflow/capture_view` | `std_srvs/Trigger` | workflow | Advances workflow capture state |
| `/supervised_cube_workflow/finish_scan` | `std_srvs/Trigger` | workflow | Finalizes workflow scan |
| `/supervised_cube_workflow/abort` | `std_srvs/Trigger` | workflow | Aborts workflow |
| `/supervised_cube_workflow/diagnostic_state` | `std_srvs/Trigger` | workflow | JSON workflow diagnostic snapshot |
| `*/capture_sample`, `*/finalize` | `std_srvs/Trigger` | obstacle validator | Repeatability validation tools |

`RequestTesseractPlan` retains its existing home fields for backward
compatibility, but the production mission now uses `ExecuteHomeStage` for
configured home. Its request binds `task_id`, `mission_sha256`, one of
`STARTUP_WRIST`, `PRE_HOME`, `ROUGH_HOME`, `STORAGE_WRIST`, and exactly six
finite configured joint radians. Response correlation is `accepted`,
`execution_id`, and `message`.

## ROS topics

### Driver and command authority

| Topic | Type | Direction/owner | QoS |
|---|---|---|---|
| `/joint_ctrl_single` | `sensor_msgs/JointState` | Command input; executor or manual GUI must be the sole publisher | depth 10 |
| `/pos_cmd` | `piper_msgs/PosCmd` | Legacy Cartesian driver input | depth 10 |
| `/enable_flag` | `std_msgs/Bool` | Legacy enable input | depth 10 |
| `/joint_states_single` | `sensor_msgs/JointState` | Driver feedback | depth 10; critical consumers explicitly use reliable/volatile where declared |
| `/arm_status` | `piper_msgs/PiperStatusMsg` | Driver aggregate and per-motor state | depth 10; safety consumers reliable/volatile |
| `/end_pose` | `geometry_msgs/Pose` | Driver FK/end-pose feedback | depth 10 |
| `/piper/motion_limits` | `piper_msgs/PiperMotionLimits` | Driver controller-derived velocity/acceleration limits and digest | depth 10 |

`JointState.position[0:6]` is the six arm joints in canonical joint order;
generic sequence types are accepted. Command `velocity[0]` carries PiPER MoveJ
speed percent and `effort[0]` carries the SDK mode/effort convention used by
the driver. Do not reinterpret these as a ROS trajectory controller interface.

### Camera and perception

| Topic | Type | Producer -> consumers |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RealSense -> bridges, quality, capture, overlay |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RealSense -> geometry/watchdog/capture |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | RealSense -> depth geometry, occlusion, quality, capture |
| `/piper/heavy_refresh_request` | `std_msgs/String` JSON | Executor/SAM2/geometry -> heavy bridge |
| `/piper/heavy_refresh_status` | `std_msgs/String` JSON | Heavy bridge -> executor/workflow/SAM2/GUI |
| `/piper/heavy_target_mask` | `sensor_msgs/Image` | Heavy bridge -> SAM2 seed/target cloud |
| `/piper/candidate_movable_obstacle_mask` | `sensor_msgs/Image` | Heavy bridge -> diagnostics |
| `/piper/unsafe_obstacle_mask` | `sensor_msgs/Image` | Heavy bridge -> diagnostics |
| `/piper/heavy_obstacle_mask` | `sensor_msgs/Image` | Heavy bridge -> diagnostics |
| `/piper/sam2_target_mask` | `sensor_msgs/Image` | SAM2 bridge -> geometry/quality/capture |
| `/piper/sam2_obstacle_mask` | `sensor_msgs/Image` | SAM2 bridge -> diagnostics |
| `/piper/sam2_unsafe_obstacle_mask` | `sensor_msgs/Image` | SAM2 bridge -> diagnostics |
| `/piper/sam2_candidate_movable_obstacle_mask` | `sensor_msgs/Image` | SAM2 bridge -> diagnostics |
| `/piper/sam2_object_ids` | `sensor_msgs/Image` | SAM2 bridge -> obstacle geometry |
| `/piper/sam2_tracking_status` | `std_msgs/String` JSON | SAM2 bridge -> geometry/diagnostics |
| `/piper/motion_compensated_target_prompt` | `sensor_msgs/Image` | Motion prompt -> SAM2 bridge |
| `/piper/motion_compensated_prompt_status` | `std_msgs/String` JSON | Motion prompt diagnostics |
| `/piper/sam2_detection_2d` | `Detection2D` | Mask geometry -> depth/quality/occlusion |
| `/piper/target_3d` | `Target3D` | Depth geometry -> tracker/quality/capture |
| `/piper/tracked_target` | `TrackedTarget` | Tracker -> planner/executor/workflow/TF |
| `/piper/tracking_health` | `TrackingHealth` | SAM2 bridge -> Tesseract bridge/executor/GUI; mission consumes the derived readiness contract |
| `/piper/target_status` | `std_msgs/String` | Tracker -> planning/execution/workflow |
| `/piper/camera_timestamp_health` | `CameraTimestampHealth` | Watchdog -> planning/execution/mission |
| `/piper/target_landmark` | `geometry_msgs/PointStamped` | Landmark -> workflow |
| `/piper/target_landmark_projection` | `Detection2D` | Landmark diagnostic projection |
| `/piper/target_landmark_status` | `std_msgs/String` JSON | Landmark -> workflow |
| `/piper/obstacle_instances_3d` | `ObstacleInstance3DArray` | Obstacle geometry -> bridge/executor/workflow |
| `/piper/target_cloud` | `sensor_msgs/PointCloud2` | Target cloud -> workflow/RViz |
| `/piper/scene_cloud` | `sensor_msgs/PointCloud2` | Scene cloud -> workflow/RViz |
| `/piper/target_cloud_request` | `std_msgs/String` JSON | Workflow -> target cloud |
| `/piper/target_cloud_status` | `std_msgs/String` JSON | Target cloud -> workflow |

Image, CameraInfo and PointCloud2 paths use Foxy's
`qos_profile_sensor_data` (best-effort/volatile) unless explicitly overridden.
This compatibility is part of the camera pipeline contract.

### Planning, execution, capture and mission

| Topic | Type | Producer -> consumers | QoS |
|---|---|---|---|
| `/piper/acquisition_viewpoints` | `std_msgs/String` JSON | Acquisition -> reachability/Tesseract | depth 10 |
| `/piper/reachable_acquisition_viewpoints` | `std_msgs/String` JSON | Reachability -> acquisition/Tesseract | depth 10 |
| `/piper/scan_viewpoints` | `std_msgs/String` JSON | Planner -> reachability/debug/capture | depth 10 |
| `/piper/reachable_scan_viewpoints` | `std_msgs/String` JSON | Reachability -> bridge/executor/quality/capture | depth 10 |
| `/piper/scan_coverage` | `std_msgs/String` JSON | Planner -> debug/capture | depth 10 |
| `/piper/tesseract_plan` | `TesseractPlan` | Bridge -> executor | reliable, transient-local, depth 1 |
| `/piper/tesseract_plan_status` | `TesseractPlanStatus` | Bridge diagnostics | depth 10 |
| `/piper/tesseract_readiness` | `TesseractReadiness` | Bridge -> mission/GUI | reliable, volatile, depth 1 |
| `/piper/scan_execution_plan` | `ScanExecutionPlan` | Executor -> mission | reliable, transient-local, depth 1 |
| `/piper/scan_execution_status` | `ScanExecutionStatus` | Executor -> mission/workflow/capture | depth 10 |
| `/piper/scan_session_history` | `std_msgs/String` JSON | Executor -> planner/mission | reliable, transient-local, depth 1 |
| `/piper/scan_quality` | `std_msgs/String` JSON | Quality -> workflow/capture/debug | depth 10 |
| `/piper/scan_quality_debug` | `std_msgs/String` JSON | Quality diagnostics | depth 10 |
| `/piper/useful_scan_coverage` | `std_msgs/String` JSON | Quality -> debug | depth 10 |
| `/piper/occlusion_status` | `std_msgs/String` JSON | Occlusion checker -> workflow/capture/SAM2 | depth 10 |
| `/piper/occlusion_debug` | `std_msgs/String` JSON | Occlusion diagnostics | depth 10 |
| `/piper/supervised_workflow_status` | `std_msgs/String` JSON | Workflow -> executor/acquisition/mission/GUI | depth 10 |
| `/piper/removal_plan` | `std_msgs/String` JSON | Workflow proposal | depth 10 |
| `/piper/target_model` | `std_msgs/String` JSON | Workflow model diagnostic | depth 10 |
| `/piper/supervised_workflow_markers` | `visualization_msgs/MarkerArray` | Workflow -> RViz | depth 10 |
| `/piper/scan_capture_status` | `std_msgs/String` JSON | Capture -> quality/mission/GUI | depth 10 |
| `/piper/scan_summary` | `std_msgs/String` JSON | Capture terminal summary | depth 10 |
| `/piper/active_scan_debug_image` | `sensor_msgs/Image` | Overlay -> viewer | sensor data |
| `/piper/mesh_job_status` | `MeshJobStatus` | Gateway -> GUI/tracked robot | reliable, transient-local, depth 10 |

### Legacy/manual GUI topics

`/piper_gui/preview_set`, `/piper_gui/preview_joint_states`,
`/base/target_pose`, `/piper/handoff_target`, `/piper/target_piper_base`,
`/piper/target_camera`, `/piper/target_error`, `/piper/servo_cmd`,
`/piper/manipulation_target`, `/piper/manipulation_state`, and
`/piper/manipulation_command` remain public commissioning interfaces.

## Message types

`piper_msgs` defines `PosCmd`, `PiperStatusMsg`, `PiperMotionLimits`, and
`Enable`. `PiperStatusMsg`'s per-axis driver-enabled fields,
`motor_feedback_valid`, `motor_faults`, and `motor_watchdog_reason` are safety
authority and cannot be dropped in favor of aggregate `arm_status` alone.

`piper_mobile_manipulation` defines:

- Perception/tracking: `Detection2D`, `Target3D`, `TrackedTarget`,
  `TrackingHealth`, `CameraTimestampHealth`.
- Legacy manipulation: `TargetError`, `ServoCommand`, `HandoffTarget`,
  `ManipulationCommand`, `ManipulationState`.
- Obstacle/removal: `ObstacleInstance3D`, `ObstacleInstance3DArray`,
  `OccluderAction`.
- Planning/execution: `ScanExecutionPlan`, `ScanExecutionStatus`,
  `TesseractPlan`, `TesseractPlanStatus`, `TesseractReadiness`.
- Reconstruction: `MeshJobStatus`.

Obstacle classification numeric constants (blocked 0, movable 1, unsafe 2),
occluder action constants (none 0, pick/place 1, push 2), plan-kind strings,
field order and semantics are wire contracts.

## JSON-over-String contracts

The following are not untyped implementation details. Consumers classify exact
keys and, in several cases, exact state/reason text:

- Acquisition/reachable candidate payloads: header stamp/frame, request/session
  correlation, target center/provenance, candidate IDs, camera positions, look
  directions and coverage metadata.
- `/piper/heavy_refresh_request` and status: request/job ID, reason, target
  prompt/profile, image stamp, state, object/mask metadata and errors.
- `/piper/sam2_tracking_status`: generation, object IDs/labels/classes,
  confidence, state and source stamp.
- Quality/occlusion: state/label, score, target/depth/mask validity, age,
  closer-depth ratios, visibility ratios and reason.
- Workflow diagnostic/status: state, reason, session, measured-lock readiness,
  occlusion probe correlation, accepted/modeled views and input ages.
- Scan capture status/history: dataset/session identity, accepted/rejected view,
  actual camera pose/look, quality/occlusion, surface gain, manifest and counts.
- Executor diagnostic/status: plan/state/reason, current/total view, speed,
  tracking scale, maximum joint error and approval requirements.

Before extraction, Phase 1 must characterize these payloads with golden tests;
silently changing a key, capitalization, state or reason fragment can alter
retry, cancellation or failure classification.

## TF and frame contracts

- `base_link` is the local arm planning/world frame in the current bench setup.
- `odom` is accepted on an incoming rough goal and snapshotted to `base_link`
  at mission start.
- `piper_base_link` is the future tracked-robot integration frame exposed by
  gateway/handoff contracts; it is not interchangeable with `base_link`
  without a transform.
- `camera_link`, `l515_visual`, `camera_depth_optical_frame`, and
  `camera_color_optical_frame` retain RealSense optical conventions.
- The accepted calibration stores `T_link6_camera_optical`; composition order
  and quaternion convention must remain unchanged.
- Captures store the capture-time camera-to-base transform and exact image
  stamp. Latest TF/depth must never be paired with an older heavy result.

## Persisted files and spools

| Contract | Default location | Compatibility requirement |
|---|---|---|
| Home profile | `piper_home_pose.json` | Schema 3, six joint names/positions, mission-ready/storage J6, direction fields and SHA-256 |
| Joint bounds | `piper_joint_bounds.json` | Per-joint min/max/unit/validity and source metadata |
| Hand-eye calibration | active August 8 session YAML | Accepted status, transform convention, mechanical registration and validation provenance |
| Tesseract spool | `/tmp/piper_tesseract_plans` | Schema/hash/TTL/boot-generation/request-response/worker-health semantics |
| Heavy spool | `/tmp/piper_heavy_refresh` | Atomic job directories, `READY`, archived RGB-D/intrinsics/masks and correlated result |
| SAM2 spool | `/tmp/piper_sam2_live` | Labelled seed manifests/masks and generation state |
| Mission spool | `/tmp/piper_target_scan_missions` | Goals/status/results/heartbeat, task ID and mission/result hashes |
| Capture dataset | `datasets/active_scan/scan_*` | Synchronized per-view bundle, metadata and terminal manifest |
| Reconstruction job | mission spool/job result | Manifest hash, mesh path/hash and quality-report JSON |
| Process registry/logs | runtime directory | Exact owned process groups and generation-scoped logs |

## Environment and command-line contracts

Mission composition exports `PIPER_ARM_ROOT`, `PIPER_AUTO_ENABLE`,
`PIPER_ENABLE_REAL_VIEWPOINT_MOTION`, `PIPER_VIEWPOINT_MISSION_POLICY`,
`PIPER_VIEWPOINT_CLOSED_LOOP_ONE_VIEW`, `PIPER_VIEWPOINT_SPEED_PERCENT`,
`PIPER_VIEWPOINT_MAX_VIEWS`, `PIPER_VIEWPOINT_MIN_VIEWS`,
`PIPER_RETURN_HOME_POSITIONS_RAD`, `PIPER_MISSION_TASK_ID`,
`PIPER_MISSION_SHA256`, `PIPER_TARGET_LABEL`, `PIPER_TARGET_PROFILE`, and
`PIPER_TARGET_PROMPT`. Tesseract/vision scripts also consume their documented
spool, ROS domain/RMW/FastDDS and model-path variables.

Stable executable/operator entry points include `start_piper.sh`,
`disable_piper.py`, `start_gui.sh`, `run_target_scan_mission.sh`,
`run_target_scan_gateway.sh`, the `L515_camera/run_*` and `stop_*` scripts,
`motion_planning/tesseract/{run_worker,qualify_rootless_worker}.sh`, and ROS
console scripts `piper_single_ctrl`, `tesseract_plan_bridge`, and
`tesseract_plan_worker`.

## Complete ROS parameter declaration catalog

This is the exhaustive source-level parameter-name inventory for Python ROS nodes in the workspace at the baseline commit. Defaults remain authoritative in the named source files and deployed overrides in the configuration/launch files listed in `current_architecture.md`.

Phase 9 preserves every name in this catalog. The coordinator and executor
defaults are now owned by `piper_mobile_manipulation/configuration.py`, while
the named nodes remain the ROS declaration/override boundaries. Independent
default-equivalence tests freeze all 98 values.

- `piper_ros_foxy/src/piper/piper/piper_ctrl_single_node.py`: `auto_enable`, `can_port`, `enable_timeout`, `gripper_exist`, `joint_bounds_path`, `motion_limit_max_age_sec`, `motion_limit_query_period_sec`.
- `piper_ros_foxy/src/piper_description/scripts/piper_joint_preview_node.py`: `frame_prefix`, `urdf_path`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/active_scan_debug_overlay_node.py`: `color_image_topic`, `debug_image_topic`, `detection_debug_image_topic`, `detection_topic`, `dry_run`, `enable_real_arm_motion`, `occlusion_status_topic`, `prefer_detection_debug_image`, `reachable_scan_viewpoints_topic`, `scan_coverage_topic`, `scan_quality_topic`, `scan_stale_timeout_s`, `scan_viewpoints_topic`, `stale_timeout_s`, `target_3d_topic`, `useful_scan_coverage_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/camera_timestamp_watchdog_node.py`: `backward_tolerance_sec`, `enable_recovery_request`, `frame_timeout_sec`, `health_topic`, `healthy_frames_required`, `image_topic`, `joint_state_timeout_sec`, `joint_states_topic`, `joint_subscription_retry_sec`, `max_timestamp_offset_sec`, `publish_period_sec`, `recovery_request_path`, `startup_grace_sec`, `stationary_duration_sec`, `stationary_position_tolerance_rad`, `stationary_velocity_rad_s`, `unhealthy_frames_before_recovery`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/depth_to_3d_node.py`: `bbox_scale`, `camera_info_topic`, `confidence_depth_stddev_m`, `crop_half_size_px`, `debug`, `depth_jump_reacquire_samples`, `depth_jump_reacquire_tolerance_m`, `depth_max_m`, `depth_min_m`, `depth_percentile`, `depth_topic`, `detection_topic`, `mask_erode_px`, `mask_max_age_s`, `mask_topic`, `max_depth_jump_m`, `max_depth_m`, `max_depth_stddev_m`, `min_depth_m`, `min_valid_depth_pixels`, `min_valid_depth_ratio`, `roi_half_size_px`, `smoothing_alpha`, `sync_queue_size`, `sync_slop_sec`, `target_topic`, `use_detection_bbox`, `use_mask_depth`, `use_median_depth`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/fake_arm_interface_node.py`: `command_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/fake_visual_servo_node.py`: `command_deadband_m`, `gain_xy`, `gain_z`, `max_forward_speed_mps`, `max_lateral_speed_mps`, `max_vertical_speed_mps`, `servo_cmd_topic`, `target_error_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/heavy_refresh_bridge_node.py`: `all_obstacle_mask_topic`, `camera_info_topic`, `color_image_topic`, `depth_image_topic`, `dry_run`, `enable_real_arm_motion`, `idle_status_interval_sec`, `max_image_age_sec`, `min_target_depth_valid_px`, `min_target_depth_valid_ratio`, `movable_obstacle_mask_topic`, `output_mask_topic`, `request_topic`, `response_poll_period_sec`, `rgbd_sync_queue_size`, `rgbd_sync_slop_sec`, `sam2_live_spool_dir`, `seed_sam2_live`, `spool_dir`, `status_topic`, `tracked_mask_topic`, `unsafe_obstacle_mask_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/manipulation_state_machine_node.py`: `command_period_s`, `command_topic`, `final_servo_step_m`, `grab_distance_threshold_m`, `handoff_topic`, `pre_grasp_offset_m`, `pre_push_offset_m`, `push_distance_m`, `push_speed_mps`, `require_stable_before_grab`, `require_valid_depth`, `require_valid_tf`, `state_topic`, `task_command`, `tracked_topic`, `workspace_x_max`, `workspace_x_min`, `workspace_y_max`, `workspace_y_min`, `workspace_z_max`, `workspace_z_min`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/manipulation_target_node.py`: `approach_offset_m`, `manipulation_mode`, `manipulation_target_topic`, `stop_on_low_confidence`, `stop_on_target_lost`, `target_status_topic`, `target_type`, `tracked_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/mask_to_detection_node.py`: `detection_topic`, `mask_topic`, `min_area_px`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/motion_compensated_prompt_node.py`: `base_frame`, `camera_info_topic`, `color_topic`, `depth_scale`, `depth_sync_tolerance_sec`, `depth_topic`, `mask_topic`, `max_point_age_sec`, `max_prediction_horizon_sec`, `min_support_points`, `output_topic`, `point_stride`, `prompt_dilation_px`, `status_topic`, `tracked_target_topic`, `transform_timeout_sec`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/object_frame_broadcaster_node.py`: `min_confidence`, `object_frame`, `predicted_object_frame`, `publish_predicted_frame`, `republish_hz`, `tracked_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/obstacle_instance_3d_node.py`: `base_frame`, `bounds_high_percentile`, `bounds_low_percentile`, `camera_info_topic`, `depth_max_m`, `depth_min_m`, `depth_topic`, `heavy_spool_dir`, `heavy_status_topic`, `mask_erode_px`, `max_source_age_sec`, `max_transform_age_sec`, `metadata_topic`, `metadata_wait_sec`, `min_valid_depth_pixels`, `min_valid_depth_ratio`, `movable_whitelist`, `object_ids_topic`, `output_topic`, `sync_queue_size`, `sync_slop_sec`, `transform_listener_retry_sec`, `transform_listener_stall_sec`, `transform_timeout_sec`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/obstacle_repeatability_validator.py`: `expect_scene_blocked`, `expected_label`, `input_topic`, `max_drift_m`, `max_sample_age_sec`, `max_samples`, `min_samples`, `report_dir`, `scenario`, `stability_max_drift_m`, `stability_observations`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/occlusion_checker_node.py`: `color_image_topic`, `debug`, `depth_image_topic`, `detection_topic`, `dry_run`, `edge_margin_px`, `enable_real_arm_motion`, `evaluation_interval_sec`, `heavy_occlusion_ratio`, `heavy_visible_ratio`, `lost_transition_confirmations`, `mask_topic`, `max_valid_depth_m`, `min_mask_area_px`, `min_occluder_area_px`, `min_reference_mask_area_px`, `min_valid_depth_m`, `min_valid_depth_ratio`, `near_mask_dilation_px`, `occlusion_debug_topic`, `occlusion_depth_margin_m`, `occlusion_persistence_frames`, `occlusion_status_topic`, `partial_occlusion_ratio`, `partial_visible_ratio`, `reference_initialization_frames`, `reference_update_alpha`, `scan_execution_status_topic`, `scan_quality_topic`, `stale_timeout_sec`, `state_transition_confirmations`, `target_3d_topic`, `use_reference_mask_area`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/safe_servo_node.py`: `enable_real_arm_motion`, `gain_xy`, `gain_z`, `manipulation_target_topic`, `max_speed`, `max_target_jump_m`, `min_depth_m`, `servo_cmd_topic`, `target_status_topic`, `tracking_health_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/sam2_live_bridge_node.py`: `absent_retry_sec`, `allow_heavy_topic_seed`, `arm_motion_window_sec`, `arm_moving_position_delta_rad`, `arm_moving_threshold_rad_s`, `arm_settled_position_delta_rad`, `arm_settled_threshold_rad_s`, `auto_initial_mask`, `camera_settle_time_sec`, `camera_timestamp_health_timeout_sec`, `camera_timestamp_health_topic`, `color_image_topic`, `degraded_speed_scale`, `frame_rate_hz`, `health_frame_id`, `heavy_request_ack_timeout_sec`, `heavy_request_topic`, `heavy_status_topic`, `joint_states_topic`, `lost_refresh_retry_sec`, `low_confidence_refresh_duration_sec`, `low_confidence_refresh_hysteresis`, `low_confidence_refresh_threshold`, `max_reacquisition_attempts`, `min_target_area_px`, `motion_prompt_max_age_sec`, `motion_prompt_recovery_grace_sec`, `motion_prompt_topic`, `movable_obstacle_mask_topic`, `no_mask_refresh_timeout_sec`, `object_ids_topic`, `obstacle_mask_topic`, `occlusion_status_topic`, `output_mask_topic`, `recovery_valid_frames`, `refresh_cooldown_sec`, `seed_cache_sec`, `seed_mask_topic`, `semantic_refresh_interval_sec`, `spool_dir`, `status_topic`, `target_status_topic`, `tracked_target_topic`, `tracking_health_topic`, `tracking_measurement_stale_sec`, `unsafe_obstacle_mask_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/scan_capture_node.py`: `base_frame`, `calibration_sha256`, `camera_info_topic`, `camera_optical_frame`, `camera_transform_timeout_sec`, `capture_interval_sec`, `capture_mode`, `color_image_topic`, `dataset_root`, `debug`, `depth_image_topic`, `diagnostic_timeout_sec`, `dry_run`, `enable_real_arm_motion`, `joint_state_topic`, `mask_topic`, `max_bundle_age_sec`, `max_frames_per_scan`, `minimum_accepted_quality_score`, `mission_sha256`, `occlusion_status_topic`, `reachable_scan_viewpoints_topic`, `require_camera_transform`, `require_clear_occlusion_for_service`, `require_depth`, `require_good_quality_for_service`, `require_mask`, `require_valid_target`, `scan_capture_status_topic`, `scan_coverage_topic`, `scan_execution_status_topic`, `scan_quality_topic`, `scan_summary_topic`, `scan_viewpoints_topic`, `synchronization_slop_sec`, `target_3d_topic`, `target_label`, `target_profile`, `target_prompt`, `task_id`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/scan_quality_node.py`: `color_image_topic`, `debug`, `depth_image_topic`, `detection_topic`, `dry_run`, `edge_margin_px`, `enable_real_arm_motion`, `evaluation_interval_sec`, `mask_topic`, `max_depth_stddev_good_m`, `max_valid_depth_m`, `min_acceptable_scan_quality`, `min_good_scan_quality`, `min_mask_area_px`, `min_valid_depth_m`, `min_valid_depth_ratio`, `reachable_scan_viewpoints_topic`, `scan_capture_status_topic`, `scan_quality_debug_topic`, `scan_quality_topic`, `stale_timeout_sec`, `target_3d_topic`, `useful_scan_coverage_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/scan_target_acquisition_node.py`: `acquisition_camera_pitch_deg`, `acquisition_viewpoints_topic`, `base_frame`, `camera_optical_frame`, `dry_run`, `fallback_standoff_m`, `future_tolerance_sec`, `handoff_retry_sec`, `handoff_timeout_sec`, `hint_max_age_sec`, `reachable_acquisition_viewpoints_topic`, `request_acquisition_plan_service`, `scan_execution_status_topic`, `standoff_m`, `sweep_angle_deg`, `transform_timeout_sec`, `workflow_start_service`, `workflow_status_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/scan_viewpoint_executor_node.py`: `acquisition_fresh_frame_timeout_sec`, `acquisition_grounding_timeout_sec`, `acquisition_max_viewpoints`, `acquisition_scene_timeout_sec`, `acquisition_target_tolerance_m`, `acquisition_tracking_lock_timeout_sec`, `allow_mission_policy`, `allow_target_motion_during_scan`, `approval_confirmation`, `arm_status_topic`, `auto_capture`, `camera_holder_envelope_center_link6_m`, `camera_holder_envelope_size_m`, `camera_holder_external_clearance_m`, `camera_timestamp_health_topic`, `capture_service`, `capture_status_propagation_sec`, `capture_timeout_sec`, `closed_loop_one_view`, `configured_home_feedback_limit_tolerance_rad`, `data_timeout_sec`, `debug`, `enable_real_arm_motion`, `endpoint_position_settled_rad`, `executor_tick_rate_hz`, `finish_scan_service`, `finish_scan_timeout_sec`, `floor_z_m`, `hand_eye_calibration_path`, `heavy_refresh_request_topic`, `heavy_refresh_status_topic`, `home_goal_tolerance_rad`, `home_joint_feedback_timeout_sec`, `home_motion_tolerance_rad`, `home_settle_duration_sec`, `home_settle_timeout_sec`, `joint_bounds_path`, `joint_command_topic`, `joint_feedback_limit_tolerance_rad`, `joint_goal_tolerance_rad`, `joint_states_topic`, `joint_velocity_settled`, `link_radius_m`, `max_execution_viewpoints`, `max_target_drift_before_approval_m`, `max_tracking_measurement_age_sec`, `min_execution_viewpoints`, `min_tracking_speed_scale`, `motion_limits_change_confirmation_sec`, `motion_limits_change_minimum_samples`, `motion_limits_timeout_sec`, `motion_limits_topic`, `obstacle_topic`, `plan_max_age_sec`, `plan_start_tolerance_rad`, `plan_topic`, `reachable_viewpoints_topic`, `return_home_positions_rad`, `rgbd_capture_service`, `runtime_recovery_timeout_sec`, `runtime_refresh_timeout_sec`, `scan_session_history_topic`, `scan_target_max_boresight_deg`, `scan_target_min_distance_m`, `self_clearance_m`, `settle_duration_sec`, `settle_timeout_sec`, `speed_percent`, `status_topic`, `target_status_topic`, `tesseract_plan_topic`, `tracked_target_topic`, `tracking_health_topic`, `trajectory_command_rate_hz`, `trajectory_following_error_grace_sec`, `trajectory_following_error_rad`, `trajectory_joint_step_rad`, `waypoint_progress_epsilon_rad`, `waypoint_progress_timeout_sec`, `waypoint_reached_tolerance_rad`, `waypoint_timeout_sec`, `workflow_status_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/scan_viewpoint_planner_node.py`: `camera_info_topic`, `camera_pitch_deg`, `camera_pitch_offsets_deg`, `debug`, `desired_scan_angle_deg`, `dry_run`, `duplicate_look_tolerance_deg`, `duplicate_position_tolerance_m`, `fallback_target_topic`, `keep_object_centered`, `max_scan_radius_m`, `max_viewpoints`, `min_scan_radius_m`, `minimum_useful_direction_separation_deg`, `nbv_maximum_radius_m`, `nbv_maximum_scoring_voxels`, `nbv_minimum_radius_m`, `nbv_padding_voxels`, `nbv_radius_scale`, `nbv_render_height`, `nbv_render_width`, `nbv_surface_tolerance_m`, `nbv_voxel_size_m`, `object_topic`, `planning_frame_id`, `scan_capture_status_topic`, `scan_coverage_topic`, `scan_radius_m`, `scan_radius_offsets_m`, `scan_session_history_topic`, `scan_viewpoints_topic`, `session_max_views`, `target_plan_refresh_period_sec`, `target_replan_min_period_sec`, `target_replan_translation_m`, `target_status_topic`, `tracked_preference_timeout_s`, `tracked_target_topic`, `use_predicted_target_for_scan`, `view_selection_policy`, `viewpoint_center_angle_deg`, `viewpoint_step_deg`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/supervised_cube_workflow_node.py`: `allow_target_motion_during_scan`, `approach_height_m`, `center_convergence_m`, `cloud_request_topic`, `cloud_status_topic`, `cloud_topic`, `data_timeout_sec`, `drop_obstacle_clearance_m`, `drop_search_radius_m`, `drop_support_max_stddev_m`, `drop_support_min_points`, `drop_support_radius_m`, `drop_target_clearance_m`, `enforce_static_workspace`, `first_push_distance_m`, `heavy_refresh_request_topic`, `heavy_refresh_status_topic`, `landmark_status_topic`, `landmark_topic`, `later_push_distance_m`, `marker_topic`, `max_contact_actions`, `max_grasp_width_m`, `max_tracking_measurement_age_sec`, `max_views`, `min_quality_score`, `min_views`, `movable_whitelist`, `obstacle_displacement_m`, `obstacle_topic`, `occlusion_probe_timeout_sec`, `occlusion_status_topic`, `plan_topic`, `pre_push_offset_m`, `push_distance_m`, `request_optional_cloud_refinement`, `require_observed_drop_support`, `scan_quality_topic`, `scene_cloud_topic`, `status_topic`, `target_clearance_m`, `target_model_topic`, `target_motion_abort_m`, `target_status_topic`, `target_surface_measurement_uncertainty_m`, `tracked_target_topic`, `tracking_health_topic`, `workspace_x_max`, `workspace_x_min`, `workspace_y_max`, `workspace_y_min`, `workspace_z_max`, `workspace_z_min`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_cloud_node.py`: `accumulate_live_masks`, `camera_info_topic`, `cloud_topic`, `color_topic`, `depth_max_m`, `depth_min_m`, `depth_topic`, `frame_cache_size`, `heavy_request_topic`, `heavy_status_topic`, `mask_erode_px`, `mask_max_age_sec`, `mask_topic`, `max_voxels`, `output_dir`, `pixel_stride`, `publish_period_sec`, `refined_capture_retry_sec`, `refined_capture_timeout_sec`, `refined_mask_topic`, `refined_match_tolerance_sec`, `request_topic`, `require_transform`, `scene_accumulate_period_sec`, `scene_cloud_topic`, `scene_max_voxels`, `scene_pixel_stride`, `scene_voxel_size_m`, `status_topic`, `target_frame`, `voxel_size_m`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_error_node.py`: `desired_distance_m`, `distance_tolerance_m`, `error_topic`, `position_tolerance_m`, `tracked_topic`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_handoff_node.py`: `default_confidence`, `input_topic`, `min_confidence`, `output_topic`, `target_type`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_landmark_node.py`: `base_frame`, `camera_info_topic`, `depth_max_m`, `depth_min_m`, `depth_topic`, `heavy_request_topic`, `initial_max_spread_m`, `initial_sample_count`, `landmark_topic`, `landmark_update_alpha`, `mask_erode_px`, `mask_topic`, `measurement_gate_m`, `min_valid_depth_pixels`, `min_valid_depth_ratio`, `new_view_angle_deg`, `projection_disagreement_px`, `projection_topic`, `refresh_cooldown_sec`, `request_refresh_on_mask_disagreement`, `request_refresh_on_new_view`, `status_topic`, `sync_queue_size`, `sync_slop_sec`, `transform_timeout_sec`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_scan_gateway_node.py`: `local_base_frame`, `max_pending_missions`, `mission_spool_root`, `piper_base_frame`, `project_root`, `reconstruction_python`, `reconstruction_timeout_sec`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_scan_mission_node.py`: `contact_speed_percent`, `debug`, `enable_real_arm_motion`, `free_motion_speed_percent`, `home_pose_path`, `manage_processes`, `max_pending_missions`, `maximum_captures`, `mission_queue_coalesce_sec`, `mission_spool_root`, `motion_speed_profile_qualified`, `process_log_root`, `project_root`, `require_gateway_heartbeat`, `require_staged_home_profile`, `required_captures`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/target_tracker_node.py`: `camera_frame`, `confidence_noise_scale`, `debug`, `depth_gate_m`, `innovation_gate_threshold`, `lost_timeout_s`, `low_confidence_timeout_s`, `max_3d_jump_m`, `max_area_ratio`, `max_missed_frames`, `max_pixel_jump`, `max_target_speed_mps`, `measurement_noise`, `min_area_ratio`, `min_confidence`, `min_measurement_confidence`, `min_track_frames`, `piper_base_frame`, `prediction_horizon_s`, `process_noise`, `reject_out_of_order_measurements`, `stable_speed_threshold_mps`, `stable_time_s`, `target_status_topic`, `target_topic`, `tracked_topic`, `transform_timeout_s`, `use_camera_space_gates`, `use_tf_transform`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/tf_target_transform_node.py`: `camera_frame`, `camera_output_topic`, `input_topic`, `piper_base_frame`, `piper_output_topic`, `transform_timeout_s`.
- `piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/viewpoint_reachability_filter_node.py`: `acquisition_viewpoints_topic`, `arm_status_timeout_sec`, `arm_status_topic`, `debug`, `dry_run`, `enforce_static_reach_bounds`, `joint_states_topic`, `max_camera_object_distance_m`, `max_height_change_m`, `max_reach_m`, `min_camera_object_distance_m`, `min_reach_m`, `reachable_acquisition_viewpoints_topic`, `reachable_scan_viewpoints_topic`, `scan_viewpoints_topic`, `target_status_topic`.
- `piper_ros_foxy/src/piper_tesseract_foxy/piper_tesseract_foxy/bridge_node.py`: `camera_timestamp_health_topic`, `closed_loop_candidate_limit`, `closed_loop_max_aim_offset_deg`, `closed_loop_max_view_step_deg`, `closed_loop_min_view_step_deg`, `closed_loop_one_view`, `collision_manifest_path`, `data_timeout_sec`, `debug`, `deterministic_seed`, `hand_eye_calibration_path`, `joint_bounds_path`, `joint_limit_margin_rad`, `joint_states_topic`, `manipulation_model_qualified`, `max_execution_viewpoints`, `max_tracking_measurement_age_sec`, `motion_limits_change_confirmation_sec`, `motion_limits_change_minimum_samples`, `motion_limits_timeout_sec`, `motion_limits_topic`, `obstacle_topic`, `plan_topic`, `reachable_acquisition_viewpoints_topic`, `reachable_viewpoints_topic`, `readiness_topic`, `request_ttl_sec`, `response_timeout_sec`, `return_home_positions_rad`, `robot_xacro_path`, `roll_samples_rad`, `speed_percent`, `spool_root`, `srdf_path`, `status_topic`, `tracking_health_topic`, `trajectory_command_rate_hz`, `trajectory_joint_step_rad`, `worker_heartbeat_timeout_sec`.


## Parameter compatibility

The 2026-08-20 global-NBV repair keeps
`closed_loop_min_view_step_deg` and `closed_loop_max_view_step_deg` declared for
launch/YAML compatibility, but authoritative `voxel_nbv` no longer uses them
as a movement frontier. `minimum_useful_direction_separation_deg` is the
inclusive accepted-view redundancy threshold and is currently 15 degrees.
`closed_loop_candidate_limit` is 12 and
`closed_loop_max_aim_offset_deg` is 5; the latter binds only the one fallback
attempt after exact target aim fails. No ROS message or service schema changed.

Post-baseline additive NBV audit interfaces (2026-08-17) are
`/piper/tesseract_view_generation` and
`/piper/tesseract_plan_provenance`, both `std_msgs/String` JSON. The first is
reliable/transient-local and is emitted only after the Tesseract bridge caches
an exact session/accepted-count generation. The second is reliable/volatile
and binds the selected policy, generation, candidate ID, rank and predicted
gain—including additive `nbv_marginal_information_pixels` and
`nbv_marginal_information_fraction` fields—to `plan_id`; capture frame metadata
persists that binding. Existing ROS
actions, services, messages and topics were not renamed or changed.

The additive parameter names are `view_generation_receipt_topic` and
`plan_provenance_topic` on `bridge_node.py`, and `plan_provenance_topic` on
`scan_capture_node.py`. Their deployed defaults are recorded in the matching
YAML files.

All declared parameter names are external configuration interfaces. The full
source inventory is the 34 node files containing 424 `declare_parameter`
calls plus the configuration files listed in `current_architecture.md`.
Especially sensitive groups are:

- driver CAN/enable/limit query parameters;
- all topic/service/frame and spool paths;
- mission queue, process, speed, capture and home-profile parameters;
- executor approval, motion, settle, tracking, obstacle and holder-envelope
  parameters;
- acquisition/planning radius, angular and candidate parameters;
- perception synchronization, mask/depth, tracking and recovery parameters;
- Tesseract model, collision, timing, hash, TTL and worker-health parameters;
- reconstruction executable and timeout parameters.

Phase 1 may move parameter declaration code only if names, types, defaults,
launch overrides and evaluation timing remain behaviorally identical.
