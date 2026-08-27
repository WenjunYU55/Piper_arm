std_msgs/Header header
string KIND_MULTIVIEW_SCAN="MULTIVIEW_SCAN"
string KIND_ROUGH_ACQUISITION="ROUGH_ACQUISITION"
string KIND_RETURN_HOME="RETURN_HOME"
string KIND_OCCLUSION_PROBE="OCCLUSION_PROBE"
string KIND_OCCLUDER_PICK_PLACE="OCCLUDER_PICK_PLACE"
string KIND_OCCLUDER_PUSH="OCCLUDER_PUSH"
string plan_id
string plan_kind
string source_request_id
string request_sha256
string trajectory_sha256
string motion_limits_sha256
float32 execution_speed_percent
float32 command_rate_hz
string timing_policy
string backend
string backend_version
bool valid
bool dry_run
bool real_arm_motion
bool collision_model_qualified
string reason
string[] rejection_codes
geometry_msgs/Point target_center
uint32[] viewpoint_indices
geometry_msgs/Point[] camera_positions
geometry_msgs/Vector3[] look_directions
trajectory_msgs/JointTrajectory[] trajectories
float64[] minimum_clearance_m
string[] limiting_link_pairs
# Optional backend-neutral acquisition/start-state recovery evidence. -1 means
# no recovery prefix. A non-negative index is valid only for segment zero of a
# ROUGH_ACQUISITION plan and indexes trajectories[i].points.
int32[] bootstrap_recovery_end_points
int32[] bootstrap_recovery_joints
float64[] bootstrap_recovery_delta_rad
string[] bootstrap_recovery_evidence_json
# Comparable planner diagnostics. Unknown numeric values are encoded as -1.
float64 planning_duration_sec
float64 trajectory_duration_sec
float64 joint_space_path_length_rad
uint32 candidate_viewpoints_considered
uint32 candidate_viewpoints_rejected
uint32 feasible_viewpoints
uint32 successful_viewpoints
