std_msgs/Header header
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
# Per-segment boundary for a Tesseract-validated folded-start escape prefix.
# -1 means no prefix. A non-negative value is permitted only on segment zero
# of a ROUGH_ACQUISITION plan and indexes trajectories[i].points.
int32[] bootstrap_recovery_end_points
int32[] bootstrap_recovery_joints
float64[] bootstrap_recovery_delta_rad
string[] bootstrap_recovery_evidence_json
