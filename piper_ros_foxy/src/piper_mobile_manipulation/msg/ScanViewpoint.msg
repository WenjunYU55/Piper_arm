uint8 STATUS_UNCHECKED=0
uint8 STATUS_REACHABLE=1
uint8 STATUS_REJECTED=2

uint32 index
std_msgs/Header header
geometry_msgs/Pose camera_pose
geometry_msgs/Point target_center
float32 view_angle_deg
float32 camera_distance_m
float32 coverage_score
float32 safety_score
float64[6] joint_solution
uint8 status
bool reachable
bool safe
string[] rejection_reasons
