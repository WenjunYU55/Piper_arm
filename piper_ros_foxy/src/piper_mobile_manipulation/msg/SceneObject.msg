uint8 CLASS_GROUND=0
uint8 CLASS_MOVABLE=1
uint8 CLASS_PROTECTED=2
uint8 CLASS_UNKNOWN=3

std_msgs/Header header
uint32 object_id
string semantic_label
uint8 classification
geometry_msgs/PoseWithCovariance pose
geometry_msgs/Vector3 size
float32 confidence
uint32 observation_count
bool valid
string reason
