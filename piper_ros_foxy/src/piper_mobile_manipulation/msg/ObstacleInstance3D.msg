uint8 CLASSIFICATION_BLOCKED=0
uint8 CLASSIFICATION_MOVABLE=1
uint8 CLASSIFICATION_UNSAFE=2

std_msgs/Header header
uint32 object_id
string semantic_label
float32 confidence
uint8 classification
string camera_frame
string base_frame
geometry_msgs/Point camera_centroid
geometry_msgs/Point camera_bounds_min
geometry_msgs/Point camera_bounds_max
geometry_msgs/Point base_centroid
geometry_msgs/Point base_bounds_min
geometry_msgs/Point base_bounds_max
float32 valid_depth_ratio
uint32 valid_depth_pixels
float32 transform_age_sec
bool valid
string validity_reason
# Session-scoped identity and target-centred visibility evidence. Zero/empty
# values mean that the legacy single-frame geometry producer has not yet
# supplied the optional evidence; they never authorize contact by themselves.
string track_id
uint32 observation_count
geometry_msgs/Polygon tabletop_footprint
float32 position_uncertainty_m
float32 target_overlap_ratio
float32 closer_depth_occlusion_ratio
bool confirmed_in_probe_view
