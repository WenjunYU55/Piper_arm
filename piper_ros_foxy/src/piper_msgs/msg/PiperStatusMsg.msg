uint8 ctrl_mode
uint8 arm_status
uint8 mode_feedback
uint8 teach_status
uint8 motion_status
uint8 trajectory_num
int64 err_code
bool joint_1_angle_limit
bool joint_2_angle_limit
bool joint_3_angle_limit
bool joint_4_angle_limit
bool joint_5_angle_limit
bool joint_6_angle_limit
bool communication_status_joint_1
bool communication_status_joint_2
bool communication_status_joint_3
bool communication_status_joint_4
bool communication_status_joint_5
bool communication_status_joint_6

# Authoritative low-speed FOC feedback.  The aggregate arm status can remain
# zero when one motor has independently disabled, so motion consumers must use
# these fields as well as err_code.
bool motor_feedback_valid
bool motor_1_driver_enabled
bool motor_2_driver_enabled
bool motor_3_driver_enabled
bool motor_4_driver_enabled
bool motor_5_driver_enabled
bool motor_6_driver_enabled
string[] motor_faults
string motor_watchdog_reason
