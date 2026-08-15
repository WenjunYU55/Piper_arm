import math

from geometry_msgs.msg import PoseStamped


def offset_pose_away_from_target(target_pose, offset_m):
    pose = PoseStamped()
    pose.header = target_pose.header
    pose.pose.orientation = target_pose.pose.orientation
    x = target_pose.pose.position.x
    y = target_pose.pose.position.y
    z = target_pose.pose.position.z
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-6:
        pose.pose.position = target_pose.pose.position
        return pose
    pose.pose.position.x = x - offset_m * x / norm
    pose.pose.position.y = y - offset_m * y / norm
    pose.pose.position.z = z - offset_m * z / norm
    return pose
