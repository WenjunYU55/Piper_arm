import math

from geometry_msgs.msg import Point, PoseStamped


def point_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def make_pose_stamped(frame_id, stamp, x, y, z):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation.w = 1.0
    return pose


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


def point_from_xyz(x, y, z):
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p
