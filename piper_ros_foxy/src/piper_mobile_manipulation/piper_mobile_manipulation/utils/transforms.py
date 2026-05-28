def pose_frame(pose_stamped):
    return pose_stamped.header.frame_id


def has_frame(pose_stamped):
    return bool(pose_stamped.header.frame_id)
