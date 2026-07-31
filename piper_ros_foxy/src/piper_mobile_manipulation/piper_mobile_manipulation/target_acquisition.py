"""Pure geometry and gating helpers for rough-coordinate target acquisition."""

import math

import numpy as np

from piper_mobile_manipulation.scan_motion import orbit_camera_view


ROUGH_ACQUISITION = 'ROUGH_ACQUISITION'
MULTIVIEW_SCAN = 'MULTIVIEW_SCAN'


def viewpoint_payload_matches(
        payload, expected_plan_kind, rough_hint_stamp_ns=None):
    if not isinstance(payload, dict) or payload.get('dry_run') is not True:
        return False
    if payload.get('plan_kind', MULTIVIEW_SCAN) != expected_plan_kind:
        return False
    if rough_hint_stamp_ns is not None:
        try:
            return int(payload.get('rough_hint_stamp_ns')) == int(
                rough_hint_stamp_ns)
        except (TypeError, ValueError):
            return False
    return True


def rough_hint_rejection_reason(
        frame_id, point, stamp_ns, now_ns, max_age_sec, future_tolerance_sec=0.1):
    if str(frame_id) != 'base_link':
        return 'rough target frame must be base_link'
    values = np.asarray(point, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return 'rough target point must contain three finite coordinates'
    stamp_ns = int(stamp_ns)
    now_ns = int(now_ns)
    if stamp_ns <= 0:
        return 'rough target stamp is missing'
    age_sec = float(now_ns - stamp_ns) * 1e-9
    if age_sec < -abs(float(future_tolerance_sec)):
        return 'rough target stamp is in the future'
    if age_sec > float(max_age_sec):
        return 'rough target hint is stale %.3fs > %.3fs' % (
            age_sec, float(max_age_sec))
    return ''


def build_acquisition_viewpoints(
        rough_target, current_camera_position, standoff_m=0.45,
        camera_pitch_deg=-10.0, sweep_angle_deg=45.0,
        fallback_standoff_m=None, current_camera_look_direction=None):
    """
    Build a bounded look-direction cone without overshooting the camera.

    ``standoff_m`` is the maximum acquisition radius.  When the camera is
    already closer to the rough target, the center look stays at the current
    camera position.  The five primary looks keep that camera position and
    sweep its optical direction through the configured yaw/pitch cone.  This
    searches around an imperfect coordinate instead of repeatedly centering
    that coordinate from different orbit positions, and keeps a hint near the
    arm from pushing the camera through or behind the robot base.

    ``camera_pitch_deg`` remains in the signature for configuration/API
    compatibility.  The center pitch is derived from the live camera-to-target
    vector so the first look is the minimum-translation look-at pose.
    """
    target = np.asarray(rough_target, dtype=float)
    current_camera = np.asarray(current_camera_position, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError('rough target must contain three finite coordinates')
    if current_camera.shape != (3,) or not np.all(np.isfinite(current_camera)):
        raise ValueError('current camera position must contain three finite coordinates')
    current_look = None
    if current_camera_look_direction is not None:
        current_look = np.asarray(
            current_camera_look_direction, dtype=float)
        if (
                current_look.shape != (3,)
                or not np.all(np.isfinite(current_look))
                or float(np.linalg.norm(current_look)) < 1e-9):
            raise ValueError(
                'current camera look direction must be finite and non-zero')
        current_look /= np.linalg.norm(current_look)
    maximum_standoff = float(standoff_m)
    if not math.isfinite(maximum_standoff) or maximum_standoff <= 0.0:
        raise ValueError('acquisition standoff must be positive and finite')
    if not math.isfinite(float(camera_pitch_deg)):
        raise ValueError('acquisition camera pitch must be finite')
    sweep = abs(float(sweep_angle_deg))
    if not math.isfinite(sweep):
        raise ValueError('acquisition sweep angle must be finite')

    offset = current_camera - target
    current_distance = float(np.linalg.norm(offset))
    if current_distance < 1e-6:
        raise ValueError('current camera position coincides with rough target')
    horizontal = offset[:2]
    if float(np.linalg.norm(horizontal)) < 1e-6:
        raise ValueError('current camera azimuth is undefined at the target axis')
    azimuth_deg = math.degrees(math.atan2(horizontal[1], horizontal[0]))
    pitch_ratio = float(np.clip(offset[2] / current_distance, -1.0, 1.0))
    center_pitch_deg = -math.degrees(math.asin(pitch_ratio))
    effective_standoff = min(current_distance, maximum_standoff)
    diagonal_pitch = min(sweep, 30.0)
    cone_offsets = [
        ('center', 0.0, 0.0),
        ('left', sweep, 0.0),
        ('right', -sweep, 0.0),
        ('up', 0.0, sweep),
        ('down', 0.0, -sweep),
        ('left_up', sweep, diagonal_pitch),
        ('right_up', -sweep, diagonal_pitch),
        ('left_down', sweep, -diagonal_pitch),
        ('right_down', -sweep, -diagonal_pitch),
    ]
    fallback_radius = None
    fallback_offsets = []
    if fallback_standoff_m is not None:
        fallback_radius = float(fallback_standoff_m)
        if (
                not math.isfinite(fallback_radius)
                or fallback_radius <= 0.0
                or fallback_radius > maximum_standoff):
            raise ValueError(
                'acquisition fallback standoff must be positive, finite, '
                'and no greater than the maximum standoff')
        # A centreline target can make all five minimum-translation look-at
        # orientations unreachable. Offer Tesseract an interleaved compact
        # yaw/pitch search as additional candidates. Tesseract still owns
        # bounded IK, collision checking and final selection.
        fallback_radius = min(effective_standoff, fallback_radius)
        fallback_offsets = [
            ('compact_left', sweep, 0.0),
            ('compact_right_wide', -sweep - 5.0, 0.0),
            ('compact_left_high', sweep, -5.0),
            ('compact_right_low', -sweep, 15.0),
            ('compact_left_low', sweep, 5.0),
            ('compact_right_wide_high', -sweep - 5.0, -5.0),
            ('compact_right', -sweep, 0.0),
            ('compact_left_wide', sweep + 5.0, 0.0),
            ('compact_left_wide_high', sweep + 5.0, -5.0),
            ('compact_left_wide_low', sweep + 5.0, 5.0),
            ('compact_right_wide_low', -sweep - 5.0, 5.0),
            ('compact_right_high', -sweep, -5.0),
            ('compact_right_mid_low', -sweep, 5.0),
            ('compact_left_deep_low', sweep, 15.0),
        ]
        if current_look is None:
            fallback_offsets.append(('compact_center', 0.0, 0.0))
    center = dict(zip(('x', 'y', 'z'), (float(value) for value in target)))
    viewpoints = []
    pose_keys = set()

    def rotated(vector, axis, angle_deg):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        angle = math.radians(float(angle_deg))
        return (
            vector * math.cos(angle)
            + np.cross(axis, vector) * math.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
        )

    center_camera, center_look = orbit_camera_view(
        target, azimuth_deg, effective_standoff, center_pitch_deg)
    world_up = np.asarray([0.0, 0.0, 1.0])

    def append_current_view():
        if current_look is None:
            return
        pose_key = tuple(np.round(np.concatenate((
            current_camera, current_look)), 9))
        pose_keys.add(pose_key)
        viewpoints.append({
            'index': len(viewpoints),
            'acquisition_look': 'current_view',
            'acquisition_yaw_offset_deg': 0.0,
            'acquisition_pitch_offset_deg': 0.0,
            'acquisition_search_stage': 'current_camera',
            'frame_id': 'base_link',
            'target_object_center': center,
            'desired_camera_position': dict(
                zip(
                    ('x', 'y', 'z'),
                    (float(value) for value in current_camera))),
            'desired_look_at_direction': dict(
                zip(
                    ('x', 'y', 'z'),
                    (float(value) for value in current_look))),
            'camera_object_distance_m': current_distance,
            'maximum_standoff_m': maximum_standoff,
            'keep_object_centered': False,
            'reachable': False,
            'safe': False,
        })

    def append_cone_view(name, yaw_deg, pitch_deg):
        if len(viewpoints) >= 20:
            return
        look = rotated(center_look, world_up, yaw_deg)
        right = np.cross(look, world_up)
        if float(np.linalg.norm(right)) < 1e-9:
            raise ValueError('acquisition look direction has no pitch axis')
        look = rotated(look, right, pitch_deg)
        look /= np.linalg.norm(look)
        pose_key = tuple(np.round(np.concatenate((center_camera, look)), 9))
        if pose_key in pose_keys:
            return
        pose_keys.add(pose_key)
        viewpoints.append({
            'index': len(viewpoints),
            'acquisition_look': name,
            'acquisition_yaw_offset_deg': float(yaw_deg),
            'acquisition_pitch_offset_deg': float(pitch_deg),
            'acquisition_search_stage': 'orientation_cone',
            'frame_id': 'base_link',
            'target_object_center': center,
            'desired_camera_position': dict(
                zip(
                    ('x', 'y', 'z'),
                    (float(value) for value in center_camera))),
            'desired_look_at_direction': dict(
                zip(('x', 'y', 'z'), (float(value) for value in look))),
            'camera_object_distance_m': float(effective_standoff),
            'maximum_standoff_m': maximum_standoff,
            'keep_object_centered': name == 'center',
            'reachable': False,
            'safe': False,
        })

    def append_view(name, yaw_deg, pitch_deg, radius, search_stage):
        if len(viewpoints) >= 20:
            return
        camera, look = orbit_camera_view(
            target,
            azimuth_deg + yaw_deg,
            radius,
            center_pitch_deg + pitch_deg,
        )
        pose_key = tuple(np.round(np.concatenate((camera, look)), 9))
        if pose_key in pose_keys:
            return
        pose_keys.add(pose_key)
        camera_position = dict(
            zip(('x', 'y', 'z'), (float(value) for value in camera)))
        viewpoints.append({
            'index': len(viewpoints),
            'acquisition_look': name,
            'acquisition_yaw_offset_deg': float(yaw_deg),
            'acquisition_pitch_offset_deg': float(pitch_deg),
            'acquisition_search_stage': search_stage,
            'frame_id': 'base_link',
            'target_object_center': center,
            'desired_camera_position': camera_position,
            'desired_look_at_direction': dict(
                zip(('x', 'y', 'z'), (float(value) for value in look))),
            'camera_object_distance_m': float(radius),
            'maximum_standoff_m': maximum_standoff,
            'keep_object_centered': True,
            'reachable': False,
            'safe': False,
        })

    append_current_view()
    for name, yaw_deg, pitch_deg in cone_offsets:
        append_cone_view(name, yaw_deg, pitch_deg)
    for name, yaw_deg, pitch_deg in fallback_offsets:
        append_view(
            name, yaw_deg, pitch_deg, fallback_radius, 'compact_fallback')
    return viewpoints
