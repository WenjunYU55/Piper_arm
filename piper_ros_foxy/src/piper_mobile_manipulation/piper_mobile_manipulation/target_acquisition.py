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
        fallback_standoff_m=None, current_camera_look_direction=None,
        look_index=0):
    """
    Build a bounded look-direction cone without overshooting the camera.

    ``standoff_m`` is the maximum acquisition radius.  When the camera is
    already closer to the rough target, the center look stays at the current
    camera position.  The first view always rotates the optical axis toward
    the supplied landmark; later views sweep through the configured yaw/pitch
    cone.  This searches around an imperfect coordinate without first running
    perception from an unrelated camera direction, and keeps a hint near the
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
    look_index = int(look_index)
    if look_index < 0 or look_index >= 5:
        raise ValueError('acquisition look index must be from 0 through 4')

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
    image_up = world_up - center_look * np.dot(world_up, center_look)
    if float(np.linalg.norm(image_up)) < 1e-9:
        image_up = np.asarray([0.0, 1.0, 0.0])
    image_up /= np.linalg.norm(image_up)

    def append_cone_view(name, yaw_deg, pitch_deg):
        if len(viewpoints) >= 20:
            return
        look = rotated(center_look, image_up, yaw_deg)
        right = np.cross(look, image_up)
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

    # The centered landmark view is a mandatory semantic first step. Offer a
    # compact centered alternative immediately after it so Tesseract can solve
    # a different camera radius without starting from a cone-offset view.
    name, yaw_deg, pitch_deg = cone_offsets[0]
    append_cone_view(name, yaw_deg, pitch_deg)
    if fallback_radius is not None:
        append_view(
            'compact_center', 0.0, 0.0, fallback_radius,
            'compact_fallback')
    for name, yaw_deg, pitch_deg in cone_offsets[1:]:
        append_cone_view(name, yaw_deg, pitch_deg)
    for name, yaw_deg, pitch_deg in fallback_offsets:
        append_view(
            name, yaw_deg, pitch_deg, fallback_radius, 'compact_fallback')
    look_groups = (
        ('center', 'compact_center', 'compact_left', 'compact_right'),
        ('left', 'left_up', 'left_down', 'compact_left'),
        ('right', 'right_up', 'right_down', 'compact_right'),
        ('up', 'left_up', 'right_up', 'compact_left_high', 'compact_right_high'),
        ('down', 'left_down', 'right_down', 'compact_left_low', 'compact_right_low'),
    )
    prefixes = look_groups[look_index]
    selected = [
        item for item in viewpoints
        if any(
            str(item.get('acquisition_look', '')).startswith(prefix)
            for prefix in prefixes)
    ]
    if not selected:
        raise ValueError('acquisition transaction has no candidate look')
    primary = selected[0]
    primary_camera = np.asarray([
        primary['desired_camera_position'][axis] for axis in ('x', 'y', 'z')
    ], dtype=float)
    primary_look = np.asarray([
        primary['desired_look_at_direction'][axis] for axis in ('x', 'y', 'z')
    ], dtype=float)
    local_candidates = (
        selected[:5]
        if look_index == 0 and fallback_standoff_m is not None
        else [primary])
    primary_image_up = world_up - primary_look * np.dot(world_up, primary_look)
    if float(np.linalg.norm(primary_image_up)) < 1e-9:
        primary_image_up = np.asarray([0.0, 1.0, 0.0])
    primary_image_up /= np.linalg.norm(primary_image_up)
    local_angle_deg = min(sweep, 15.0) if look_index == 0 else min(sweep, 5.0)
    local_axes = (
        ('local_left', primary_image_up, local_angle_deg),
        ('local_right', primary_image_up, -local_angle_deg),
    )
    pitch_axis = np.cross(primary_look, world_up)
    if float(np.linalg.norm(pitch_axis)) > 1e-9:
        local_axes += (
            ('local_up', pitch_axis, local_angle_deg),
            ('local_down', pitch_axis, -local_angle_deg),
        )
    for suffix, axis, angle_deg in local_axes:
        if len(local_candidates) >= 5:
            break
        camera = primary_camera.copy()
        look = rotated(primary_look, axis, angle_deg)
        look /= np.linalg.norm(look)
        local_candidates.append({
            **primary,
            'acquisition_look': '%s_%s' % (
                primary['acquisition_look'], suffix),
            'acquisition_search_stage': 'flexible_local_region',
            'desired_camera_position': dict(zip(
                ('x', 'y', 'z'), (float(value) for value in camera))),
            'desired_look_at_direction': dict(zip(
                ('x', 'y', 'z'), (float(value) for value in look))),
        })
    selected = local_candidates
    # Every transaction offers only one compact angular neighborhood at the
    # measured camera position. The
    # first candidate is the mandatory semantic look for this transaction;
    # remaining candidates are bounded IK alternatives, not extra looks.
    for index, item in enumerate(selected[:5]):
        item['index'] = index
        item['acquisition_transaction_index'] = look_index
        # In the closed-loop contract this marker means "eligible as the one
        # mandatory look in this transaction", not an exact center-pixel pose.
        item['keep_object_centered'] = True
        radial = np.asarray([
            item['desired_camera_position'][axis] for axis in ('x', 'y', 'z')
        ], dtype=float) - target
        if (
                float(np.dot(radial, offset)) <= 0.0
                or float(np.linalg.norm(radial)) > current_distance + 1e-6):
            raise ValueError(
                'acquisition candidate crosses behind or beyond the close target')
    return selected[:5]
