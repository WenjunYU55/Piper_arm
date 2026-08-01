"""Pure kinematics and conservative validation for supervised scan motion."""

from dataclasses import dataclass
import json
import math

import numpy as np
import yaml


# Conservative planning-model limits. The small negative J2 range admits a
# measured disabled start for collision-checked bootstrap recovery; powered
# command targets use the stricter controller limits below.
URDF_JOINT_LIMITS = np.asarray([
    [-2.6180, 2.1680],
    [-0.044796192, 3.1400],
    [-2.9670, 0.0000],
    [-1.7450, 1.7450],
    [-1.2200, 1.2200],
    [-2.0944, 2.0944],
], dtype=float)

CONTROLLER_COMMAND_LIMITS = URDF_JOINT_LIMITS.copy()
CONTROLLER_COMMAND_LIMITS[1, 0] = 0.0

# Feedback at an encoder boundary can quantize just outside the planning
# model's inclusive limit. This cap is deliberately much smaller than the
# controller and plan-start tolerances and never expands a planned target.
MAX_FEEDBACK_LIMIT_TOLERANCE_RAD = 0.002


@dataclass(frozen=True)
class CollisionBox:
    label: str
    minimum: np.ndarray
    maximum: np.ndarray


class PiperScanKinematics:
    """PiPER SDK modified-DH mode-0 FK extended to the calibrated camera."""

    def __init__(self, link6_from_camera):
        value = np.asarray(link6_from_camera, dtype=float)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError('link6_from_camera must be a finite 4x4 matrix')
        self.link6_from_camera = value
        self.a = np.asarray([0, 0, 285.03, -21.98, 0, 0], dtype=float) / 1000.0
        self.alpha = np.asarray(
            [0, -math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2],
            dtype=float,
        )
        self.theta = np.asarray(
            [0, -math.radians(174.22), -math.radians(100.78), 0, 0, 0],
            dtype=float,
        )
        self.d = np.asarray([123, 0, 0, 250.75, 0, 91], dtype=float) / 1000.0

    @staticmethod
    def link_transform(alpha, a, theta, d):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.asarray([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0, 0, 0, 1],
        ], dtype=float)

    def chain_transforms(self, joints):
        values = finite_joints(joints)
        result = np.eye(4)
        transforms = []
        for index, angle in enumerate(values):
            result = result @ self.link_transform(
                self.alpha[index],
                self.a[index],
                angle + self.theta[index],
                self.d[index],
            )
            transforms.append(result.copy())
        return transforms

    def forward(self, joints):
        return self.chain_transforms(joints)[-1]

    def camera_transform(self, joints):
        return self.forward(joints) @ self.link6_from_camera

    def collision_points(self, joints):
        points = [np.zeros(3, dtype=float)]
        points.extend(transform[:3, 3].copy() for transform in self.chain_transforms(joints))
        points.append(self.camera_transform(joints)[:3, 3].copy())
        return points


def finite_joints(joints):
    values = np.asarray(joints, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError('expected six finite joint positions')
    return values


def energized_hold_target(joints):
    """Return the nearest pose representable while motor power is enabled."""
    values = finite_joints(joints)
    return np.clip(
        values,
        CONTROLLER_COMMAND_LIMITS[:, 0],
        CONTROLLER_COMMAND_LIMITS[:, 1],
    )


def orbit_camera_view(center, angle_deg, radius_m, camera_pitch_deg):
    center_vector = np.asarray(center, dtype=float)
    if center_vector.shape != (3,) or not np.all(np.isfinite(center_vector)):
        raise ValueError('orbit center must contain three finite values')
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError('orbit radius must be positive and finite')
    angle = math.radians(float(angle_deg))
    pitch = math.radians(float(camera_pitch_deg))
    horizontal_radius = radius * math.cos(pitch)
    camera = center_vector + np.asarray([
        horizontal_radius * math.cos(angle),
        horizontal_radius * math.sin(angle),
        -radius * math.sin(pitch),
    ])
    look = center_vector - camera
    look /= np.linalg.norm(look)
    return camera, look


def load_accepted_hand_eye(path):
    with open(path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get('status') != 'accepted':
        raise ValueError('hand-eye calibration is not accepted')
    matrix = np.asarray(data['camera_to_link6']['matrix'], dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('camera_to_link6 matrix must be finite and 4x4')
    return matrix


def load_conservative_joint_limits(path):
    limits = URDF_JOINT_LIMITS.copy()
    ignored = []
    with open(path, 'r', encoding='utf-8') as stream:
        data = json.load(stream)
    records = data.get('joints', {}) if isinstance(data, dict) else {}
    for index in range(6):
        name = 'joint%d' % (index + 1)
        record = records.get(name)
        if not isinstance(record, dict):
            ignored.append(name)
            continue
        if record.get('valid', True) is False:
            ignored.append(name)
            continue
        try:
            low = float(record['min'])
            high = float(record['max'])
        except (KeyError, TypeError, ValueError):
            ignored.append(name)
            continue
        if not math.isfinite(low) or not math.isfinite(high) or low == high:
            ignored.append(name)
            continue
        saved_low, saved_high = min(low, high), max(low, high)
        limits[index, 0] = max(limits[index, 0], saved_low)
        limits[index, 1] = min(limits[index, 1], saved_high)
        if limits[index, 0] >= limits[index, 1]:
            raise ValueError('%s saved bounds do not overlap URDF limits' % name)
    return limits, ignored


def joint_limit_reasons(joints, limits, margin_rad=0.0):
    values = finite_joints(joints)
    bounds = np.asarray(limits, dtype=float)
    reasons = []
    for index, value in enumerate(values):
        low = float(bounds[index, 0]) + float(margin_rad)
        high = float(bounds[index, 1]) - float(margin_rad)
        if value < low or value > high:
            reasons.append(
                'joint%d %.4f outside conservative range [%.4f, %.4f]'
                % (index + 1, value, low, high)
            )
    return reasons


def feedback_joint_limit_reasons(joints, limits, tolerance_rad=0.0):
    """Reject measured feedback beyond a tightly bounded encoder tolerance."""
    values = finite_joints(joints)
    bounds = np.asarray(limits, dtype=float)
    tolerance = float(tolerance_rad)
    if (
            not math.isfinite(tolerance)
            or tolerance < 0.0
            or tolerance > MAX_FEEDBACK_LIMIT_TOLERANCE_RAD):
        raise ValueError(
            'joint feedback limit tolerance must be within [0.0, %.4f] rad'
            % MAX_FEEDBACK_LIMIT_TOLERANCE_RAD)
    reasons = []
    for index, value in enumerate(values):
        low = float(bounds[index, 0])
        high = float(bounds[index, 1])
        if value < low - tolerance or value > high + tolerance:
            reasons.append(
                'current joint%d feedback %.6f is outside configured limits '
                '[%.6f, %.6f] with %.6f rad encoder tolerance'
                % (index + 1, value, low, high, tolerance))
    return reasons


def approval_rejection_reason(
        state,
        current_plan_id,
        requested_plan_id,
        confirmation,
        expected_confirmation,
        real_motion_enabled,
        plan_age_sec,
        plan_max_age_sec,
        current_trajectory_sha256='',
        requested_trajectory_sha256='',
        require_trajectory_hash=False):
    if state != 'PROPOSAL_READY' or not current_plan_id:
        return 'no valid proposal is ready'
    if requested_plan_id != current_plan_id:
        return 'plan_id does not match the current proposal'
    if require_trajectory_hash:
        if not current_trajectory_sha256:
            return 'current proposal has no trajectory hash'
        if requested_trajectory_sha256 != current_trajectory_sha256:
            return 'trajectory_sha256 does not match the current proposal'
    if confirmation != expected_confirmation:
        return 'confirmation must exactly match: %s' % expected_confirmation
    if not real_motion_enabled:
        return 'enable_real_arm_motion is false; node remains proposal-only'
    if float(plan_age_sec) > float(plan_max_age_sec):
        return 'proposal expired; request a fresh plan'
    return ''


def interpolate_joint_path(start, goal, max_step_rad):
    first = finite_joints(start)
    second = finite_joints(goal)
    maximum = max(float(max_step_rad), 1e-5)
    steps = max(1, int(math.ceil(float(np.max(np.abs(second - first))) / maximum)))
    return [first + (second - first) * (index / float(steps)) for index in range(1, steps + 1)]


def segment_distance(first_start, first_end, second_start, second_end):
    p1, q1 = np.asarray(first_start, dtype=float), np.asarray(first_end, dtype=float)
    p2, q2 = np.asarray(second_start, dtype=float), np.asarray(second_end, dtype=float)
    d1, d2, offset = q1 - p1, q2 - p2, p1 - p2
    a, e = float(np.dot(d1, d1)), float(np.dot(d2, d2))
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(p1 - p2))
    if a <= epsilon:
        s = 0.0
        t = float(np.clip(np.dot(d2, offset) / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, offset))
        if e <= epsilon:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = float(np.clip((b * np.dot(d2, offset) - c * e) / denominator, 0.0, 1.0)) \
                if abs(denominator) > epsilon else 0.0
            t = (b * s + float(np.dot(d2, offset))) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    closest_first = p1 + d1 * s
    closest_second = p2 + d2 * t
    return float(np.linalg.norm(closest_first - closest_second))


def segment_intersects_expanded_box(start, end, box, expansion_m):
    minimum = np.asarray(box.minimum, dtype=float) - float(expansion_m)
    maximum = np.asarray(box.maximum, dtype=float) + float(expansion_m)
    origin = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - origin
    low, high = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < minimum[axis] or origin[axis] > maximum[axis]:
                return False
            continue
        first = (minimum[axis] - origin[axis]) / direction[axis]
        second = (maximum[axis] - origin[axis]) / direction[axis]
        entry, leave = min(first, second), max(first, second)
        low, high = max(low, entry), min(high, leave)
        if low > high:
            return False
    return True


def collision_segments(kinematics, joints):
    """Return non-degenerate segments used by the Foxy collision proxy."""
    points = kinematics.collision_points(joints)
    return [
        (index, points[index], points[index + 1])
        for index in range(len(points) - 1)
        if float(np.linalg.norm(points[index + 1] - points[index])) > 1e-7
    ]


def minimum_self_segment_clearance(kinematics, joints):
    """Return the minimum non-adjacent proxy-segment clearance and pair."""
    segments = collision_segments(kinematics, joints)
    minimum = math.inf
    limiting_pair = (-1, -1)
    for first_index, first_start, first_end in segments:
        for second_index, second_start, second_end in segments:
            if second_index <= first_index + 2:
                continue
            distance = segment_distance(
                first_start, first_end, second_start, second_end)
            if distance < minimum:
                minimum = distance
                limiting_pair = (first_index, second_index)
    return float(minimum), limiting_pair


def configuration_collision_reasons(
        kinematics,
        joints,
        obstacle_boxes=(),
        floor_z_m=0.0,
        link_radius_m=0.035,
        self_clearance_m=0.075):
    points = kinematics.collision_points(joints)
    reasons = []
    for index, point in enumerate(points[1:], start=1):
        if float(point[2]) < float(floor_z_m) + float(link_radius_m):
            reasons.append('link point %d violates floor clearance' % index)
            break

    segments = collision_segments(kinematics, joints)
    minimum, pair = minimum_self_segment_clearance(kinematics, joints)
    if minimum < float(self_clearance_m):
        reasons.append(
            'self-collision clearance between link segments %d and %d' % pair)
        return reasons

    expansion = float(link_radius_m)
    for box in obstacle_boxes:
        for segment_index, start, end in segments:
            if segment_intersects_expanded_box(start, end, box, expansion):
                reasons.append(
                    'link segment %d intersects obstacle %s clearance'
                    % (segment_index, box.label)
                )
                return reasons
    return reasons


def validate_joint_path(
        kinematics,
        path,
        joint_limits,
        obstacle_boxes=(),
        joint_margin_rad=0.03,
        floor_z_m=0.0,
        link_radius_m=0.035,
        self_clearance_m=0.075):
    for index, joints in enumerate(path):
        reasons = joint_limit_reasons(joints, joint_limits, joint_margin_rad)
        reasons.extend(configuration_collision_reasons(
            kinematics,
            joints,
            obstacle_boxes=obstacle_boxes,
            floor_z_m=floor_z_m,
            link_radius_m=link_radius_m,
            self_clearance_m=self_clearance_m,
        ))
        if reasons:
            return ['trajectory step %d: %s' % (index, reason) for reason in reasons]
    return []


def validate_monotonic_self_clearance_escape(
        kinematics,
        path,
        joint_limits,
        obstacle_boxes=(),
        joint_margin_rad=0.0,
        floor_z_m=0.0,
        link_radius_m=0.035,
        self_clearance_m=0.075,
        monotonic_tolerance_m=0.002,
        recovery_joint_number=None,
        maximum_start_limit_violation_rad=0.0):
    """
    Validate a hash-bound acquisition prefix that only leaves a folded start.

    Joint, floor, and world checks remain unchanged. Only the Foxy capsule
    proxy's self-clearance threshold is relaxed before the declared endpoint,
    and clearance must not worsen along that prefix. The endpoint must satisfy
    the ordinary threshold before normal trajectory validation resumes.
    """
    if len(path) < 2:
        return ['bootstrap recovery path has fewer than two points']
    tolerance = float(monotonic_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        return ['bootstrap recovery monotonic tolerance is invalid']
    if recovery_joint_number is None:
        limit_reasons = []
        for index, joints in enumerate(path):
            limit_reasons.extend(
                'bootstrap recovery step %d: %s' % (index, reason)
                for reason in joint_limit_reasons(
                    joints, joint_limits, joint_margin_rad))
    else:
        limit_reasons = bootstrap_start_limit_recovery_reasons(
            path,
            joint_limits,
            recovery_joint_number,
            maximum_start_limit_violation_rad,
            joint_margin_rad=joint_margin_rad,
        )
    if limit_reasons:
        return limit_reasons
    previous = None
    endpoint_clearance = math.inf
    endpoint_pair = (-1, -1)
    for index, joints in enumerate(path):
        reasons = []
        # Keep floor and obstacle checks while disabling only the proxy
        # self-distance threshold for this bounded prefix.
        reasons.extend(configuration_collision_reasons(
            kinematics,
            joints,
            obstacle_boxes=obstacle_boxes,
            floor_z_m=floor_z_m,
            link_radius_m=link_radius_m,
            self_clearance_m=-math.inf,
        ))
        if reasons:
            return [
                'bootstrap recovery step %d: %s' % (index, reason)
                for reason in reasons
            ]
        clearance, pair = minimum_self_segment_clearance(kinematics, joints)
        if not math.isfinite(clearance):
            return [
                'bootstrap recovery step %d has no finite self-clearance' % index]
        if previous is not None and clearance + tolerance < previous:
            return [
                'bootstrap recovery step %d worsens proxy self-clearance '
                'from %.6fm to %.6fm' % (index, previous, clearance)]
        previous = clearance
        endpoint_clearance = clearance
        endpoint_pair = pair
    if endpoint_clearance < float(self_clearance_m):
        return [
            'bootstrap recovery endpoint proxy clearance %.6fm between link '
            'segments %d and %d remains below %.6fm'
            % (
                endpoint_clearance,
                endpoint_pair[0],
                endpoint_pair[1],
                float(self_clearance_m),
            )
        ]
    return []


def bootstrap_start_limit_recovery_reasons(
        path,
        joint_limits,
        joint_number,
        maximum_start_violation_rad,
        joint_margin_rad=0.0,
        tolerance_rad=1e-6):
    """Allow declared acquisition joints to move monotonically into range."""
    try:
        supplied = (
            list(joint_number)
            if isinstance(joint_number, (list, tuple, np.ndarray))
            else [joint_number])
        joints = [int(value) - 1 for value in supplied]
        maximum = float(maximum_start_violation_rad)
        margin = float(joint_margin_rad)
        tolerance = float(tolerance_rad)
    except (TypeError, ValueError):
        return ['bootstrap start limit recovery policy is not numeric']
    if (
            not joints
            or len(joints) > 2
            or len(set(joints)) != len(joints)
            or any(joint < 0 or joint >= 6 for joint in joints)):
        return [
            'bootstrap start limit recovery joints must contain one or two '
            'unique values from 1 through 6']
    if (
            not math.isfinite(maximum) or maximum < 0.0
            or maximum > 0.04
            or not math.isfinite(margin)
            or not math.isfinite(tolerance) or tolerance < 0.0):
        return ['bootstrap start limit recovery policy is invalid']
    bounds = np.asarray(joint_limits, dtype=float)
    if bounds.shape != (6, 2):
        return ['bootstrap start limit recovery limits are invalid']

    previous_distances = {joint: math.inf for joint in joints}
    for point_index, position in enumerate(path):
        current = finite_joints(position)
        for index, value in enumerate(current):
            low = float(bounds[index, 0]) + margin
            high = float(bounds[index, 1]) - margin
            if index not in joints and (value < low or value > high):
                return [
                    'bootstrap recovery step %d: joint%d %.4f outside '
                    'conservative range [%.4f, %.4f]'
                    % (point_index, index + 1, value, low, high)]
        for joint in joints:
            low = float(bounds[joint, 0]) + margin
            high = float(bounds[joint, 1]) - margin
            value = float(current[joint])
            distance = max(low - value, value - high, 0.0)
            if point_index == 0 and distance > maximum + tolerance:
                return [
                    'bootstrap recovery start joint%d exceeds its %.4f rad '
                    'limit-recovery bound' % (joint + 1, maximum)]
            if distance > previous_distances[joint] + tolerance:
                return [
                    'bootstrap recovery step %d moves joint%d farther outside '
                    'its configured limits' % (point_index, joint + 1)]
            previous_distances[joint] = distance
    for joint, distance in previous_distances.items():
        if distance > tolerance:
            return [
                'bootstrap recovery endpoint joint%d remains outside '
                'configured limits' % (joint + 1)]
    return []


def bootstrap_recovery_declaration_reasons(
        path,
        end_point,
        joint_number,
        declared_delta_rad,
        maximum_delta_rad=0.15,
        tolerance_rad=1e-5):
    """Verify a bounded recovery prefix moves only its declared joints."""
    try:
        end = int(end_point)
        supplied_joints = (
            list(joint_number)
            if isinstance(joint_number, (list, tuple, np.ndarray))
            else [joint_number])
        supplied_deltas = (
            list(declared_delta_rad)
            if isinstance(declared_delta_rad, (list, tuple, np.ndarray))
            else [declared_delta_rad])
        joints = [int(value) - 1 for value in supplied_joints]
        declared = [float(value) for value in supplied_deltas]
        maximum = float(maximum_delta_rad)
        tolerance = float(tolerance_rad)
    except (TypeError, ValueError):
        return ['bootstrap recovery declaration is not numeric']
    if end < 1 or end >= len(path):
        return ['bootstrap recovery endpoint is outside the trajectory']
    if (
            not joints
            or len(joints) > 2
            or len(joints) != len(declared)
            or len(set(joints)) != len(joints)
            or any(joint < 0 or joint >= 6 for joint in joints)):
        return [
            'bootstrap recovery joints must contain one or two unique values '
            'from 1 through 6']
    if (
            any(not math.isfinite(value) for value in declared)
            or any(abs(value) <= tolerance for value in declared)
            or any(abs(value) > maximum + tolerance for value in declared)):
        return ['bootstrap recovery delta is outside its bounded range']
    start = finite_joints(path[0])
    previous_progress = {joint: 0.0 for joint in joints}
    other_joints = [index for index in range(6) if index not in joints]
    for index, position in enumerate(path[:end + 1]):
        current = finite_joints(position)
        delta = current - start
        if (
                other_joints
                and float(np.max(np.abs(delta[other_joints]))) > tolerance):
            return [
                'bootstrap recovery step %d moves a non-declared joint' % index]
        for joint, expected in zip(joints, declared):
            progress = abs(float(delta[joint]))
            if progress > maximum + tolerance:
                return [
                    'bootstrap recovery step %d exceeds its maximum delta'
                    % index]
            if float(delta[joint]) * expected < -tolerance:
                return [
                    'bootstrap recovery step %d moves opposite its declared '
                    'direction' % index]
            if progress + tolerance < previous_progress[joint]:
                return [
                    'bootstrap recovery step %d reverses before its endpoint'
                    % index]
            previous_progress[joint] = progress
    endpoint = finite_joints(path[end])
    for joint, expected in zip(joints, declared):
        actual = float(endpoint[joint] - start[joint])
        if abs(actual - expected) > tolerance:
            return [
                'bootstrap recovery endpoint does not match declared delta']
    return []
