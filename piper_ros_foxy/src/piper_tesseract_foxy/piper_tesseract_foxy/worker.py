"""
ROS-free Tesseract 0.35 plan worker.

The process has no command or motor interface. It accepts only the validated
filesystem contract and returns dry-run trajectory proposals.
"""

import argparse
import gc
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from piper_tesseract_foxy.contract import (
    attach_digest,
    ContractError,
    JOINT_NAMES,
    SCHEMA_VERSION,
    Spool,
    trajectory_digest,
    validate_request,
)

WORKER_PLANNING_BUDGET_SEC = 150.0
WORKER_RESPONSE_RESERVE_SEC = 5.0


class BackendUnavailable(RuntimeError):
    pass


def look_at_quaternion(look_direction, roll):
    z_axis = np.asarray(look_direction, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up, z_axis))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    cosine, sine = math.cos(float(roll)), math.sin(float(roll))
    rolled_x = cosine * x_axis + sine * y_axis
    rolled_y = -sine * x_axis + cosine * y_axis
    matrix = np.column_stack([rolled_x, rolled_y, z_axis])
    return Rotation.from_matrix(matrix).as_quat().tolist()


def finite_six(value, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ContractError('%s must contain six finite values' % label)
    return array


def _quintic_segment(q0, v0, a0, q1, v1, a1, duration, ratio):
    """Evaluate the C2 quintic joining two timed joint states."""
    duration = float(duration)
    ratio = float(ratio)
    c0 = q0
    c1 = v0 * duration
    c2 = 0.5 * a0 * duration * duration
    position_residual = q1 - (c0 + c1 + c2)
    velocity_residual = v1 * duration - (c1 + 2.0 * c2)
    acceleration_residual = a1 * duration * duration - 2.0 * c2
    c3 = (
        10.0 * position_residual
        - 4.0 * velocity_residual
        + 0.5 * acceleration_residual)
    c4 = (
        -15.0 * position_residual
        + 7.0 * velocity_residual
        - acceleration_residual)
    c5 = (
        6.0 * position_residual
        - 3.0 * velocity_residual
        + 0.5 * acceleration_residual)
    position = (
        c0 + c1 * ratio + c2 * ratio ** 2 + c3 * ratio ** 3
        + c4 * ratio ** 4 + c5 * ratio ** 5)
    velocity = (
        c1 + 2.0 * c2 * ratio + 3.0 * c3 * ratio ** 2
        + 4.0 * c4 * ratio ** 3 + 5.0 * c5 * ratio ** 4) / duration
    acceleration = (
        2.0 * c2 + 6.0 * c3 * ratio + 12.0 * c4 * ratio ** 2
        + 20.0 * c5 * ratio ** 3) / (duration * duration)
    return position, velocity, acceleration


def quiesce_bootstrap_recovery_prefix(
        points, endpoint_positions, joint_numbers, tolerance_rad=1e-8):
    """
    Stop every recovery knot so spline derivatives cannot move other joints.

    ISP sees the recovery and normal OMPL path as one program.  Its derivatives
    can consequently anticipate post-recovery motion on otherwise stationary
    joints.  A quintic reconstructed from those derivatives then moves an
    undeclared joint before the recovery endpoint.  The recovery is deliberately
    short and bounded, so making each prefix knot a rest point preserves
    the geometric path while keeping the declaration exact.
    """
    endpoint = finite_six(endpoint_positions, 'bootstrap recovery endpoint')
    try:
        supplied = (
            list(joint_numbers)
            if isinstance(joint_numbers, (list, tuple, np.ndarray))
            else [joint_numbers])
        joints = [int(value) - 1 for value in supplied]
        tolerance = float(tolerance_rad)
    except (TypeError, ValueError):
        raise ContractError('bootstrap recovery declaration is invalid')
    if (
            not joints
            or len(joints) > 2
            or len(set(joints)) != len(joints)
            or any(joint < 0 or joint >= 6 for joint in joints)):
        raise ContractError(
            'bootstrap recovery joints must contain one or two unique values '
            'from 1 through 6')
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ContractError('bootstrap recovery tolerance is invalid')
    if len(points) < 2:
        raise ContractError('bootstrap recovery path has fewer than two points')

    boundary = next((
        index for index, point in enumerate(points)
        if float(np.max(np.abs(
            finite_six(point['positions_rad'], 'trajectory position')
            - endpoint))) <= tolerance
    ), None)
    if boundary is None or boundary < 1:
        raise ContractError(
            'time-parameterized path lost bootstrap recovery endpoint')

    start = finite_six(points[0]['positions_rad'], 'trajectory position')
    previous_progress = {joint: 0.0 for joint in joints}
    other_joints = [index for index in range(6) if index not in joints]
    for index, point in enumerate(points[:boundary + 1]):
        position = finite_six(point['positions_rad'], 'trajectory position')
        delta = position - start
        if (
                other_joints
                and float(np.max(np.abs(delta[other_joints]))) > tolerance):
            raise ContractError(
                'bootstrap recovery step %d moves a non-declared joint' % index)
        for joint in joints:
            progress = abs(float(delta[joint]))
            if progress + tolerance < previous_progress[joint]:
                raise ContractError(
                    'bootstrap recovery step %d reverses before its endpoint'
                    % index)
            previous_progress[joint] = progress
        point['velocities_rad_s'] = [0.0] * 6
        point['accelerations_rad_s2'] = [0.0] * 6
    return boundary


def sdk_movej_waypoint_trajectory(
        points,
        speed_percent,
        command_rate_hz,
        maximum_step_rad,
        position_limits,
        velocity_limits,
        acceleration_limits,
        mandatory_waypoints=(),
        bootstrap_start_limit_tolerance_rad=0.0,
):
    """
    Create the exact position targets sent as feedback-gated SDK MoveJ goals.

    OMPL/ISP establishes a feasible goal, but the PiPER SDK exposes only a
    position target plus one aggregate speed percentage. The real controller
    interpolates directly to that target, so bind only the measured start and
    final goal. A folded-start acquisition may bind one separately proven
    bootstrap target. The caller collision-validates these exact straight
    joint-space segments. Derivatives are zero because the SDK cannot consume
    per-joint qdot/qddot commands.
    """
    if len(points) < 2:
        raise ContractError('trajectory has fewer than two points')
    speed = float(speed_percent)
    rate = float(command_rate_hz)
    maximum_step = float(maximum_step_rad)
    if not math.isfinite(speed) or speed < 1.0 or speed > 100.0:
        raise ContractError('execution speed must be within 1..100 percent')
    if not math.isfinite(rate) or rate <= 0.0:
        raise ContractError('command rate must be finite and positive')
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ContractError('maximum joint step must be finite and positive')
    positions = np.asarray([
        finite_six(point['positions_rad'], 'trajectory position')
        for point in points])
    for point in points:
        finite_six(point['velocities_rad_s'], 'trajectory velocity')
        finite_six(point['accelerations_rad_s2'], 'trajectory acceleration')
    times = np.asarray([
        float(point['time_from_start_s']) for point in points], dtype=float)
    if (
            not np.all(np.isfinite(times))
            or times[0] < 0.0
            or np.any(np.diff(times) <= 0.0)):
        raise ContractError('trajectory times are not strictly increasing')
    position_bounds = np.asarray(position_limits, dtype=float)
    velocity_bounds = finite_six(
        velocity_limits, 'controller velocity limits')
    acceleration_bounds = finite_six(
        acceleration_limits, 'controller acceleration limits')
    if position_bounds.shape != (6, 2):
        raise ContractError('position limits must have shape 6 x 2')
    if (
            np.any(velocity_bounds <= 0.0)
            or np.any(acceleration_bounds <= 0.0)):
        raise ContractError('controller motion limits must be positive')

    # The limits are still hash-bound and must be valid even though the SDK
    # itself owns velocity/acceleration shaping for MoveJ.
    if np.any(velocity_bounds <= 0.0) or np.any(acceleration_bounds <= 0.0):
        raise ContractError('controller motion limits must be positive')

    output_positions = [positions[0].copy()]
    for waypoint in mandatory_waypoints:
        value = finite_six(waypoint, 'mandatory SDK MoveJ waypoint')
        if float(np.max(np.abs(value - output_positions[-1]))) > 1e-9:
            output_positions.append(value.copy())
    endpoint = positions[-1]
    if float(np.max(np.abs(endpoint - output_positions[-1]))) > 1e-9:
        output_positions.append(endpoint.copy())
    if len(output_positions) < 2:
        raise ContractError('SDK MoveJ target path has no joint motion')
    if len(output_positions) > 3:
        raise ContractError(
            'SDK MoveJ target path has more than one bootstrap target')

    start_tolerance = float(bootstrap_start_limit_tolerance_rad)
    if (
            not math.isfinite(start_tolerance)
            or start_tolerance < 0.0
            or start_tolerance > 0.04):
        raise ContractError(
            'SDK MoveJ bootstrap start limit tolerance is invalid')
    for index, position in enumerate(output_positions):
        tolerance = start_tolerance if index == 0 else 0.0
        position_excess = max(
            float(np.max(position_bounds[:, 0] - position)),
            float(np.max(position - position_bounds[:, 1])),
        )
        if position_excess > tolerance + 1e-9:
            raise ContractError(
                'SDK MoveJ target path exceeds a position limit')

    period = 1.0 / rate
    emitted = []
    for index, position in enumerate(output_positions):
        emitted.append({
            # This is an ordinal transport stamp, not a controller schedule.
            'time_from_start_s': round(index * period * 1e9) / 1e9,
            'positions_rad': position.tolist(),
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        })
    return emitted, [dict(point) for point in emitted]


def subdivide_joint_segment(first, second, maximum_l1_step):
    """Return endpoint-inclusive interpolation with bounded total joint change."""
    q0 = finite_six(first, 'validation segment start')
    q1 = finite_six(second, 'validation segment end')
    maximum_l1_step = float(maximum_l1_step)
    if not math.isfinite(maximum_l1_step) or maximum_l1_step <= 0.0:
        raise ContractError('validation maximum L1 joint step is invalid')
    steps = max(
        1,
        int(math.ceil(float(np.sum(np.abs(q1 - q0))) / maximum_l1_step)),
    )
    return [
        q0 + (q1 - q0) * (index / float(steps))
        for index in range(steps + 1)
    ]


class TesseractBackend:
    def __init__(self, urdf_path, srdf_path, manifest_path, deterministic_seed=42):
        try:
            from tesseract_robotics.planning import (
                box,
                CartesianTarget,
                create_obstacle,
                JointTarget,
                MotionProgram,
                plan_ompl,
                Pose,
                Robot,
                StateTarget,
            )
            from tesseract_robotics.tesseract_collision import (
                ContactRequest,
                ContactResultMap,
                ContactTestType_ALL,
            )
            from tesseract_robotics.tesseract_motion_planners_ompl import (
                RNG_setSeed,
            )
            import tesseract_robotics
        except ImportError as error:
            raise BackendUnavailable('Tesseract 0.35 bindings are unavailable: %s' % error)
        self.api = {
            'box': box,
            'CartesianTarget': CartesianTarget,
            'create_obstacle': create_obstacle,
            'JointTarget': JointTarget,
            'MotionProgram': MotionProgram,
            'plan_ompl': plan_ompl,
            'Pose': Pose,
            'Robot': Robot,
            'StateTarget': StateTarget,
            'ContactRequest': ContactRequest,
            'ContactResultMap': ContactResultMap,
            'ContactTestType_ALL': ContactTestType_ALL,
        }
        self.version = getattr(tesseract_robotics, '__version__', '0.35.0.6')
        if (
                isinstance(deterministic_seed, bool)
                or not isinstance(deterministic_seed, int)
                or deterministic_seed < 1
                or deterministic_seed > 0xffffffff):
            raise ContractError(
                'deterministic seed must be an integer from 1 through 2^32-1')
        self.deterministic_seed = deterministic_seed
        # OMPL must be seeded before any planner or sampler is constructed.
        RNG_setSeed(self.deterministic_seed)
        self.urdf_path = str(Path(urdf_path).resolve())
        self.srdf_path = str(Path(srdf_path).resolve())
        with open(manifest_path, 'r', encoding='utf-8') as stream:
            self.manifest = yaml.safe_load(stream)
        self.robot = None

    def collision_policy(self):
        default = float(self.manifest.get('hardware_clearance_margin_m', -1.0))
        report = float(self.manifest.get('clearance_report_distance_m', -1.0))
        maximum_l1 = float(
            self.manifest.get('validation_max_joint_l1_step_rad', -1.0))
        if not math.isfinite(default) or default < 0.0:
            raise ContractError('hardware_clearance_margin_m is missing or invalid')
        if not math.isfinite(report) or report <= default:
            raise ContractError(
                'clearance_report_distance_m must exceed the hardware margin')
        if not math.isfinite(maximum_l1) or maximum_l1 <= 0.0:
            raise ContractError(
                'validation_max_joint_l1_step_rad is missing or invalid')
        overrides = {}
        for index, item in enumerate(
                self.manifest.get('pair_clearance_overrides_m', [])):
            links = tuple(sorted(str(value) for value in item.get('links', [])))
            margin = float(item.get('margin_m', -1.0))
            if len(links) != 2 or not all(links) or links[0] == links[1]:
                raise ContractError(
                    'pair_clearance_overrides_m[%d] has invalid links' % index)
            if not math.isfinite(margin) or margin < 0.0 or margin >= report:
                raise ContractError(
                    'pair_clearance_overrides_m[%d] has invalid margin' % index)
            overrides[links] = margin
        return default, report, maximum_l1, overrides

    def configure_contact_manager(self, manager, report_distance):
        manager.setActiveCollisionObjects(self.robot.env.getActiveLinkNames())
        manager.setDefaultCollisionMargin(float(report_distance))
        return manager

    def motion_allowances(self):
        _default, _report, maximum_l1, _overrides = self.collision_policy()
        default_radius = float(
            self.manifest.get('default_relative_motion_radius_m', -1.0))
        if not math.isfinite(default_radius) or default_radius <= 0.0:
            raise ContractError(
                'default_relative_motion_radius_m is missing or invalid')
        allowances = {None: default_radius * maximum_l1}
        for index, item in enumerate(
                self.manifest.get('pair_clearance_overrides_m', [])):
            radius = float(item.get('relative_motion_radius_m', -1.0))
            if not math.isfinite(radius) or radius <= 0.0:
                raise ContractError(
                    'pair_clearance_overrides_m[%d] relative motion radius is invalid'
                    % index)
            pair = tuple(sorted(str(value) for value in item['links']))
            allowances[pair] = radius * maximum_l1
        return allowances

    @staticmethod
    def flattened_contacts(contacts):
        values = contacts.flattenCopyResults()
        return [values[index] for index in range(len(values))]

    @staticmethod
    def required_pair_clearance(link_names, default, overrides):
        pair = tuple(sorted(str(value) for value in link_names))
        return float(overrides.get(pair, default))

    def evaluate_contacts(
            self, contacts, default, overrides, stage, motion_allowances=None):
        minimum = math.inf
        limiting_pair = ('none', 'none')
        motion_allowances = motion_allowances or {}
        for contact in self.flattened_contacts(contacts):
            distance = float(contact.distance)
            pair = tuple(str(value) for value in contact.link_names)
            if not math.isfinite(distance):
                raise ContractError('%s returned a non-finite distance' % stage)
            if distance < minimum:
                minimum = distance
                limiting_pair = pair
            required = self.required_pair_clearance(pair, default, overrides)
            allowance = float(
                motion_allowances.get(tuple(sorted(pair)),
                                      motion_allowances.get(None, 0.0)))
            if distance < required + allowance:
                raise ContractError(
                    '%s clearance %.6fm is below %.6fm plus %.6fm motion bound '
                    'for %s/%s'
                    % (stage, distance, required, allowance, pair[0], pair[1]))
        return minimum, limiting_pair

    def contact_minimums(self, joints, report_distance):
        manager = self.configure_contact_manager(
            self.robot.env.getDiscreteContactManager(), report_distance)
        self.robot.env.setState(JOINT_NAMES, finite_six(joints, 'collision state'))
        state = self.robot.env.getState()
        manager.setCollisionObjectsTransform(state.link_transforms)
        contacts = self.api['ContactResultMap']()
        request = self.api['ContactRequest'](self.api['ContactTestType_ALL'])
        request.calculate_distance = True
        manager.contactTest(contacts, request)
        minimums = {}
        for contact in self.flattened_contacts(contacts):
            distance = float(contact.distance)
            if not math.isfinite(distance):
                raise ContractError('collision query returned non-finite distance')
            pair = tuple(sorted(str(value) for value in contact.link_names))
            minimums[pair] = min(distance, minimums.get(pair, math.inf))
        return minimums

    def clearance_violations(self, minimums):
        default, _report, _maximum_l1, overrides = self.collision_policy()
        allowances = self.motion_allowances()
        violations = {}
        for pair, distance in minimums.items():
            required = float(overrides.get(pair, default))
            required += float(allowances.get(pair, allowances[None]))
            if float(distance) < required:
                violations[pair] = {
                    'distance_m': float(distance),
                    'required_m': required,
                }
        return violations

    def bootstrap_recovery_policy(self, request):
        policy = self.manifest.get('bootstrap_start_recovery', {})
        if not bool(policy.get('enabled', False)):
            return None
        if request.get('plan_kind') != str(policy.get('plan_kind', '')):
            return None
        if request.get('scene', {}).get('observation_mode') != str(
                policy.get('observation_mode', '')):
            return None
        step = float(policy.get('search_step_rad', -1.0))
        maximum = float(policy.get('maximum_single_joint_delta_rad', -1.0))
        maximum_start_limit_violation = float(
            policy.get('maximum_start_limit_violation_rad', -1.0))
        tolerance = float(policy.get('monotonic_tolerance_m', -1.0))
        if (
                not math.isfinite(step) or step <= 0.0
                or not math.isfinite(maximum) or maximum < step
                or not math.isfinite(maximum_start_limit_violation)
                or maximum_start_limit_violation < 0.0
                or maximum_start_limit_violation > 0.04
                or not math.isfinite(tolerance) or tolerance < 0.0):
            raise ContractError('bootstrap start recovery policy is invalid')
        requested_start_tolerance = float(request.get('limits', {}).get(
            'bootstrap_start_limit_tolerance_rad', 0.0))
        if abs(
                requested_start_tolerance
                - maximum_start_limit_violation) > 1e-12:
            raise ContractError(
                'bootstrap start limit tolerance does not match qualified policy')
        allowed_start_limit_joints = [
            int(value) for value in
            policy.get('allowed_start_limit_joints', [])]
        maximum_recovery_joints = int(
            policy.get('maximum_recovery_joints', 1))
        if (
                not allowed_start_limit_joints
                or any(
                    value < 1 or value > 6
                    for value in allowed_start_limit_joints)
                or len(set(allowed_start_limit_joints))
                != len(allowed_start_limit_joints)
                or maximum_recovery_joints < 1
                or maximum_recovery_joints > 2):
            raise ContractError(
                'bootstrap allowed start limit joints are invalid')
        allowed = {}
        for index, item in enumerate(policy.get('allowed_start_contacts', [])):
            pair = tuple(sorted(str(value) for value in item.get('links', [])))
            penetration = float(item.get('maximum_penetration_m', -1.0))
            if (
                    len(pair) != 2 or pair[0] == pair[1]
                    or not math.isfinite(penetration) or penetration <= 0.0):
                raise ContractError(
                    'bootstrap allowed_start_contacts[%d] is invalid' % index)
            allowed[pair] = penetration
        if not allowed:
            raise ContractError('bootstrap start recovery has no allowed contacts')
        return {
            'step_rad': step,
            'maximum_delta_rad': maximum,
            'maximum_start_limit_violation_rad':
                maximum_start_limit_violation,
            'allowed_start_limit_joints': allowed_start_limit_joints,
            'maximum_recovery_joints': maximum_recovery_joints,
            'monotonic_tolerance_m': tolerance,
            'allowed_contacts': allowed,
        }

    def validate_bootstrap_recovery_path(self, positions, policy):
        """Prove a short path only leaves its bounded folded-start contacts."""
        if len(positions) < 2:
            raise ContractError('bootstrap recovery path has fewer than two points')
        _default, report, maximum_l1, _overrides = self.collision_policy()
        allowed = policy['allowed_contacts']
        tolerance = float(policy['monotonic_tolerance_m'])
        previous = None
        minimum = math.inf
        limiting_pair = ('none', 'none')
        sample_count = 0
        prior_position = None
        for point_index, position in enumerate(positions):
            current = finite_six(position, 'bootstrap recovery position')
            samples = [current] if prior_position is None else \
                subdivide_joint_segment(prior_position, current, maximum_l1)[1:]
            for local_index, sample in enumerate(samples):
                minimums = self.contact_minimums(sample, report)
                violations = self.clearance_violations(minimums)
                for pair, details in violations.items():
                    if pair not in allowed:
                        raise ContractError(
                            'bootstrap recovery introduces unapproved contact %s/%s'
                            % pair)
                    if details['distance_m'] < -float(allowed[pair]):
                        raise ContractError(
                            'bootstrap recovery contact %s/%s exceeds bounded '
                            'penetration' % pair)
                current_allowed = {
                    pair: float(minimums.get(pair, report)) for pair in allowed
                }
                if previous is not None:
                    for pair, distance in current_allowed.items():
                        if distance + tolerance < previous[pair]:
                            raise ContractError(
                                'bootstrap recovery worsens %s/%s contact'
                                % pair)
                previous = current_allowed
                for pair, distance in minimums.items():
                    if distance < minimum:
                        minimum = distance
                        limiting_pair = pair
                sample_count += 1
                prior_position = sample
        endpoint_violations = self.clearance_violations(
            self.contact_minimums(positions[-1], report))
        if endpoint_violations:
            pair = next(iter(endpoint_violations))
            raise ContractError(
                'bootstrap recovery endpoint does not reach normal clearance '
                'for %s/%s' % pair)
        if math.isinf(minimum):
            minimum = report
        return {
            'bootstrap_recovery_used': True,
            'bootstrap_recovery_end_positions_rad': finite_six(
                positions[-1], 'bootstrap recovery endpoint').tolist(),
            'bootstrap_recovery_minimum_clearance_m': float(minimum),
            'bootstrap_recovery_limiting_link_pair': '%s/%s' % limiting_pair,
            'bootstrap_recovery_samples': int(sample_count),
        }

    def find_bootstrap_recovery(self, request):
        policy = self.bootstrap_recovery_policy(request)
        if policy is None:
            return None
        start = finite_six(
            request['start_state']['positions_rad'], 'bootstrap start')
        _default, report, _maximum_l1, _overrides = self.collision_policy()
        start_minimums = self.contact_minimums(start, report)
        start_violations = self.clearance_violations(start_minimums)
        bounds = np.asarray(request['limits']['position_rad'], dtype=float)
        limit_violations = []
        for joint_index, value in enumerate(start):
            low, high = bounds[joint_index]
            if value < low or value > high:
                distance = max(low - value, value - high)
                if (
                        joint_index + 1
                        not in policy['allowed_start_limit_joints']):
                    raise ContractError(
                        'start state joint%d has no qualified limit recovery'
                        % (joint_index + 1))
                if (
                        distance
                        > policy['maximum_start_limit_violation_rad'] + 1e-12):
                    raise ContractError(
                        'start state joint%d exceeds bootstrap limit recovery bound'
                        % (joint_index + 1))
                limit_violations.append(joint_index)
        if len(limit_violations) > int(
                policy.get('maximum_recovery_joints', 1)):
            raise ContractError(
                'bootstrap limit recovery has too many outside joints')
        if not start_violations and not limit_violations:
            return None
        for pair, details in start_violations.items():
            if pair not in policy['allowed_contacts']:
                raise ContractError(
                    'start state violates unapproved collision pair %s/%s' % pair)
            if details['distance_m'] < -float(policy['allowed_contacts'][pair]):
                raise ContractError(
                    'start state %s/%s penetration exceeds bootstrap bound' % pair)

        maximum_steps = int(math.floor(
            policy['maximum_delta_rad'] / policy['step_rad'] + 1e-9))
        failures = []
        if len(limit_violations) > 1:
            margin = float(request.get('limits', {}).get(
                'joint_margin_rad', 0.0))
            candidate = start.copy()
            for joint_index in limit_violations:
                low = bounds[joint_index, 0] + margin
                high = bounds[joint_index, 1] - margin
                if low >= high:
                    raise ContractError(
                        'bootstrap recovery joint margin collapses a limit')
                candidate[joint_index] = float(np.clip(
                    candidate[joint_index], low, high))
            deltas = candidate - start
            if any(
                    abs(float(deltas[index]))
                    > policy['maximum_delta_rad'] + 1e-12
                    for index in limit_violations):
                raise ContractError(
                    'bootstrap multi-joint recovery exceeds its delta bound')
            positions = subdivide_joint_segment(
                start, candidate, self.collision_policy()[2])
            try:
                evidence = self.validate_bootstrap_recovery_path(
                    positions, policy)
            except ContractError as error:
                failures.append(str(error))
            else:
                joint_numbers = [
                    int(index + 1) for index in limit_violations]
                declared_deltas = [
                    float(deltas[index]) for index in limit_violations]
                evidence.update({
                    'bootstrap_recovery_joint': 0,
                    'bootstrap_recovery_delta_rad': 0.0,
                    'bootstrap_recovery_joints': joint_numbers,
                    'bootstrap_recovery_deltas_rad': declared_deltas,
                    'bootstrap_start_contacts': [
                        {
                            'links': list(pair),
                            'distance_m': float(details['distance_m']),
                            'required_m': float(details['required_m']),
                        }
                        for pair, details in sorted(
                            start_violations.items())
                    ],
                    'positions': positions,
                })
                return evidence
        for step_index in range(1, maximum_steps + 1):
            magnitude = step_index * policy['step_rad']
            joint_indices = limit_violations or list(range(6))
            for joint_index in joint_indices:
                if limit_violations:
                    signs = (
                        (1.0,) if start[joint_index] < bounds[joint_index, 0]
                        else (-1.0,))
                else:
                    signs = (-1.0, 1.0)
                for sign in signs:
                    candidate = start.copy()
                    candidate[joint_index] += sign * magnitude
                    if (
                            candidate[joint_index] < bounds[joint_index, 0]
                            or candidate[joint_index] > bounds[joint_index, 1]):
                        continue
                    if self.clearance_violations(
                            self.contact_minimums(candidate, report)):
                        continue
                    positions = subdivide_joint_segment(
                        start, candidate,
                        self.collision_policy()[2])
                    try:
                        evidence = self.validate_bootstrap_recovery_path(
                            positions, policy)
                    except ContractError as error:
                        failures.append(str(error))
                        continue
                    evidence.update({
                        'bootstrap_recovery_joint': int(joint_index + 1),
                        'bootstrap_recovery_delta_rad': float(
                            candidate[joint_index] - start[joint_index]),
                        'bootstrap_recovery_joints': [
                            int(joint_index + 1)],
                        'bootstrap_recovery_deltas_rad': [
                            float(candidate[joint_index] - start[joint_index])],
                        'bootstrap_start_contacts': [
                            {
                                'links': list(pair),
                                'distance_m': float(details['distance_m']),
                                'required_m': float(details['required_m']),
                            }
                            for pair, details in sorted(start_violations.items())
                        ],
                        'positions': positions,
                    })
                    return evidence
        raise ContractError(
            'no bounded monotonic bootstrap recovery reaches normal clearance%s'
            % (': ' + failures[0] if failures else ''))

    def reset_scene(self):
        """Create a clean environment so request-local obstacles cannot leak."""
        # The nanobind Robot owns native Bullet/OMPL resources.  Drop the old
        # environment before loading the next request-local scene so repeated
        # qualification/planning does not retain every decomposed collision
        # object until process exit.
        self.robot = None
        gc.collect()
        self.robot = self.api['Robot'].from_files(self.urdf_path, self.srdf_path)

    def remaining_planning_time(self, context):
        """Return bounded OMPL time while reserving time to emit the response."""
        deadline = getattr(self, 'planning_deadline_monotonic', None)
        if deadline is None:
            return 5.0
        remaining = (
            float(deadline) - time.monotonic()
            - WORKER_RESPONSE_RESERVE_SEC)
        if remaining <= 0.0:
            raise ContractError(
                'Tesseract planning exceeded the internal %.0f-second '
                'budget before the bridge 180-second timeout (%s)'
                % (WORKER_PLANNING_BUDGET_SEC, context))
        return min(5.0, remaining)

    def ensure_planning_time(self, context):
        TesseractBackend.remaining_planning_time(self, context)

    def planning_profiles(self, planning_time_sec=5.0):
        """
        Build RRTConnect/TrajOpt profiles with the model's explicit margin.

        Tesseract's convenience profile uses a generic 10 mm robot margin. That
        is larger than normal internal clearances in this compact arm, so the
        proposal-only model records its own temporary margin explicitly. It is
        not a hardware qualification value.
        """
        from tesseract_robotics.planning.profiles import (
            _add_trajopt_to_profiles,
            OMPL_DEFAULT_NAMESPACE,
            STANDARD_PROFILE_NAMES,
        )
        from tesseract_robotics.tesseract_command_language import ProfileDictionary
        from tesseract_robotics.tesseract_common import (
            CollisionMarginPairOverrideType,
        )
        from tesseract_robotics.tesseract_motion_planners_ompl import (
            OMPLRealVectorPlanProfile,
            ProfileDictionary_addOMPLProfile,
            RRTConnectConfigurator,
        )

        margin = float(self.manifest.get('proposal_collision_margin_m', -1.0))
        if not math.isfinite(margin) or margin < 0.0:
            raise ContractError('proposal_collision_margin_m is missing or invalid')
        profile = OMPLRealVectorPlanProfile()
        profile.solver_config.planning_time = max(
            0.05, min(5.0, float(planning_time_sec)))
        profile.solver_config.max_solutions = 4
        profile.solver_config.optimize = False
        profile.solver_config.simplify = True
        profile.solver_config.clearPlanners()
        profile.solver_config.addPlanner(RRTConnectConfigurator())
        profile.contact_manager_config.default_margin = margin
        for item in self.manifest.get('pair_clearance_overrides_m', []):
            links = item['links']
            profile.contact_manager_config.pair_margin_data.setCollisionMargin(
                str(links[0]), str(links[1]), float(item['margin_m']))
        profile.contact_manager_config.pair_margin_override_type = (
            CollisionMarginPairOverrideType.REPLACE)
        profiles = ProfileDictionary()
        for name in STANDARD_PROFILE_NAMES:
            ProfileDictionary_addOMPLProfile(
                profiles, OMPL_DEFAULT_NAMESPACE, name, profile)
        _add_trajopt_to_profiles(profiles, STANDARD_PROFILE_NAMES)
        return profiles

    def add_obstacles(self, obstacles):
        for index, obstacle in enumerate(obstacles):
            minimum = np.asarray(obstacle['minimum_m'], dtype=float)
            maximum = np.asarray(obstacle['maximum_m'], dtype=float)
            size = maximum - minimum
            if np.any(size <= 0.0) or not np.all(np.isfinite(size)):
                raise ContractError('obstacle %d has invalid bounds' % index)
            center = 0.5 * (minimum + maximum)
            name = re.sub(r'[^A-Za-z0-9_]', '_', str(obstacle['id']))[:60]
            added = self.api['create_obstacle'](
                self.robot,
                'scene_%03d_%s' % (index, name),
                self.api['box'](*size.tolist()),
                self.api['Pose'].from_xyz(*center.tolist()),
            )
            if not added:
                raise ContractError('obstacle %d could not be added to the scene' % index)

    def state_in_collision(self, joints):
        manager = self.robot.env.getDiscreteContactManager()
        manager.setActiveCollisionObjects(self.robot.env.getActiveLinkNames())
        self.robot.env.setState(JOINT_NAMES, np.asarray(joints, dtype=float))
        state = self.robot.env.getState()
        manager.setCollisionObjectsTransform(state.link_transforms)
        contacts = self.api['ContactResultMap']()
        manager.contactTest(
            contacts,
            self.api['ContactRequest'](self.api['ContactTestType_ALL']),
        )
        return contacts.size() != 0

    def ik_joint_goals(self, start, candidate, roll, position_limits=None,
                       joint_margin=0.0):
        """Return finite, bounded, collision-free IK branches nearest the start."""
        position = candidate['camera_position_m']
        quaternion = look_at_quaternion(candidate['look_direction'], roll)
        pose = self.api['Pose'].from_xyz_quat(*(position + quaternion))
        start_array = finite_six(start, 'segment start')
        if position_limits is None:
            lower = np.full(6, -math.inf)
            upper = np.full(6, math.inf)
            random_lower = np.asarray([-2.6, 0.0, -2.7, -1.8, -1.2, -2.1])
            random_upper = np.asarray([2.6, 3.1, 0.0, 1.8, 1.2, 2.1])
        else:
            bounds = np.asarray(position_limits, dtype=float)
            if bounds.shape != (6, 2) or not np.all(np.isfinite(bounds)):
                raise ContractError('position limits must be a finite 6x2 array')
            lower = bounds[:, 0] + float(joint_margin)
            upper = bounds[:, 1] - float(joint_margin)
            if np.any(lower >= upper):
                raise ContractError('joint margin collapses a position limit')
            random_lower, random_upper = lower, upper
        seeds = [
            start_array,
            np.clip(np.asarray([0.0, 0.8, -0.7, 0.0, 0.7, 0.0]),
                    random_lower, random_upper),
            np.clip(np.asarray([
                0.3189509166, 0.7800870124, -1.6258884709,
                -0.6660237320, -0.2154052887, 0.0403545644,
            ]), random_lower, random_upper),
        ]
        rng = np.random.default_rng(42)
        seeds.extend(rng.uniform(random_lower, random_upper) for _ in range(6))
        goals = []
        desired = np.asarray(pose.matrix)
        for seed in seeds:
            solution = self.robot.ik(
                'manipulator', pose, seed=np.asarray(seed, dtype=float),
                tip_link='camera_optical_frame', all_solutions=False,
            )
            if solution is None:
                continue
            solution = finite_six(solution, 'IK solution')
            if np.any(solution < lower) or np.any(solution > upper):
                continue
            if any(float(np.max(np.abs(solution - existing))) < 1e-5
                   for existing in goals):
                continue
            if self.state_in_collision(solution):
                continue
            actual = np.asarray(self.robot.fk(
                'manipulator', solution,
                tip_link='camera_optical_frame').matrix)
            position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
            rotation_error = actual[:3, :3].T @ desired[:3, :3]
            angle_error = float(math.acos(np.clip(
                (np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0)))
            if position_error > 1e-4 or angle_error > 1e-3:
                continue
            goals.append(solution)
        goals.sort(key=lambda goal: (
            float(np.max(np.abs(goal - start_array))),
            float(np.linalg.norm(goal - start_array)),
        ))
        return goals

    def plan_segment(self, start, candidate, roll, maximum_step,
                     position_limits=None, joint_margin=0.0):
        goals = self.ik_joint_goals(
            start, candidate, roll, position_limits, joint_margin)
        if not goals:
            raise ContractError('no finite bounded collision-free IK goal')
        failures = []
        for goal in goals[:4]:
            self.ensure_planning_time('before an IK-goal OMPL attempt')
            try:
                return self.plan_segment_to_joint_goal(
                    start, goal, maximum_step)
            except (ContractError, RuntimeError, ValueError) as error:
                failures.append(str(error))
        raise ContractError('all %d shortlisted IK goals failed planning: %s' % (
            min(len(goals), 4), failures[0] if failures else 'unknown failure'))

    def plan_candidate(self, start, candidate, rolls, maximum_step,
                       position_limits, joint_margin, bootstrap_recovery=None):
        """Rank IK across every roll, then bound expensive planner attempts."""
        start_array = finite_six(start, 'candidate start')
        ranked = []
        for roll in rolls:
            goals = self.ik_joint_goals(
                start_array, candidate, roll, position_limits, joint_margin)
            for goal in goals[:2]:
                ranked.append((
                    float(np.max(np.abs(goal - start_array))),
                    float(np.linalg.norm(goal - start_array)),
                    float(roll),
                    goal,
                ))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if not ranked:
            raise ContractError('no finite bounded collision-free IK goal for any roll')
        failures = []
        for _, _, roll, goal in ranked[:4]:
            self.ensure_planning_time(
                'before a candidate roll/IK OMPL attempt')
            try:
                points, validation = self.plan_segment_to_joint_goal(
                    start_array, goal, maximum_step, bootstrap_recovery)
                return roll, points, validation
            except (ContractError, RuntimeError, ValueError) as error:
                failures.append('roll %.3f: %s' % (roll, error))
        raise ContractError('all %d shortlisted roll/IK goals failed planning: %s' % (
            min(len(ranked), 4), failures[0] if failures else 'unknown failure'))

    def plan_segment_to_joint_goal(
            self, start, goal, maximum_step, bootstrap_recovery=None):
        planning_time = self.remaining_planning_time(
            'before an OMPL segment solve')
        program = self.api['MotionProgram'](
            'manipulator', tcp_frame='camera_optical_frame').set_joint_names(JOINT_NAMES)
        program.start_at(self.api['JointTarget'](np.asarray(start, dtype=float)))
        program.move_to(self.api['JointTarget'](np.asarray(goal, dtype=float)))
        result = self.api['plan_ompl'](
            self.robot, program, pipeline='OMPLPipeline',
            profiles=self.planning_profiles(planning_time))
        self.ensure_planning_time('after an OMPL segment solve')
        if not result.successful:
            raise ContractError('OMPL failed: %s' % result.message)
        if result.raw_results is None:
            raise ContractError('OMPL returned no instruction trajectory')
        raw = self.time_parameterize(result.raw_results)
        self.ensure_planning_time('after ISP time parameterization')
        if bootstrap_recovery is not None:
            recovery_positions = [
                finite_six(item, 'bootstrap recovery position')
                for item in bootstrap_recovery['positions']
            ]
            ompl_positions = [
                finite_six(item['positions_rad'], 'OMPL trajectory position')
                for item in raw
            ]
            recovery_start_delta = np.abs(
                recovery_positions[-1] - ompl_positions[0])
            if float(np.max(recovery_start_delta)) > 1e-6:
                raise ContractError(
                    'bootstrap recovery endpoint does not match OMPL start: '
                    'maximum delta %.9f rad; endpoint=%s; OMPL=%s; '
                    'request_limits=%s'
                    % (
                        float(np.max(recovery_start_delta)),
                        recovery_positions[-1].tolist(),
                        ompl_positions[0].tolist(),
                        self.execution_position_limits,
                    ))
            combined = recovery_positions + ompl_positions[1:]
            raw = self.time_parameterize_positions(combined)
            quiesce_bootstrap_recovery_prefix(
                raw,
                bootstrap_recovery['bootstrap_recovery_end_positions_rad'],
                bootstrap_recovery['bootstrap_recovery_joints'])

        points, validation_points = sdk_movej_waypoint_trajectory(
            raw,
            self.execution_speed_percent,
            self.command_rate_hz,
            maximum_step,
            self.execution_position_limits,
            self.execution_velocity_limits,
            self.execution_acceleration_limits,
            mandatory_waypoints=(
                [bootstrap_recovery[
                    'bootstrap_recovery_end_positions_rad']]
                if bootstrap_recovery is not None else []),
            bootstrap_start_limit_tolerance_rad=(
                self.bootstrap_start_limit_tolerance_rad
                if bootstrap_recovery is not None else 0.0),
        )
        if float(np.max(np.abs(
                np.asarray(points[-1]['positions_rad']) - np.asarray(goal)))) > 1e-4:
            raise ContractError('planned endpoint does not match validated IK goal')
        if bootstrap_recovery is None:
            self.ensure_planning_time(
                'before adaptive collision validation')
            validation = self.final_validate(validation_points)
        else:
            endpoint = finite_six(
                bootstrap_recovery['bootstrap_recovery_end_positions_rad'],
                'bootstrap recovery endpoint')
            boundary = next((
                index for index, point in enumerate(points)
                if float(np.max(np.abs(
                    finite_six(point['positions_rad'], 'trajectory position')
                    - endpoint))) <= 1e-7
            ), None)
            if boundary is None or boundary < 1:
                raise ContractError(
                    'time-parameterized path lost bootstrap recovery endpoint')
            validation_boundary = next((
                index for index, point in enumerate(validation_points)
                if float(np.max(np.abs(
                    finite_six(point['positions_rad'], 'trajectory position')
                    - endpoint))) <= 1e-7
            ), None)
            if validation_boundary is None or validation_boundary < 1:
                raise ContractError(
                    'validation path lost bootstrap recovery endpoint')
            recovery_validation = self.validate_bootstrap_recovery_path(
                [
                    item['positions_rad']
                    for item in validation_points[:validation_boundary + 1]
                ],
                self.bootstrap_recovery_policy({
                    'plan_kind': 'ROUGH_ACQUISITION',
                    'scene': {'observation_mode': 'bootstrap_static'},
                    'limits': {
                        'bootstrap_start_limit_tolerance_rad':
                            self.bootstrap_start_limit_tolerance_rad,
                    },
                }),
            )
            self.ensure_planning_time(
                'before adaptive collision validation')
            validation = self.final_validate(
                validation_points[validation_boundary:])
            validation.update({
                key: value for key, value in recovery_validation.items()
                if key != 'positions'
            })
            validation.update({
                'bootstrap_recovery_end_point': int(boundary),
                'bootstrap_recovery_joint': int(
                    bootstrap_recovery['bootstrap_recovery_joint']),
                'bootstrap_recovery_delta_rad': float(
                    bootstrap_recovery['bootstrap_recovery_delta_rad']),
                'bootstrap_recovery_joints': [
                    int(value) for value in
                    bootstrap_recovery['bootstrap_recovery_joints']],
                'bootstrap_recovery_deltas_rad': [
                    float(value) for value in
                    bootstrap_recovery['bootstrap_recovery_deltas_rad']],
                'bootstrap_start_contacts':
                    bootstrap_recovery['bootstrap_start_contacts'],
            })
        return points, validation

    def time_parameterize_positions(self, positions):
        """Run ISP once over a validated geometric joint path."""
        program = self.api['MotionProgram'](
            'manipulator', tcp_frame='camera_optical_frame').set_joint_names(
                JOINT_NAMES)
        for index, position in enumerate(positions):
            target = self.api['StateTarget'](
                finite_six(position, 'joint path position'),
                names=list(JOINT_NAMES),
            )
            if index == 0:
                program.start_at(target)
            else:
                program.move_to(target)
        return self.time_parameterize(program.to_composite_instruction())

    def time_parameterize(self, instructions):
        """Apply ISP to the exact raw OMPL joint path and extract finite fields."""
        from tesseract_robotics.tesseract_command_language import ProfileDictionary
        from tesseract_robotics.tesseract_time_parameterization import (
            InstructionsTrajectory,
            ISPCompositeProfile,
            IterativeSplineParameterization,
        )

        profile = ISPCompositeProfile()
        profile.max_velocity_scaling_factor = 1.0
        profile.max_acceleration_scaling_factor = 1.0
        profiles = ProfileDictionary()
        profiles.addProfile('ISP', 'DEFAULT', profile)
        parameterizer = IterativeSplineParameterization()
        if not parameterizer.compute(instructions, self.robot.env, profiles):
            raise ContractError('ISP time parameterization failed')
        trajectory = InstructionsTrajectory(instructions)
        raw = []
        for index in range(trajectory.size()):
            positions = finite_six(trajectory.getPosition(index), 'positions')
            velocities = finite_six(trajectory.getVelocity(index), 'velocities')
            accelerations = finite_six(
                trajectory.getAcceleration(index), 'accelerations')
            when = float(trajectory.getTimeFromStart(index))
            if not math.isfinite(when) or when < 0.0:
                raise ContractError('trajectory time is missing or non-finite')
            raw.append({
                'time_from_start_s': when,
                'positions_rad': positions.tolist(),
                'velocities_rad_s': velocities.tolist(),
                'accelerations_rad_s2': accelerations.tolist(),
            })
        if len(raw) < 2 or any(
                second['time_from_start_s'] <= first['time_from_start_s']
                for first, second in zip(raw[:-1], raw[1:])):
            raise ContractError('ISP timestamps are not strictly increasing')
        return raw

    def final_validate(self, points):
        """
        Validate the exact post-ISP path with bounded adaptive sampling.

        The current 0.35.0.6 nanobind continuous manager crashes in this model.
        A conservative relative-link motion bound makes dense discrete checking
        a deterministic swept-path proof without calling that unstable binding.
        """
        default, report, maximum_l1, overrides = self.collision_policy()
        discrete = self.configure_contact_manager(
            self.robot.env.getDiscreteContactManager(), report)
        request = self.api['ContactRequest'](self.api['ContactTestType_ALL'])
        request.calculate_distance = True
        motion_allowances = self.motion_allowances()
        default_radius = (
            float(motion_allowances[None]) / float(maximum_l1))
        minimum = math.inf
        limiting_pair = ('none', 'none')
        sample_count = 0

        def record(value, pair):
            nonlocal minimum, limiting_pair
            if value < minimum:
                minimum = value
                limiting_pair = pair

        previous = None
        for point_index, point in enumerate(points):
            current = finite_six(point['positions_rad'], 'trajectory position')
            samples = [current] if previous is None else subdivide_joint_segment(
                previous, current, maximum_l1)[1:]
            for local_index, sample in enumerate(samples):
                self.ensure_planning_time(
                    'during adaptive collision validation')
                self.robot.env.setState(JOINT_NAMES, sample)
                state = self.robot.env.getState()
                discrete.setCollisionObjectsTransform(state.link_transforms)
                contacts = self.api['ContactResultMap']()
                discrete.contactTest(contacts, request)
                value, pair = self.evaluate_contacts(
                    contacts, default, overrides,
                    'adaptive point %d.%d' % (point_index, local_index),
                    motion_allowances)
                record(value, pair)
                sample_count += 1
                previous = sample

        if math.isinf(minimum):
            minimum = report
            limiting_pair = ('none_within_report_distance', 'none')
        return {
            'minimum_clearance_m': float(minimum),
            'limiting_link_pair': '%s/%s' % limiting_pair,
            'validation': 'adaptive_dense_discrete_bullet_with_motion_bound',
            'discrete_samples': sample_count,
            'maximum_joint_l1_step_rad': maximum_l1,
            'default_relative_motion_bound_m': (
                default_radius * maximum_l1),
        }

    def plan(self, request):
        planning_deadline = time.monotonic() + WORKER_PLANNING_BUDGET_SEC
        self.planning_deadline_monotonic = planning_deadline
        self.reset_scene()
        self.add_obstacles(request['scene'].get('obstacles', []))
        self.execution_speed_percent = float(
            request['planning']['effective_speed_percent'])
        self.command_rate_hz = float(
            request['planning']['command_rate_hz'])
        self.execution_position_limits = request['limits']['position_rad']
        self.execution_velocity_limits = request[
            'limits']['max_velocity_rad_s']
        self.execution_acceleration_limits = request[
            'limits']['max_acceleration_rad_s2']
        self.bootstrap_start_limit_tolerance_rad = float(
            request['limits'].get(
                'bootstrap_start_limit_tolerance_rad', 0.0))
        minimum_views = int(request['planning']['min_viewpoints'])
        maximum_views = int(request['planning']['max_viewpoints'])
        maximum_step = float(request['planning']['max_execution_joint_step_rad'])
        position_limits = request['limits']['position_rad']
        joint_margin = float(request['limits'].get('joint_margin_rad', 0.0))
        rolls = [float(value) for value in request['planning']['roll_samples_rad']]
        current = request['start_state']['positions_rad']
        bootstrap_recovery = self.find_bootstrap_recovery(request)
        if bootstrap_recovery is not None:
            current = bootstrap_recovery[
                'bootstrap_recovery_end_positions_rad']
        selected = []
        segments = []
        failures = {}
        pending = list(request['scene']['candidate_views'])
        while pending and len(selected) < maximum_views:
            next_pending = []
            progress = False
            for candidate_index, candidate in enumerate(pending):
                TesseractBackend.ensure_planning_time(
                    self, 'before starting a viewpoint candidate')
                try:
                    accepted = self.plan_candidate(
                        current, candidate, rolls, maximum_step,
                        position_limits, joint_margin, bootstrap_recovery)
                except (ContractError, RuntimeError, ValueError) as error:
                    failures[int(candidate['id'])] = str(error)
                    next_pending.append(candidate)
                    continue
                roll, points, validation = accepted
                selected.append({
                    'id': int(candidate['id']),
                    'camera_position_m': candidate['camera_position_m'],
                    'look_direction': candidate['look_direction'],
                    'roll_rad': roll,
                })
                segments.append({
                    'from_viewpoint': (
                        -1 if len(selected) == 1
                        else int(selected[-2]['id'])),
                    'to_viewpoint': int(candidate['id']),
                    'points': points,
                    **validation,
                })
                current = points[-1]['positions_rad']
                bootstrap_recovery = None
                failures.pop(int(candidate['id']), None)
                pending = (
                    next_pending + pending[candidate_index + 1:])
                progress = True
                break
            if not progress:
                break
        if len(selected) < minimum_views:
            raise ContractError(
                'only %d viewpoints planned; require at least %d of %d (%s)' % (
                    len(selected), minimum_views, maximum_views,
                    '; '.join(
                        'view %s: %s' % item
                        for item in sorted(failures.items()))
                    if failures else 'no candidates'))
        if request.get('plan_kind') == 'MULTIVIEW_SCAN':
            TesseractBackend.ensure_planning_time(
                self, 'before return-home qualification')
            home = finite_six(
                request['planning']['return_home_positions_rad'],
                'return home positions')
            points, validation = self.plan_segment_to_joint_goal(
                current, home, maximum_step)
            segments.append({
                'from_viewpoint': int(selected[-1]['id']),
                'to_viewpoint': -2,
                'is_return_home': True,
                'points': points,
                **validation,
            })
        return selected, segments


class Worker:
    def __init__(self, spool_root, urdf_path, srdf_path, manifest_path):
        self.spool = Spool(spool_root)
        self.backend_error = None
        try:
            self.backend = TesseractBackend(urdf_path, srdf_path, manifest_path)
        except (BackendUnavailable, ContractError, OSError, RuntimeError, ValueError) as error:
            self.backend = None
            self.backend_error = str(error)
        self.running = True
        self.generation_id = uuid.uuid4().hex
        self.last_heartbeat_at = -float('inf')
        self.heartbeat_stop = threading.Event()
        self.heartbeat_thread = None

    def stop(self, *_args):
        self.running = False

    def publish_health(self, ready=None):
        if ready is None:
            ready = self.running and self.backend is not None
        self.spool.write_health({
            'schema_version': SCHEMA_VERSION,
            'generation_id': self.generation_id,
            'written_at_ns': time.time_ns(),
            'worker_ready': bool(ready),
            'backend': 'tesseract',
            'backend_version': getattr(
                self.backend, 'version', 'unavailable'),
            'backend_error': self.backend_error or '',
        })
        self.last_heartbeat_at = time.monotonic()

    def heartbeat_loop(self):
        while not self.heartbeat_stop.wait(0.5):
            try:
                self.publish_health()
            except (ContractError, OSError, TypeError, ValueError):
                # The bridge fails closed when the record is absent or stale.
                # Keep processing so a transient filesystem error can recover.
                pass

    def process_once(self):
        request_id, request = self.spool.claim_next()
        if request_id is None:
            return False
        try:
            validate_request(request)
            if self.backend is None:
                raise BackendUnavailable(self.backend_error or 'backend unavailable')
            requested_seed = request['planning']['deterministic_seed']
            if requested_seed != self.backend.deterministic_seed:
                raise ContractError(
                    'request deterministic seed %d does not match worker seed %d'
                    % (requested_seed, self.backend.deterministic_seed))
            selected, segments = self.backend.plan(request)
            binding = {
                'request_sha256': request['request_sha256'],
                'plan_kind': request['plan_kind'],
                'target_provenance': request['target_provenance'],
                'model': request['model'],
                'calibration': request['calibration']['hand_eye_sha256'],
                'limits': request['limits'],
                'execution': {
                    'effective_speed_percent':
                        request['planning']['effective_speed_percent'],
                    'command_rate_hz':
                        request['planning']['command_rate_hz'],
                    'timing_policy':
                        request['planning']['timing_policy'],
                },
            }
            response = {
                'schema_version': SCHEMA_VERSION,
                'plan_kind': request['plan_kind'],
                'target_provenance': request['target_provenance'],
                'request_id': request_id,
                'request_sha256': request['request_sha256'],
                'plan_id': request_id[:16],
                'status': 'success',
                'backend': 'tesseract',
                'backend_version': self.backend.version,
                'deterministic_seed': self.backend.deterministic_seed,
                'joint_names': list(JOINT_NAMES),
                'collision_model_qualified': bool(
                    self.backend.manifest.get('qualified_for_hardware', False)),
                'diagnostic': 'CPU Tesseract proposal; hardware qualification=%s' % bool(
                    self.backend.manifest.get('qualified_for_hardware', False)),
                'rejection_codes': [],
                'target_center_m': request['scene']['target_center_m'],
                'selected_viewpoints': selected,
                'segments': segments,
                'trajectory_binding': binding,
                'trajectory_sha256': trajectory_digest(segments, binding),
            }
        except (
                BackendUnavailable, ContractError, KeyError, OSError,
                RuntimeError, ValueError) as error:
            response = {
                'schema_version': SCHEMA_VERSION,
                'plan_kind': request.get('plan_kind', ''),
                'target_provenance': request.get('target_provenance', {}),
                'request_id': request_id,
                'request_sha256': request.get('request_sha256', ''),
                'status': 'failed',
                'backend': 'tesseract',
                'backend_version': getattr(self.backend, 'version', 'unavailable'),
                'rejection_codes': [
                    'BACKEND_UNAVAILABLE' if isinstance(error, BackendUnavailable)
                    else 'PLANNING_FAILED'
                ],
                'diagnostic': str(error),
            }
        response = attach_digest(response, 'response_sha256')
        self.spool.write('responses', request_id, response)
        processing = self.spool.path('processing', request_id)
        if processing.exists():
            processing.unlink()
        return True

    def run(self, once=False):
        try:
            self.publish_health()
            self.heartbeat_thread = threading.Thread(
                target=self.heartbeat_loop,
                name='tesseract-worker-heartbeat',
                daemon=True,
            )
            self.heartbeat_thread.start()
            while self.running:
                processed = self.process_once()
                if once:
                    return 0 if processed else 1
                if not processed:
                    time.sleep(0.2)
            return 0
        finally:
            self.heartbeat_stop.set()
            if self.heartbeat_thread is not None:
                self.heartbeat_thread.join(timeout=1.0)
            try:
                self.publish_health(ready=False)
            except (ContractError, OSError, TypeError, ValueError):
                pass


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--spool-root', default=os.environ.get(
        'PIPER_TESSERACT_SPOOL', '/spool'))
    parser.add_argument('--urdf', default=os.environ.get(
        'PIPER_TESSERACT_URDF', '/models/piper_planning.urdf'))
    parser.add_argument('--srdf', default=os.environ.get(
        'PIPER_TESSERACT_SRDF', '/models/piper.srdf'))
    parser.add_argument('--collision-manifest', default=os.environ.get(
        'PIPER_TESSERACT_COLLISION_MANIFEST', '/models/collision_model.yaml'))
    parser.add_argument('--once', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    worker = Worker(args.spool_root, args.urdf, args.srdf, args.collision_manifest)
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    return worker.run(once=args.once)


if __name__ == '__main__':
    raise SystemExit(main())
