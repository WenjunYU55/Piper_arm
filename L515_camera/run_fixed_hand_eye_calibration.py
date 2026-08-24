#!/usr/bin/env python3
"""Run the recorded, repeatable L515 hand-eye calibration pose sequence.

This commissioning tool is deliberately separate from the production mission.
It assumes that the operator has already started the normal PiPER driver and
camera and has enabled the arm.  It owns the joint-command topic only for the
duration of the run, captures one validated sample after every settled pose,
and always attempts PRE_HOME -> ROUGH_HOME -> STORAGE_WRIST before disable.
"""

import argparse
from datetime import datetime
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable
from piper_mobile_manipulation.home_pose import load_home_pose
from sensor_msgs.msg import CameraInfo, JointState
import yaml


EXACT_CONFIRMATION = 'RUN FIXED HAND EYE POSES'
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def load_pose_file(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    if data.get('schema_version') != 1:
        raise ValueError('fixed-pose schema must be 1')
    if data.get('joint_names') != JOINT_NAMES:
        raise ValueError('fixed-pose joint order is invalid')
    poses = data.get('poses')
    if not isinstance(poses, list) or not poses:
        raise ValueError('fixed-pose list is empty')
    seen = set()
    groups = {'fitting': 0, 'validation': 0}
    for pose in poses:
        identifier = str(pose.get('id', ''))
        values = pose.get('positions_rad')
        if not identifier or identifier in seen:
            raise ValueError('pose IDs must be nonempty and unique')
        seen.add(identifier)
        if (not isinstance(values, list) or len(values) != 6
                or not all(math.isfinite(float(value)) for value in values)):
            raise ValueError('%s must contain six finite joints' % identifier)
        if abs(float(values[5])) > math.pi + 1e-9:
            raise ValueError('%s exceeds the configured J6 +/-pi bound' % identifier)
        if pose.get('capture', True):
            group = pose.get('group')
            if group not in groups:
                raise ValueError('%s has an invalid capture group' % identifier)
            groups[group] += 1
        elif pose.get('group') is not None:
            raise ValueError('%s transit must not have a capture group' % identifier)
    if groups['fitting'] < 3 or groups['validation'] < 1:
        raise ValueError('need at least three fitting and one validation pose')
    profile = data.get('recommended_calibration_profile') or {}
    if int(profile.get('width', 0)) <= 0 or int(profile.get('height', 0)) <= 0:
        raise ValueError('recommended camera profile is invalid')
    return data


def calibration_home_targets(profile):
    if not isinstance(profile, dict):
        raise ValueError('home profile is missing')
    if not profile.get('pre_home_configured', False):
        raise ValueError('configured PRE_HOME is required')
    if not profile.get('staged_home_configured', False):
        raise ValueError('configured staged home is required')
    pre_home = [float(value) for value in profile['pre_home_positions_rad']]
    rough_home = [float(value) for value in profile['positions_rad']]
    storage = list(rough_home)
    storage[5] = float(profile['storage_joint6_rad'])
    for label, values in (
            ('PRE_HOME', pre_home), ('ROUGH_HOME', rough_home),
            ('STORAGE_WRIST', storage)):
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError('%s must contain six finite joints' % label)
    return pre_home, rough_home, storage


class FixedPoseRunner(Node):
    def __init__(self, speed_percent):
        super().__init__('fixed_hand_eye_calibration_runner')
        self.speed_percent = float(speed_percent)
        self.lock = threading.Lock()
        self.latest_joints = None
        self.joints_at = 0.0
        self.latest_status = None
        self.status_at = 0.0
        self.latest_camera = None
        self.camera_at = 0.0
        self.publisher = self.create_publisher(
            JointState, '/joint_ctrl_single', 1)
        self.create_subscription(
            JointState, '/joint_states_single', self.joints_cb, 10)
        self.create_subscription(
            PiperStatusMsg, '/arm_status', self.status_cb, 10)
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_cb, 10)
        self.enable_client = self.create_client(Enable, '/enable_srv')

    def joints_cb(self, message):
        values = list(message.position[:6])
        if len(values) == 6 and all(math.isfinite(value) for value in values):
            with self.lock:
                self.latest_joints = values
                self.joints_at = time.monotonic()

    def status_cb(self, message):
        with self.lock:
            self.latest_status = message
            self.status_at = time.monotonic()

    def camera_cb(self, message):
        with self.lock:
            self.latest_camera = message
            self.camera_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            return (
                None if self.latest_joints is None else list(self.latest_joints),
                float(self.joints_at), self.latest_status,
                float(self.status_at), self.latest_camera,
                float(self.camera_at))

    def motion_authority(self):
        joints, joints_at, status, status_at, _camera, _camera_at = self.snapshot()
        now = time.monotonic()
        if joints is None or now - joints_at > 0.5:
            return False, '/joint_states_single is unavailable or stale'
        if status is None or now - status_at > 0.5:
            return False, '/arm_status is unavailable or stale'
        if not bool(status.motor_feedback_valid):
            return False, 'motor feedback is invalid'
        enabled = [bool(getattr(status, 'motor_%d_driver_enabled' % index))
                   for index in range(1, 7)]
        if not all(enabled):
            return False, 'all six motors are not enabled'
        if status.motor_faults:
            return False, 'motor faults are present: %s' % ', '.join(status.motor_faults)
        if int(status.ctrl_mode) != 1:
            return False, 'controller is not in CAN command mode'
        return True, 'ready'

    def preflight(self, camera_profile):
        ready, reason = self.motion_authority()
        if not ready:
            return ready, reason
        _joints, _joints_at, _status, _status_at, camera, camera_at = self.snapshot()
        if camera is None or time.monotonic() - camera_at > 1.0:
            return False, 'camera CameraInfo is unavailable or stale'
        expected = (int(camera_profile['width']), int(camera_profile['height']))
        actual = (int(camera.width), int(camera.height))
        if actual != expected:
            return False, 'camera profile is %dx%d, expected %dx%d' % (
                actual[0], actual[1], expected[0], expected[1])
        others = [
            '%s/%s' % (info.node_namespace.strip('/'), info.node_name)
            for info in self.get_publishers_info_by_topic('/joint_ctrl_single')
            if info.node_name != self.get_name()
        ]
        if others:
            return False, 'another joint-command publisher is active: %s' % ', '.join(others)
        return True, 'ready'

    def wait_for_preflight(self, profile, timeout_sec=5.0):
        deadline = time.monotonic() + timeout_sec
        reason = 'preflight did not run'
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            ready, reason = self.preflight(profile)
            if ready:
                return True, reason
        return False, reason

    def publish_target(self, positions):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'fixed_hand_eye_calibration'
        message.name = list(JOINT_NAMES)
        message.position = [float(value) for value in positions]
        message.velocity = [0.0] * 6 + [self.speed_percent]
        self.publisher.publish(message)

    def wait_for_target(self, target, timeout_sec, tolerance_rad=0.025,
                        stable_sec=1.0):
        deadline = time.monotonic() + timeout_sec
        stable_since = None
        last_error = math.inf
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            ready, reason = self.motion_authority()
            if not ready:
                raise RuntimeError(reason)
            joints, received, *_rest = self.snapshot()
            if joints is None or time.monotonic() - received > 0.5:
                stable_since = None
                continue
            last_error = max(abs(a - b) for a, b in zip(joints, target))
            if last_error <= tolerance_rad:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= stable_sec:
                    return True, last_error
            else:
                stable_since = None
        return False, last_error

    def hold_measured(self):
        ready, _reason = self.motion_authority()
        joints, received, *_rest = self.snapshot()
        if ready and joints is not None and time.monotonic() - received <= 0.5:
            self.publish_target(joints)

    def request_disable(self, timeout_sec=20.0):
        if not self.enable_client.wait_for_service(timeout_sec=2.0):
            return False, '/enable_srv is unavailable'
        request = Enable.Request()
        request.enable_request = False
        future = self.enable_client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None \
                or not bool(future.result().enable_response):
            return False, 'driver did not prove disable'
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            _joints, _ja, status, status_at, _camera, _ca = self.snapshot()
            if status is not None and time.monotonic() - status_at <= 0.5:
                enabled = [bool(getattr(status, 'motor_%d_driver_enabled' % index))
                           for index in range(1, 7)]
                if not any(enabled):
                    return True, 'all six motors disabled'
        return False, 'all-six disabled feedback was not observed'

    def home_then_disable(self, profile, timeout_sec):
        targets = calibration_home_targets(profile)
        labels = ('PRE_HOME', 'ROUGH_HOME', 'STORAGE_WRIST')
        for label, target in zip(labels, targets):
            ready, reason = self.motion_authority()
            if not ready:
                return False, '%s blocked: %s' % (label, reason)
            self.publish_target(target)
            try:
                reached, error = self.wait_for_target(target, timeout_sec)
            except RuntimeError as error:
                return False, '%s lost authority: %s' % (label, error)
            if not reached:
                self.hold_measured()
                return False, '%s did not reach and settle' % label
            print('%s settled; maximum error %.5f rad' % (label, error))
        self.publish_target(targets[-1])
        return self.request_disable()


def capture_sample(script_root, output_root, group, timeout_sec, board):
    command = [
        sys.executable, str(script_root / 'capture_hand_eye_sample.py'),
        '--output-root', str(output_root / group),
        '--timeout', str(timeout_sec),
        '--squares-x', str(board['squares_x']),
        '--squares-y', str(board['squares_y']),
        '--square-length-m', str(board['square_length_m']),
        '--marker-length-m', str(board['marker_length_m']),
        '--dictionary', str(board['dictionary']),
    ]
    return subprocess.run(command, check=False).returncode == 0


def main():
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', type=Path, default=(
        script_root / 'calibration/hand_eye/fixed_calibration_poses.yaml'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--home-profile', type=Path,
                        default=script_root.parent / 'piper_home_pose.json')
    parser.add_argument('--speed-percent', type=float, default=5.0)
    parser.add_argument('--move-timeout-sec', type=float, default=90.0)
    parser.add_argument('--capture-timeout-sec', type=float, default=20.0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirmation', default='')
    args = parser.parse_args()
    try:
        pose_data = load_pose_file(args.poses)
        home_profile = load_home_pose(args.home_profile)
        calibration_home_targets(home_profile)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if not 1.0 <= args.speed_percent <= 10.0:
        parser.error('speed must be between 1 and 10 percent')
    if args.output_root is None:
        args.output_root = script_root / 'calibration/hand_eye' / (
            'session_' + datetime.now().strftime('%Y%m%d_%H%M%S_fixed'))
    print('Loaded %d fixed poses from %s' % (len(pose_data['poses']), args.poses))
    for index, pose in enumerate(pose_data['poses'], 1):
        print('%02d %-24s %-10s %s' % (
            index, pose['id'], pose.get('group', 'TRANSIT'),
            pose['positions_rad']))
    if not args.execute:
        print('Dry run only; no ROS publisher and no motion were started.')
        return 0
    if args.confirmation != EXACT_CONFIRMATION:
        parser.error('execute mode requires --confirmation "%s"' % EXACT_CONFIRMATION)
    for group in ('fitting', 'validation'):
        (args.output_root / group).mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FixedPoseRunner(args.speed_percent)
    result = 0
    reason = ''
    try:
        ready, reason = node.wait_for_preflight(
            pose_data['recommended_calibration_profile'])
        if not ready:
            raise RuntimeError('calibration preflight failed: %s' % reason)
        print('Preflight passed. This runner is the sole joint-command owner.')
        for index, pose in enumerate(pose_data['poses'], 1):
            answer = input('\nPose %d/%d %s. Enter=move, s=skip, q=quit: ' % (
                index, len(pose_data['poses']), pose['id'])).strip().lower()
            if answer == 'q':
                raise KeyboardInterrupt('operator quit')
            if answer == 's':
                continue
            ready, reason = node.wait_for_preflight(
                pose_data['recommended_calibration_profile'])
            if not ready:
                raise RuntimeError('pre-command preflight failed: %s' % reason)
            target = [float(value) for value in pose['positions_rad']]
            node.publish_target(target)
            reached, error = node.wait_for_target(target, args.move_timeout_sec)
            if not reached:
                node.hold_measured()
                raise RuntimeError('%s did not reach and settle' % pose['id'])
            print('%s settled; maximum error %.5f rad' % (pose['id'], error))
            if not pose.get('capture', True):
                print('Transit only; no capture.')
                continue
            while not capture_sample(
                    script_root, args.output_root, pose['group'],
                    args.capture_timeout_sec, pose_data['board']):
                answer = input(
                    'Capture rejected. Enter=retry, s=skip, q=quit: '
                ).strip().lower()
                if answer == 's':
                    break
                if answer == 'q':
                    raise KeyboardInterrupt('operator quit')
            else:
                print('%s capture accepted' % pose['id'])
    except KeyboardInterrupt as error:
        result = 1
        reason = str(error) or 'operator interrupted'
        node.hold_measured()
    except RuntimeError as error:
        result = 2
        reason = str(error)
        node.hold_measured()
        print(reason, file=sys.stderr)
    finally:
        ready, authority_reason = node.motion_authority()
        if ready:
            print('Terminal cleanup: PRE_HOME -> ROUGH_HOME -> '
                  'STORAGE_WRIST -> disable.')
            safe, cleanup_reason = node.home_then_disable(
                home_profile, args.move_timeout_sec)
            if not safe:
                result = 3
                reason = cleanup_reason
                print('HOME/DISABLE NOT PROVED: %s' % cleanup_reason,
                      file=sys.stderr)
        else:
            result = 3
            reason = authority_reason
            print('Motion authority unavailable; no blind home/disable: %s' %
                  authority_reason, file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
    if result:
        print('Calibration ended without deployment: %s' % reason,
              file=sys.stderr)
        return result
    print('Calibration captures complete in %s. Nothing was deployed.' %
          args.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
