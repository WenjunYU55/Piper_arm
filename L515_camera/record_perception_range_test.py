#!/usr/bin/env python3
"""Interactively record GroundingDINO, SAM2, and L515 range evidence."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from perception_range_report import analyse_job, finite_float, refresh_reports


class RangeTestNode(Node):
    """Own exact request/status correlation for a command-free range test."""

    TERMINAL_STATES = {'published', 'worker_result_rejected', 'request_rejected'}

    def __init__(self):
        super().__init__('piper_perception_range_test')
        self.publisher = self.create_publisher(
            String, '/piper/heavy_refresh_request', 10)
        self.create_subscription(
            String, '/piper/heavy_refresh_status', self.status_callback, 10)
        self.condition = threading.Condition()
        self.pending_request_id = ''
        self.terminal_status = None

    def status_callback(self, message):
        """Accept terminal status only for this exact request."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.condition:
            if str(payload.get('request_id', '')) != self.pending_request_id:
                return
            if str(payload.get('state', '')) in self.TERMINAL_STATES:
                self.terminal_status = payload
                self.condition.notify_all()

    def request(self, reference_distance_m, timeout_sec):
        """Request one new heavy frame and wait for its exact result."""
        request_id = 'range-test-%s-%s' % (
            datetime.now().strftime('%H%M%S'), uuid.uuid4().hex[:8])
        now = self.get_clock().now().to_msg()
        payload = {
            'request_id': request_id,
            'reason': 'camera_range_test',
            'min_image_stamp': {
                'sec': int(now.sec),
                'nanosec': int(now.nanosec),
            },
            'reference_surface_distance_m': float(reference_distance_m),
            'dry_run': True,
            'real_arm_motion': False,
        }
        with self.condition:
            self.pending_request_id = request_id
            self.terminal_status = None
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.publisher.publish(message)
        deadline = time.monotonic() + float(timeout_sec)
        with self.condition:
            while self.terminal_status is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self.condition.wait(timeout=min(remaining, 0.5))
            status = self.terminal_status
        if status is None:
            status = {
                'state': 'timeout',
                'request_id': request_id,
                'error': 'GroundingDINO/SAM2 exceeded %.1f seconds'
                % timeout_sec,
            }
        return request_id, status


def output_directory(root):
    """Create one human-readable range-test session directory."""
    name = datetime.now().strftime('PerceptionRange - %H-%M - %d-%m-%Y')
    path = root / name
    suffix = 2
    while path.exists():
        path = root / ('%s - %d' % (name, suffix))
        suffix += 1
    path.mkdir(parents=True)
    return path


def main():
    """Run a stationary-distance characterization session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root', type=Path,
        default=Path(os.environ.get(
            'PIPER_ARM_ROOT', '/home/prl/Piper_arm'))
        / 'datasets' / 'perception_range_tests')
    parser.add_argument('--timeout-sec', type=float, default=90.0)
    args = parser.parse_args()
    directory = output_directory(args.output_root.expanduser().resolve())
    rows = []

    rclpy.init()
    node = RangeTestNode()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    print('Output directory: %s' % directory, flush=True)
    print(
        'Measure from the L515 optical face to the visible target surface.',
        flush=True)
    print(
        'Hold the target still, enter metres, and repeat each distance 3 times.',
        flush=True)
    print('Enter done to finish.', flush=True)
    try:
        while True:
            text = input(
                '\nReference surface distance in metres [or done]: ').strip()
            if text.lower() in ('done', 'quit', 'q', 'exit'):
                break
            try:
                distance = float(text)
            except ValueError:
                print('Enter a numeric distance such as 1.25, or done.')
                continue
            if not math.isfinite(distance) or not 0.15 <= distance <= 9.0:
                print('Distance must be finite and between 0.15 m and 9.0 m.')
                continue
            input('Hold the target stationary, then press Enter to capture...')
            request_id, status = node.request(distance, args.timeout_sec)
            row = analyse_job(
                Path('/tmp/piper_heavy_refresh'),
                request_id,
                distance,
                status,
            )
            rows.append(row)
            csv_path, xlsx_path, html_path, reason = refresh_reports(
                directory, rows)
            print(
                'Result: detected=%s measured=%.3f m error=%.3f m usable=%s'
                % (
                    row['groundingdino_detected'],
                    finite_float(row['selected_depth_m'], 0.0),
                    finite_float(row['absolute_depth_error_m'], 0.0),
                    row['usable'],
                ))
            if row['failure_reason']:
                print('Reason: %s' % row['failure_reason'])
            print('CSV: %s' % csv_path)
            if xlsx_path is not None:
                print('Excel: %s' % xlsx_path)
            elif reason:
                print('Excel conversion warning: %s' % reason)
            print('Plot: %s' % html_path)
    except (EOFError, KeyboardInterrupt):
        print('\nStopping range test.')
    finally:
        if rows:
            refresh_reports(directory, rows)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
