#!/usr/bin/env python3
"""Render a textured cube into a native RGB-D session; this is not physical evidence."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'piper_ros_foxy/src/piper_mobile_manipulation'))
from piper_mobile_manipulation.reconstruction_recording.core import (  # noqa: E402
    Frame, RecorderConfig, RecordingSession,
)


def render(yaw_deg, height=480, width=640):
    """Intersect pinhole rays with a fixed textured 14 cm cube in base coordinates."""
    target = np.array([0.0, 0.0, 0.5])
    yaw = np.deg2rad(yaw_deg)
    position = target + np.array([0.42 * np.sin(yaw), -0.42 * np.cos(yaw), 0.15])
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(forward, right), forward))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, position
    K = np.array([[600., 0., width / 2], [0., 600., height / 2], [0., 0., 1.]])
    y, x = np.indices((height, width))
    rays = np.stack(((x - width / 2) / 600, (y - height / 2) / 600, np.ones_like(x)), -1)
    rays = rays @ rotation.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (target - 0.07 - position) / rays
        t1 = (target + 0.07 - position) / rays
    near = np.minimum(t0, t1).max(axis=-1)
    far = np.maximum(t0, t1).min(axis=-1)
    valid = (far >= near) & (near > 0) & np.isfinite(near)
    points = position + near[..., None] * rays
    # Fixed object-space texture: the same surface carries the same pattern in every view.
    cell = np.floor((points - target + 0.071) / 0.004).astype(np.int64)
    code = (cell[..., 0] * 73856093) ^ (cell[..., 1] * 19349663) ^ (cell[..., 2] * 83492791)
    rgb = np.full((height, width, 3), 35, np.uint8)
    for channel in range(3):
        rgb[..., channel][valid] = (50 + ((code[valid] >> (channel * 5)) % 206)).astype(np.uint8)
    depth = np.where(valid, np.rint(near * 1000), 0).astype(np.uint16)
    return rgb, depth, valid.astype(np.uint8) * 255, K, matrix, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or args.frames < 6:
        parser.error('choose a new output root and at least six frames')
    config = RecorderConfig()
    recording = RecordingSession(args.output, 'SYNTHETIC_CUBE_NOT_PHYSICAL', [0, 0, .5], config)
    try:
        for index, yaw in enumerate(np.linspace(-35, 35, args.frames)):
            rgb, depth, mask, K, transform, target = render(yaw)
            stamp = 1_000_000_000 + index * 150_000_000
            source = 'first_capture' if index == 0 else 'moving'
            view = 1 if index == 0 else 0
            if index == args.frames - 1:
                source, view = 'stationary_capture', 2
            frame = Frame(rgb, depth, mask, K, transform, stamp, stamp, stamp,
                          0., '16UC1', 'base_link', 'camera_color_optical_frame',
                          recording.target_id, stamp, source=source, accepted_view=view)
            if not recording.submit(frame):
                raise RuntimeError('synthetic admission rejected: ' + str(recording.rejected))
            time.sleep(.03)
        recording.mission_capture_count = 2
        recording.close('complete')
        if recording.error:
            raise RuntimeError(recording.error)
        print(recording.final_path)
        (args.output / 'synthetic_evidence.json').write_text(json.dumps(dict(
            evidence_class='ANALYTIC_RENDER_NOT_PHYSICAL', frames=args.frames,
            cube_side_m=.14, session=str(recording.final_path)), indent=2))
    finally:
        recording.close()


if __name__ == '__main__':
    main()
