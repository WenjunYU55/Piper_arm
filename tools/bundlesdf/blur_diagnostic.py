#!/usr/bin/env python3
"""Measure target-interior Laplacian variance; compare held and moving images."""

import argparse
import json
from pathlib import Path
import sys

import cv2

sys.path.append(
    str(
        Path(__file__).resolve().parents[2]
        / "piper_ros_foxy/src/piper_mobile_manipulation"
    )
)

from piper_mobile_manipulation.reconstruction_recording.core import (
    mask_metrics,
)  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("rgb", type=Path)
parser.add_argument("mask", type=Path)
args = parser.parse_args()
rgb = cv2.imread(str(args.rgb))
mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
if rgb is None or mask is None or mask.shape != rgb.shape[:2]:
    parser.error("RGB and binary mask must exist and have matching dimensions")
box, margin, score = mask_metrics(mask, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
print(
    json.dumps(
        dict(target_bbox_xyxy=box, edge_margin_px=margin, blur_laplacian_variance=score)
    )
)
