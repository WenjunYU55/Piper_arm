#!/usr/bin/env python3
"""Compatibility import for the shared temporal tracking core."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ROS_PACKAGE_SOURCE = REPO_ROOT / "piper_ros_foxy" / "src" / "piper_mobile_manipulation"
if str(ROS_PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE_SOURCE))

from piper_mobile_manipulation.utils.temporal_tracking import *  # noqa: F401,F403,E402
