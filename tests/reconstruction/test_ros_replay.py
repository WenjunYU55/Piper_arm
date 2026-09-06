"""Exercise the real recorder through DDS, synchronization and stamped TF."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_public_ros_replay(tmp_path):
    """Keep synthetic mission events isolated, with no robot publisher."""
    pytest.importorskip("rclpy")
    # The child process imports generated ROS messages from the sourced overlay.
    # Pure unit tests can have already loaded the source-only package here.

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "replay"
    environment = dict(
        os.environ,
        ROS_DOMAIN_ID="179",
        ROS_LOCALHOST_ONLY="1",
        PIPER_MISSION_ENABLE_REAL_MOTION="0",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/bundlesdf/replay.py"),
            "--output",
            str(output),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout
    report = json.loads((output / "replay_report.json").read_text())
    assert report["evidence_class"] == "ROS_SYNTHETIC_REPLAY_NOT_PHYSICAL"
    assert [s["accepted"] for s in report["sessions"]] == [6, 1, 6, 1, 6, 1]
    assert all(
        s["mission_capture_count"] == 2 for s in report["sessions"] if s["complete"]
    )
