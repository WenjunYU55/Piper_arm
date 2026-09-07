"""Exercise the real recorder through DDS, synchronization and stamped TF."""

import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

import pytest


@pytest.mark.parametrize('distorted', [False, True])
def test_public_ros_replay(tmp_path, distorted):
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
        ] + (["--distorted"] if distorted else []),
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


def test_repeated_shutdown_signals_are_orderly(tmp_path):
    """Supervisor signals must not inject exceptions during TF conversion or joins."""
    pytest.importorskip('rclpy')
    root = Path(__file__).resolve().parents[2]
    script = root / 'piper_ros_foxy/src/piper_mobile_manipulation' / (
        'piper_mobile_manipulation/reconstruction_recorder_node.py')
    process = subprocess.Popen(
        [sys.executable, str(script), '--ros-args', '-p', 'output_root:=' + str(tmp_path)],
        env=dict(os.environ, ROS_DOMAIN_ID='178', ROS_LOCALHOST_ONLY='1'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(2)
        assert process.poll() is None
        process.send_signal(signal.SIGINT)
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=15)
        assert process.returncode == 0, output
        assert 'Traceback' not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
