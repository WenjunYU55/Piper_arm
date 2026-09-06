"""Verify shared optical geometry and frozen configuration across actual consumers."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.execution.motion import camera_target_path_reasons
from piper_mobile_manipulation.visibility import (
    frozen_visibility_parameters,
    DEFAULT_VISIBILITY_ANGLE_DEG,
)
from piper_tesseract_foxy.worker import camera_transform_path_rejection
from motion_planning.curobo.worker import camera_path_visibility_rejection


@pytest.mark.parametrize("limit", [5.0, 10.0, 20.0])
@pytest.mark.parametrize(
    "offset,accept", [(-0.0001, True), (0.0, True), (0.0001, False)]
)
def test_planner_worker_executor_cone_agreement(limit, offset, accept):
    angle = math.radians(limit + offset)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)],
    ]
    target = [0, 0, 0.5]
    executor = camera_target_path_reasons(
        SimpleNamespace(camera_transform=lambda _: matrix),
        [[0] * 6],
        target,
        maximum_boresight_deg=limit,
    )
    tesseract = camera_transform_path_rejection(
        [matrix], target, maximum_boresight_deg=limit
    )
    curobo = camera_path_visibility_rejection(
        [matrix[:3, 3]], [matrix[:3, 2]], target, maximum_boresight_deg=limit
    )
    assert (not executor) == (not tesseract) == (not curobo) == accept


def test_default_is_shared_and_live_partial_updates_reject():
    assert DEFAULT_VISIBILITY_ANGLE_DEG == 20
    result = frozen_visibility_parameters(
        [SimpleNamespace(name="target_visibility_angle_tolerance_deg", value=10.0)]
    )
    assert not result.successful
    assert "restart" in result.reason


def test_request_angle_bound_to_trajectory_digest():
    from motion_planning.curobo.worker import Worker
    from piper_tesseract_foxy.protocol.contract import trajectory_digest

    base = dict(
        request_sha256="a" * 64,
        plan_kind="MULTIVIEW_SCAN",
        target_provenance="test",
        model={},
        calibration={"hand_eye_sha256": "b" * 64},
        limits={},
        planning=dict(
            effective_speed_percent=5.0,
            command_rate_hz=20.0,
            timing_policy="timed_stream_v1",
            target_visibility_angle_tolerance_deg=10.0,
        ),
    )
    binding = Worker.binding(base)
    assert binding["execution"]["target_visibility_angle_tolerance_deg"] == 10.0
    first = trajectory_digest([], binding)
    base["planning"]["target_visibility_angle_tolerance_deg"] = 5.0
    assert trajectory_digest([], Worker.binding(base)) != first
