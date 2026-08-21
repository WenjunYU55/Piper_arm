"""Characterize single-flight heavy-refresh request retention."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.heavy_refresh_bridge_node import (
    HeavyRefreshBridgeNode,
    masked_depth_range_diagnostic,
)
from piper_mobile_manipulation.scan_execution_modes import (
    heavy_refresh_status_action,
)


def stamp(sec, nanosec=0):
    """Build the ROS-stamp-shaped value used by the pure bridge seam."""
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def test_busy_worker_retains_one_correlated_request():
    statuses = []
    request = {
        'request_id': 'capture-refresh-1',
        'reason': 'capture visual rejection',
        'min_image_stamp': {'sec': 19, 'nanosec': 0},
    }
    bridge = SimpleNamespace(
        pending_request=None,
        latest_color=object(),
        latest_color_msg=SimpleNamespace(header=SimpleNamespace(
            stamp=stamp(20))),
        latest_color_time=time.monotonic(),
        worker_busy=lambda: True,
        get_parameter=lambda _name: SimpleNamespace(value=1.0),
        publish_status=lambda state, **fields: statuses.append(
            (state, fields)),
        enqueue_request=lambda _request: None,
    )

    HeavyRefreshBridgeNode.request_cb(
        bridge, SimpleNamespace(data=json.dumps(request)))

    assert bridge.pending_request == request
    assert statuses == [(
        'waiting_for_worker', {
            'request_id': 'capture-refresh-1',
            'reason': 'capture visual rejection',
        })]


def test_retained_request_queues_once_after_worker_becomes_idle():
    enqueued = []
    request = {
        'request_id': 'capture-refresh-2',
        'min_image_stamp': {'sec': 19, 'nanosec': 0},
    }
    busy = {'value': True}
    bridge = SimpleNamespace(
        pending_request=request,
        latest_color=object(),
        latest_color_msg=SimpleNamespace(header=SimpleNamespace(
            stamp=stamp(20))),
        latest_color_time=time.monotonic(),
        worker_busy=lambda: busy['value'],
        get_parameter=lambda _name: SimpleNamespace(value=1.0),
        enqueue_request=lambda value: enqueued.append(value),
    )

    assert not HeavyRefreshBridgeNode.try_enqueue_pending_request(bridge)
    assert bridge.pending_request == request
    assert enqueued == []

    busy['value'] = False
    assert HeavyRefreshBridgeNode.try_enqueue_pending_request(bridge)
    assert bridge.pending_request is None
    assert enqueued == [request]

    assert not HeavyRefreshBridgeNode.try_enqueue_pending_request(bridge)
    assert enqueued == [request]


def test_waiting_for_worker_status_is_nonterminal():
    action, reason, image_stamp_ns = heavy_refresh_status_action(
        {
            'state': 'waiting_for_worker',
            'request_id': 'capture-refresh-3',
        },
        'capture-refresh-3',
        10_000_000_000,
    )

    assert action == 'waiting_for_worker'
    assert reason == 'heavy worker is busy'
    assert image_stamp_ns is None


def test_only_correlated_masked_depth_can_prove_target_is_too_far():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    depth_mm = np.full((8, 8), 400, dtype=np.uint16)
    depth_mm[2:6, 2:6] = 1300

    result = masked_depth_range_diagnostic(
        mask, depth_mm, minimum_pixels=8, minimum_ratio=0.5,
        maximum_depth_m=1.20)

    assert result['target_depth_status'] == 'TOO_FAR'
    assert result['target_depth_nearest_m'] == pytest.approx(1.30)


def test_mixed_or_insufficient_masked_depth_does_not_claim_too_far():
    mask = np.ones((4, 4), dtype=np.uint8)
    mixed = np.full((4, 4), 1300, dtype=np.uint16)
    mixed[0, 0] = 1000
    invalid = np.zeros((4, 4), dtype=np.uint16)

    assert masked_depth_range_diagnostic(
        mask, mixed, 4, 0.5, 1.20)['target_depth_status'] == 'VALID'
    assert masked_depth_range_diagnostic(
        mask, invalid, 4, 0.5, 1.20)['target_depth_status'] == (
            'INSUFFICIENT')
