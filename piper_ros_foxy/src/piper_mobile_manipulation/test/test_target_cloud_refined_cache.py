from piper_mobile_manipulation.target_cloud_node import (
    closest_cached_frame,
    status_image_stamp,
)


def test_correlated_queued_status_exposes_selected_image_stamp():
    payload = {
        'state': 'queued',
        'request_id': 'cloud-1',
        'image_stamp': {'sec': 12, 'nanosec': 250_000_000},
    }
    assert status_image_stamp(payload, 'cloud-1') == 12.25


def test_status_image_stamp_rejects_wrong_request_and_invalid_stamp():
    payload = {
        'state': 'queued',
        'request_id': 'old',
        'image_stamp': {'sec': 12, 'nanosec': 250_000_000},
    }
    assert status_image_stamp(payload, 'new') is None
    payload['request_id'] = 'new'
    payload['image_stamp']['nanosec'] = 1_000_000_000
    assert status_image_stamp(payload, 'new') is None


def test_pinned_frame_remains_selectable_after_rolling_cache_eviction():
    pinned = (12.24, 'pinned-color', 'pinned-depth', 'pinned-info')
    rolling = [
        (18.0, 'new-color-1', 'new-depth-1', 'new-info-1'),
        (18.1, 'new-color-2', 'new-depth-2', 'new-info-2'),
    ]
    match = closest_cached_frame(rolling + [pinned], 12.25)
    assert match is pinned
