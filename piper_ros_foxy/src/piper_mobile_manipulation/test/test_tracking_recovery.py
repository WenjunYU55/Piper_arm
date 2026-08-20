from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from std_msgs.msg import Header

from piper_mobile_manipulation.occlusion_checker_node import OcclusionCheckerNode
from piper_mobile_manipulation.obstacle_instance_3d_node import (
    ObstacleInstance3DNode,
    tf_listener_recovery_due,
)
from piper_mobile_manipulation.safe_servo_node import SafeServoNode
from piper_mobile_manipulation.sam2_live_bridge_node import Sam2LiveBridgeNode
from piper_mobile_manipulation.target_tracker_node import TargetTrackerNode


class FakeFilter:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class FakeLogger:
    def info(self, _message):
        pass


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_live_target_only_scene_publishes_without_waiting_for_tf():
    publisher = FakePublisher()

    class FakeBridge:
        @staticmethod
        def imgmsg_to_cv2(message, _encoding):
            return message.pixels

    class TfMustNotBeUsed:
        @staticmethod
        def lookup_transform(*_args, **_kwargs):
            raise AssertionError('target-only clear scene does not require TF')

    header = Header()
    header.stamp.sec = 12
    header.stamp.nanosec = 34
    header.frame_id = 'camera_color_optical_frame'
    ids_msg = SimpleNamespace(
        header=header,
        pixels=np.ones((4, 5), dtype=np.uint16),
    )
    depth_msg = SimpleNamespace(
        header=header,
        encoding='16UC1',
        pixels=np.full((4, 5), 400, dtype=np.uint16),
    )
    info_msg = SimpleNamespace(header=header, k=[1.0] * 9)
    node = SimpleNamespace(
        bridge=FakeBridge(),
        publisher=publisher,
        tf_buffer=TfMustNotBeUsed(),
        get_parameter=lambda name: SimpleNamespace(value={
            'base_frame': 'base_link',
        }[name]),
    )

    ObstacleInstance3DNode.process(
        node, ids_msg, depth_msg, info_msg,
        {'objects': [{'object_id': 1, 'role': 'target'}]},
    )

    assert len(publisher.messages) == 1
    scene = publisher.messages[0]
    assert scene.header is header
    assert not scene.scene_blocked
    assert list(scene.instances) == []
    assert scene.blocking_reason == 'clear:live_target_only_frame'


def test_exact_empty_rough_acquisition_emits_correlated_clear_scene():
    publisher = FakePublisher()
    node = SimpleNamespace(
        publisher=publisher,
        get_parameter=lambda _name: SimpleNamespace(value='base_link'),
    )
    status = SimpleNamespace(data=(
        '{"state":"worker_result_rejected",'
        '"worker_status":"target_mask_missing",'
        '"reason":"rough_acquisition_viewpoint",'
        '"obstacle_count":0,'
        '"unsafe_obstacle_count":0,'
        '"image_stamp":{"sec":12,"nanosec":34}}'))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert len(publisher.messages) == 1
    scene = publisher.messages[0]
    assert scene.header.stamp.sec == 12
    assert scene.header.stamp.nanosec == 34
    assert scene.header.frame_id == 'base_link'
    assert not scene.scene_blocked
    assert list(scene.instances) == []


def test_obstacle_only_result_is_not_misreported_as_clear():
    publisher = FakePublisher()
    node = SimpleNamespace(
        publisher=publisher,
        get_parameter=lambda _name: SimpleNamespace(value='base_link'),
    )
    status = SimpleNamespace(data=(
        '{"state":"worker_result_rejected",'
        '"worker_status":"target_mask_missing",'
        '"reason":"rough_acquisition_viewpoint",'
        '"obstacle_count":1,'
        '"unsafe_obstacle_count":1,'
        '"image_stamp":{"sec":12,"nanosec":34}}'))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert publisher.messages == []


def test_target_missing_scan_reacquisition_never_claims_clear_scene():
    publisher = FakePublisher()
    node = SimpleNamespace(
        publisher=publisher,
        get_parameter=lambda _name: SimpleNamespace(value='base_link'),
    )
    status = SimpleNamespace(data=(
        '{"state":"worker_result_rejected",'
        '"worker_status":"target_mask_missing",'
        '"reason":"sam2_target_lost_retry",'
        '"obstacle_count":0,'
        '"unsafe_obstacle_count":0,'
        '"image_stamp":{"sec":15,"nanosec":61}}'))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert publisher.messages == []


def test_published_target_only_probe_emits_exact_clear_scene():
    publisher = FakePublisher()
    node = SimpleNamespace(
        publisher=publisher,
        get_parameter=lambda _name: SimpleNamespace(value='base_link'),
    )
    status = SimpleNamespace(data=(
        '{"state":"published","reason":"workflow_occlusion_probe",'
        '"obstacle_count":0,"unsafe_obstacle_count":0,'
        '"image_stamp":{"sec":15,"nanosec":61}}'))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert len(publisher.messages) == 1
    scene = publisher.messages[0]
    assert scene.header.stamp.sec == 15
    assert scene.header.stamp.nanosec == 61
    assert not scene.scene_blocked
    assert list(scene.instances) == []


def test_published_obstacle_probe_emits_blocking_exact_scene():
    publisher = FakePublisher()
    node = SimpleNamespace(
        publisher=publisher,
        get_parameter=lambda _name: SimpleNamespace(value='base_link'),
    )
    status = SimpleNamespace(data=(
        '{"state":"published","reason":"workflow_occlusion_probe",'
        '"obstacle_count":1,"unsafe_obstacle_count":1,'
        '"obstacle_labels":["unknown depth foreground"],'
        '"obstacle_confidences":[1.0],'
        '"tracked_objects":[{"object_id":2,"role":"obstacle",'
        '"label":"unknown depth foreground","confidence":1.0,'
        '"unsafe":true}],'
        '"image_stamp":{"sec":15,"nanosec":61}}'))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert len(publisher.messages) == 1
    scene = publisher.messages[0]
    assert scene.scene_blocked
    assert not scene.instances
    assert 'semantic_probe_projection_failed' in scene.blocking_reason


def test_published_obstacle_probe_projects_archived_masks(tmp_path):
    job_id = 'projection-regression'
    request_dir = tmp_path / 'archive' / job_id
    response_dir = tmp_path / 'consumed' / job_id
    request_dir.mkdir(parents=True)
    response_dir.mkdir(parents=True)
    depth = np.full((20, 20), 400, dtype=np.uint16)
    np.save(str(request_dir / 'depth.npy'), depth)
    with (request_dir / 'request.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump({
            'depth_encoding': '16UC1',
            'camera_matrix': [100.0, 0.0, 10.0, 0.0, 100.0, 10.0,
                              0.0, 0.0, 1.0],
            'frame_id': 'camera_color_optical_frame',
        }, stream)
    target_mask = np.zeros((20, 20), dtype=np.uint8)
    target_mask[7:17, 7:17] = 255
    obstacle_mask = np.zeros((20, 20), dtype=np.uint8)
    obstacle_mask[2:10, 7:17] = 255
    assert cv2.imwrite(str(response_dir / 'target_mask.png'), target_mask)
    assert cv2.imwrite(str(response_dir / 'obstacle_2.png'), obstacle_mask)

    transform = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    ))
    publisher = FakePublisher()
    parameters = {
        'base_frame': 'base_link',
        'heavy_spool_dir': str(tmp_path),
        'transform_timeout_sec': 0.20,
        'movable_whitelist': ['stick'],
        'depth_min_m': 0.25,
        'depth_max_m': 1.20,
        'min_valid_depth_pixels': 20,
        'min_valid_depth_ratio': 0.40,
        'mask_erode_px': 0,
        'bounds_low_percentile': 2.0,
        'bounds_high_percentile': 98.0,
    }
    node = SimpleNamespace(
        publisher=publisher,
        tf_buffer=SimpleNamespace(lookup_transform=lambda *_args, **_kwargs: transform),
        heavy_tracks={},
        track_generation='test-generation',
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        config=lambda: {
            name: parameters[name] for name in (
                'depth_min_m', 'depth_max_m', 'min_valid_depth_pixels',
                'min_valid_depth_ratio', 'mask_erode_px',
                'bounds_low_percentile', 'bounds_high_percentile')
        },
        set_point=ObstacleInstance3DNode.set_point,
        set_footprint=ObstacleInstance3DNode.set_footprint,
    )
    node.match_heavy_track = lambda label, center: (
        ObstacleInstance3DNode.match_heavy_track(node, label, center))
    status = SimpleNamespace(data=json.dumps({
        'state': 'published',
        'reason': 'workflow_occlusion_probe',
        'job_id': job_id,
        'obstacle_count': 1,
        'unsafe_obstacle_count': 0,
        'tracked_objects': [{
            'object_id': 2,
            'role': 'obstacle',
            'label': 'stick',
            'confidence': 0.9,
            'unsafe': False,
            'mask_file': 'obstacle_2.png',
        }],
        'image_stamp': {'sec': 15, 'nanosec': 61},
    }))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert len(publisher.messages) == 1
    scene = publisher.messages[0]
    assert len(scene.instances) == 1
    assert scene.instances[0].valid
    assert scene.instances[0].semantic_label == 'stick'
    assert scene.instances[0].track_id.startswith('heavy-test-generation-stick-')
    assert scene.blocking_reason == 'clear'


def test_archived_mask_read_survives_response_to_consumed_atomic_move(
        tmp_path, monkeypatch):
    job_id = 'atomic-move-regression'
    response_dir = tmp_path / 'responses' / job_id
    consumed_dir = tmp_path / 'consumed' / job_id
    response_dir.mkdir(parents=True)
    consumed_dir.parent.mkdir(parents=True)
    mask_path = response_dir / 'target_mask.png'
    mask = np.full((12, 16), 255, dtype=np.uint8)
    assert cv2.imwrite(str(mask_path), mask)
    original_imread = cv2.imread
    moved = {'done': False}

    def moving_imread(path, mode):
        if str(path) == str(mask_path.resolve()) and not moved['done']:
            response_dir.rename(consumed_dir)
            moved['done'] = True
        return original_imread(path, mode)

    monkeypatch.setattr(cv2, 'imread', moving_imread)
    node = SimpleNamespace(get_parameter=lambda _name: SimpleNamespace(
        value=str(tmp_path)))

    recovered = ObstacleInstance3DNode.heavy_response_mask(
        node, job_id, 'target_mask.png')

    assert moved['done']
    assert recovered is not None
    assert recovered.shape == mask.shape


def test_worker_failure_does_not_refresh_clear_scene():
    publisher = FakePublisher()
    node = SimpleNamespace(
        publisher=publisher,
        get_parameter=lambda _name: SimpleNamespace(value='base_link'),
    )
    status = SimpleNamespace(data=(
        '{"state":"worker_result_rejected",'
        '"worker_status":"inference_failed",'
        '"reason":"sam2_target_lost_retry",'
        '"obstacle_count":0,'
        '"unsafe_obstacle_count":0,'
        '"image_stamp":{"sec":15,"nanosec":61}}'))

    ObstacleInstance3DNode.heavy_status_cb(node, status)

    assert publisher.messages == []


def test_target_only_seed_uses_current_labelled_object_manifest(tmp_path):
    key = '0000000001_000000002'
    spool = Path(tmp_path)
    (spool / 'seeds').mkdir()
    statuses = []
    bridge = SimpleNamespace(
        spool=spool,
        jpeg_cache={key: (None, b'jpeg')},
        seed_queued=False,
        publish_status=lambda state, **values: statuses.append((state, values)),
    )
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:8, 3:10] = 255

    assert Sam2LiveBridgeNode.queue_seed(bridge, key, mask, 'initial_target')

    seed = spool / 'seeds' / (key + '_initial_target')
    with (seed / 'seed.yaml').open(encoding='utf-8') as stream:
        manifest = yaml.safe_load(stream)
    assert manifest['objects'] == [{
        'object_id': 1,
        'role': 'target',
        'label': 'target',
        'mask_file': 'object_001.png',
    }]
    assert cv2.imread(str(seed / 'object_001.png'), cv2.IMREAD_GRAYSCALE) is not None
    assert not (seed / 'mask.png').exists()
    assert statuses[-1][0] == 'seed_queued'


def motion_bridge(parameters):
    return SimpleNamespace(
        joint_position_history=deque(),
        joint_position_span_rad=0.0,
        joint_position_window_duration_sec=0.0,
        arm_moving=False,
        arm_below_settle_since=0.0,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )


def test_position_window_ignores_stationary_encoder_quantization():
    parameters = {
        'arm_motion_window_sec': 0.75,
        'arm_moving_position_delta_rad': 0.012,
        'arm_settled_position_delta_rad': 0.009,
        'camera_settle_time_sec': 0.5,
    }
    bridge = motion_bridge(parameters)
    for index in range(11):
        joints = np.zeros(6)
        joints[4] = 0.0038 if index % 2 else -0.0038
        Sam2LiveBridgeNode.update_position_motion_window(
            bridge, index * 0.1, joints)
        Sam2LiveBridgeNode.update_arm_motion_state(bridge, index * 0.1)

    assert bridge.joint_position_span_rad == pytest.approx(0.0076)
    assert not bridge.arm_moving


def test_position_window_detects_sustained_joint_motion():
    parameters = {
        'arm_motion_window_sec': 0.75,
        'arm_moving_position_delta_rad': 0.012,
        'arm_settled_position_delta_rad': 0.009,
        'camera_settle_time_sec': 0.5,
    }
    bridge = motion_bridge(parameters)
    Sam2LiveBridgeNode.update_position_motion_window(
        bridge, 0.0, np.zeros(6))
    moved = np.zeros(6)
    moved[1] = 0.020
    Sam2LiveBridgeNode.update_position_motion_window(bridge, 0.5, moved)
    Sam2LiveBridgeNode.update_arm_motion_state(bridge, 0.5)

    assert bridge.joint_position_span_rad == pytest.approx(0.020)
    assert bridge.arm_moving
    assert bridge.arm_below_settle_since is None


def test_tracker_reset_clears_stale_gate_history():
    tracker = SimpleNamespace(
        filter=FakeFilter(),
        track_frames=7,
        missed_frames=11,
        stable_since=object(),
        last_stable=True,
        last_time=object(),
        last_source_u=100.0,
        last_source_v=200.0,
        last_area=42000.0,
        last_depth=0.36,
        last_measurement=[0.4, 0.0, 0.1],
        get_logger=lambda: FakeLogger(),
    )

    TargetTrackerNode.reset_tracking_state(tracker)

    assert tracker.filter.reset_called
    assert tracker.track_frames == 0
    assert tracker.missed_frames == 0
    assert tracker.stable_since is None
    assert not tracker.last_stable
    assert tracker.last_time is None
    assert tracker.last_source_u is None
    assert tracker.last_source_v is None
    assert tracker.last_area is None
    assert tracker.last_depth is None
    assert tracker.last_measurement is None


def test_occlusion_reference_uses_multi_frame_median():
    parameters = {
        'min_reference_mask_area_px': 300,
        'reference_initialization_frames': 5,
        'reference_update_alpha': 0.05,
    }
    checker = SimpleNamespace(
        reference_mask_area_px=0.0,
        reference_target_depth_m=0.0,
        reference_normalized_mask_area_m2=0.0,
        reference_mask_area_samples=[],
        reference_target_depth_samples=[],
        reference_normalized_area_samples=[],
        param_bool=lambda _name: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    # A single oversized startup mask must not become the clear-view reference.
    for area in (42000, 5400, 5500, 5450):
        OcclusionCheckerNode.update_reference_mask_area(checker, area, 0.30)
        assert checker.reference_mask_area_px == 0.0

    OcclusionCheckerNode.update_reference_mask_area(checker, 5525, 0.30)

    assert checker.reference_mask_area_px == pytest.approx(5500.0)
    assert checker.reference_target_depth_m == pytest.approx(0.30)
    assert checker.reference_normalized_mask_area_m2 == pytest.approx(495.0)
    assert checker.reference_mask_area_samples == []


def test_occlusion_visible_ratio_is_depth_normalized():
    parameters = {
        'heavy_visible_ratio': 0.44,
        'partial_visible_ratio': 0.94,
    }
    checker = SimpleNamespace(
        reference_normalized_mask_area_m2=5000.0 * 0.40 ** 2,
        param_bool=lambda _name: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    # A clear target twice as close occupies four times the pixels, but has
    # the same depth-normalized visible area.
    state, reason = OcclusionCheckerNode.classify_by_visible_ratio(
        checker, 20000, 0.20)

    assert state is None
    assert reason == ''


def test_valid_good_target_needs_closer_depth_evidence_for_occlusion():
    parameters = {
        'partial_occlusion_ratio': 0.12,
        'heavy_occlusion_ratio': 0.60,
        'min_valid_depth_ratio': 0.20,
        'min_occluder_area_px': 200,
    }
    checker = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        classify_by_visible_ratio=lambda *_args: (
            'PARTIALLY_OCCLUDED',
            'visible mask area dropped to 0.79 of clear reference',
        ),
    )

    state, reason = OcclusionCheckerNode.classify(
        checker,
        'GOOD',
        0.95,
        closer_area=0,
        closer_ratio=0.0,
        persisted=False,
        mask_area_px=7500,
        target_depth_m=0.30,
    )

    assert state == 'CLEAR'
    assert reason == 'target visible and no significant closer depth region'


def test_occlusion_reference_resets_for_new_scan_session():
    checker = SimpleNamespace(
        reference_mask_area_px=5500.0,
        reference_target_depth_m=0.30,
        reference_normalized_mask_area_m2=495.0,
        reference_mask_area_samples=[1.0],
        reference_target_depth_samples=[1.0],
        reference_normalized_area_samples=[1.0],
        reference_session_id='old',
        occlusion_history=[True],
        filtered_occlusion_state='HEAVILY_OCCLUDED',
        pending_occlusion_state='CLEAR',
        pending_occlusion_count=1,
        get_logger=lambda: FakeLogger(),
    )

    OcclusionCheckerNode.reset_reference(checker, 'MULTIVIEW_SCAN:new:PROPOSAL_READY')

    assert checker.reference_mask_area_px == 0.0
    assert checker.reference_target_depth_m == 0.0
    assert checker.reference_normalized_mask_area_m2 == 0.0
    assert checker.reference_mask_area_samples == []
    assert checker.reference_session_id == 'MULTIVIEW_SCAN:new:PROPOSAL_READY'
    assert checker.occlusion_history == []
    assert checker.filtered_occlusion_state is None


def test_unacknowledged_heavy_refresh_is_released_for_retry(monkeypatch):
    parameters = {
        'heavy_request_ack_timeout_sec': 3.0,
        'no_mask_refresh_timeout_sec': 8.0,
    }
    requested = []
    statuses = []
    bridge = SimpleNamespace(
        heavy_refresh_in_flight=True,
        heavy_refresh_acknowledged=False,
        heavy_refresh_sent_at=10.0,
        initial_requested=True,
        last_mask_publish=0.0,
        last_refresh_request=10.0,
        target_lost=False,
        loss_episode_active=False,
        update_arm_motion_state=lambda _now: None,
        begin_loss_episode=lambda reason: requested.append(reason),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        request_heavy_refresh=lambda reason: requested.append(reason),
        publish_status=lambda state, **_values: statuses.append(state),
    )
    monkeypatch.setattr(
        'piper_mobile_manipulation.sam2_live_bridge_node.time.monotonic', lambda: 20.0
    )

    Sam2LiveBridgeNode.retry_lost_refresh(bridge)

    assert not bridge.heavy_refresh_in_flight
    assert bridge.heavy_refresh_sent_at == 0.0
    assert statuses == ['heavy_refresh_ack_timeout']
    assert requested == ['sam2_no_mask_after_initial']


def test_heavy_refresh_acknowledgement_preserves_single_flight():
    bridge = SimpleNamespace(
        heavy_refresh_in_flight=True,
        heavy_refresh_acknowledged=False,
        heavy_refresh_sent_at=10.0,
    )

    Sam2LiveBridgeNode.heavy_status_cb(
        bridge, SimpleNamespace(data='{"state": "queued"}')
    )

    assert bridge.heavy_refresh_in_flight
    assert bridge.heavy_refresh_acknowledged
    assert bridge.heavy_refresh_sent_at == 10.0


def test_confirmed_heavy_bridge_idle_releases_acknowledged_job(monkeypatch):
    bridge = SimpleNamespace(
        heavy_refresh_in_flight=True,
        heavy_refresh_acknowledged=True,
        heavy_refresh_sent_at=10.0,
        deferred_refresh_reason='sam2_target_lost',
        next_refresh_allowed=0.0,
        loss_episode_active=False,
        get_parameter=lambda _name: SimpleNamespace(value=10.0),
    )
    bridge.finish_heavy_refresh = lambda: Sam2LiveBridgeNode.finish_heavy_refresh(bridge)
    monkeypatch.setattr(
        'piper_mobile_manipulation.sam2_live_bridge_node.time.monotonic', lambda: 20.0
    )

    Sam2LiveBridgeNode.heavy_status_cb(
        bridge, SimpleNamespace(data='{"state": "idle"}')
    )

    assert not bridge.heavy_refresh_in_flight
    assert not bridge.heavy_refresh_acknowledged
    assert bridge.heavy_refresh_sent_at == 0.0
    assert bridge.deferred_refresh_reason is None
    assert bridge.next_refresh_allowed == 30.0


def test_repeated_lost_status_is_one_latched_episode():
    requested = []
    bridge = SimpleNamespace(
        loss_episode_active=False,
        heavy_attempt_count=4,
        recovery_valid_frames=3,
        loss_reason='',
        target_lost=False,
        lifecycle_state='TRACKING',
        motion_prompt_seeded_episode=True,
        motion_prompt_seeded_at=1.0,
        camera_settled=lambda: True,
        try_motion_prompt_seed=lambda: False,
        request_heavy_refresh=lambda reason: requested.append(reason),
    )

    Sam2LiveBridgeNode.begin_loss_episode(bridge, 'target_status_lost')
    Sam2LiveBridgeNode.begin_loss_episode(bridge, 'target_status_lost')

    assert bridge.loss_episode_active
    assert bridge.heavy_attempt_count == 0
    assert bridge.loss_reason == 'target_status_lost'
    assert requested == ['target_status_lost']


def test_loss_during_arm_motion_waits_without_heavy_request():
    requested = []
    statuses = []
    bridge = SimpleNamespace(
        loss_episode_active=False,
        heavy_attempt_count=0,
        recovery_valid_frames=0,
        loss_reason='',
        target_lost=False,
        lifecycle_state='TRACKING',
        motion_prompt_seeded_episode=False,
        motion_prompt_seeded_at=0.0,
        camera_settled=lambda: False,
        try_motion_prompt_seed=lambda: False,
        request_heavy_refresh=lambda reason: requested.append(reason),
        publish_status=lambda state, **values: statuses.append((state, values)),
    )

    Sam2LiveBridgeNode.begin_loss_episode(bridge, 'target_status_lost')

    assert requested == []
    assert bridge.lifecycle_state == 'WAITING_TO_REACQUIRE'
    assert statuses[0][0] == 'reacquisition_waiting_for_motion'


def test_initial_lost_status_waits_for_explicit_acquisition_seed():
    requested = []
    bridge = SimpleNamespace(
        has_ever_tracked=False,
        last_target_status='SEARCHING',
        loss_episode_active=False,
        lifecycle_state='WAITING_TO_REACQUIRE',
        begin_loss_episode=lambda reason: requested.append(reason),
        maybe_complete_recovery=lambda: None,
    )

    Sam2LiveBridgeNode.target_status_cb(
        bridge, SimpleNamespace(data='LOST'))

    assert requested == []
    assert bridge.last_target_status == 'LOST'


def test_initial_occlusion_status_does_not_start_semantic_search():
    requested = []
    bridge = SimpleNamespace(
        has_ever_tracked=False,
        begin_loss_episode=lambda reason: requested.append(reason),
    )

    Sam2LiveBridgeNode.occlusion_cb(
        bridge, SimpleNamespace(data='{"status": "HEAVILY_OCCLUDED"}'))

    assert requested == []


def test_sustained_low_confidence_requests_one_direct_heavy_refresh(monkeypatch):
    parameters = {
        'low_confidence_refresh_threshold': 0.60,
        'low_confidence_refresh_duration_sec': 1.0,
        'low_confidence_refresh_hysteresis': 0.10,
    }
    requests = []
    bridge = SimpleNamespace(
        low_confidence_since=None,
        low_confidence_refresh_latched=False,
        loss_episode_active=False,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        begin_loss_episode=lambda reason, **kwargs: requests.append((reason, kwargs)),
    )
    times = iter((10.0, 11.1, 12.0))
    monkeypatch.setattr(
        'piper_mobile_manipulation.sam2_live_bridge_node.time.monotonic',
        lambda: next(times),
    )
    low = SimpleNamespace(valid=True, confidence=0.59)

    Sam2LiveBridgeNode.tracked_target_cb(bridge, low)
    Sam2LiveBridgeNode.tracked_target_cb(bridge, low)
    Sam2LiveBridgeNode.tracked_target_cb(bridge, low)

    assert bridge.low_confidence_refresh_latched
    assert len(requests) == 1
    assert requests[0][1] == {'allow_motion_prompt': False}


def test_low_confidence_latch_requires_hysteresis_to_rearm():
    parameters = {
        'low_confidence_refresh_threshold': 0.60,
        'low_confidence_refresh_duration_sec': 1.0,
        'low_confidence_refresh_hysteresis': 0.10,
    }
    bridge = SimpleNamespace(
        low_confidence_since=1.0,
        low_confidence_refresh_latched=True,
        loss_episode_active=False,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    Sam2LiveBridgeNode.tracked_target_cb(
        bridge, SimpleNamespace(valid=True, confidence=0.65))
    assert bridge.low_confidence_refresh_latched

    Sam2LiveBridgeNode.tracked_target_cb(
        bridge, SimpleNamespace(valid=True, confidence=0.72))
    assert not bridge.low_confidence_refresh_latched


def test_eye_in_hand_camera_changes_do_not_reject_consistent_base_measurement():
    tracker = SimpleNamespace(
        min_confidence=0.2,
        use_camera_space_gates=False,
        last_depth=0.35,
        depth_gate_m=0.15,
        last_source_u=100.0,
        last_source_v=100.0,
        max_pixel_jump=80.0,
        last_measurement=[0.40, 0.0, 0.10],
        max_3d_jump=0.10,
        max_target_speed=1.0,
        last_area=5000.0,
        min_area_ratio=0.5,
        max_area_ratio=2.0,
        detection_area=TargetTrackerNode.detection_area,
    )
    measurement = [0.42, 0.0, 0.10]
    msg = SimpleNamespace(
        depth=0.70,
        source_u=420.0,
        source_v=260.0,
        detection_width=150.0,
        detection_height=120.0,
    )

    reason = TargetTrackerNode.gate_measurement(
        tracker, msg, measurement, confidence=0.9
    )

    assert reason is None


def test_loss_episode_waits_for_periodic_retry_after_burst_limit():
    parameters = {
        'motion_prompt_recovery_grace_sec': 0.75,
        'max_reacquisition_attempts': 2,
        'absent_retry_sec': 30.0,
        'refresh_cooldown_sec': 5.0,
        'lost_refresh_retry_sec': 10.0,
    }
    bridge = SimpleNamespace(
        motion_prompt_seeded_episode=False,
        motion_prompt_seeded_at=0.0,
        heavy_attempt_count=2,
        heavy_refresh_in_flight=False,
        next_refresh_allowed=0.0,
        last_refresh_request=80.0,
        lifecycle_state='WAITING_TO_REACQUIRE',
        camera_settled=lambda _now: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        request_heavy_refresh=lambda *_args, **_kwargs: pytest.fail(
            'periodic retry ran before its interval'
        ),
    )

    requested = Sam2LiveBridgeNode.retry_loss_episode(
        bridge, 'sam2_target_lost_retry', now=100.0
    )

    assert not requested
    assert bridge.lifecycle_state == 'ABSENT'


def test_loss_episode_periodically_retries_after_burst_limit():
    parameters = {
        'motion_prompt_recovery_grace_sec': 0.75,
        'max_reacquisition_attempts': 2,
        'absent_retry_sec': 30.0,
        'refresh_cooldown_sec': 5.0,
        'lost_refresh_retry_sec': 10.0,
    }
    requests = []
    bridge = SimpleNamespace(
        motion_prompt_seeded_episode=False,
        motion_prompt_seeded_at=0.0,
        heavy_attempt_count=2,
        heavy_refresh_in_flight=False,
        next_refresh_allowed=0.0,
        last_refresh_request=60.0,
        lifecycle_state='WAITING_TO_REACQUIRE',
        camera_settled=lambda _now: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        request_heavy_refresh=lambda reason, **kwargs: requests.append(
            (reason, kwargs)
        ) or True,
    )

    requested = Sam2LiveBridgeNode.retry_loss_episode(
        bridge, 'sam2_target_lost_retry', now=100.0
    )

    assert requested
    assert bridge.lifecycle_state == 'ABSENT'
    assert requests == [
        ('sam2_target_lost_retry', {'allow_attempt_limit': True})
    ]


def test_periodic_absent_request_bypasses_burst_limit_once(monkeypatch):
    parameters = {
        'max_reacquisition_attempts': 2,
        'refresh_cooldown_sec': 5.0,
    }
    published = []
    statuses = []
    bridge = SimpleNamespace(
        loss_episode_active=True,
        loss_reason='target_status_lost',
        heavy_attempt_count=2,
        lifecycle_state='ABSENT',
        heavy_refresh_in_flight=False,
        heavy_refresh_acknowledged=False,
        heavy_refresh_sent_at=0.0,
        deferred_refresh_reason=None,
        next_refresh_allowed=0.0,
        last_refresh_request=60.0,
        last_semantic_refresh=60.0,
        camera_settled=lambda _now: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        request_pub=SimpleNamespace(publish=lambda msg: published.append(msg.data)),
        publish_status=lambda state, **values: statuses.append((state, values)),
    )
    monkeypatch.setattr(
        'piper_mobile_manipulation.sam2_live_bridge_node.time.monotonic',
        lambda: 100.0,
    )

    requested = Sam2LiveBridgeNode.request_heavy_refresh(
        bridge, 'sam2_target_lost_retry', allow_attempt_limit=True
    )

    assert requested
    assert len(published) == 1
    assert bridge.heavy_attempt_count == 3
    assert bridge.heavy_refresh_in_flight
    assert bridge.lifecycle_state == 'REACQUIRING'
    assert statuses[-1][1]['retry_mode'] == 'periodic_absent'


def test_valid_masks_and_tracker_status_rearm_loss_episode():
    statuses = []
    bridge = SimpleNamespace(
        loss_episode_active=True,
        recovery_valid_frames=5,
        last_target_status='TRACKING',
        target_lost=True,
        heavy_attempt_count=2,
        loss_reason='target_status_lost',
        motion_prompt_seeded_episode=True,
        motion_prompt_seeded_at=1.0,
        lifecycle_state='DEGRADED',
        get_parameter=lambda _name: SimpleNamespace(value=5),
        publish_status=lambda state, **_values: statuses.append(state),
    )

    Sam2LiveBridgeNode.maybe_complete_recovery(bridge)

    assert not bridge.loss_episode_active
    assert not bridge.target_lost
    assert bridge.heavy_attempt_count == 0
    assert bridge.lifecycle_state == 'TRACKING'
    assert statuses == ['loss_episode_recovered']


def test_safe_servo_holds_until_tracker_is_actively_tracking():
    servo = SimpleNamespace(
        tracking_speed_scale=1.0,
        tracking_lifecycle='TRACKING',
        target_status='SEARCHING',
    )

    reason = SafeServoNode.stop_reason(servo, SimpleNamespace())

    assert reason == 'target_status=SEARCHING'


def test_tf_listener_recovery_requires_continuous_failure_window():
    assert not tf_listener_recovery_due(
        False, 10.0, 0.0, 20.0, 3.0, 2.0)
    assert not tf_listener_recovery_due(
        True, None, 0.0, 20.0, 3.0, 2.0)
    assert not tf_listener_recovery_due(
        True, 10.0, 0.0, 12.9, 3.0, 2.0)
    assert not tf_listener_recovery_due(
        True, 10.0, 12.5, 13.1, 3.0, 2.0)
    assert tf_listener_recovery_due(
        True, 10.0, 10.0, 13.1, 3.0, 2.0)
