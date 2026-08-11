import json
from types import SimpleNamespace

from piper_mobile_manipulation.supervised_workflow import (
    capture_cloud_ready,
    choose_removal_plan,
    cloud_model,
    corroborated_target_motion_rejection,
    heavy_refinement_status_action,
    semantic_scene_correlation_rejection,
    occlusion_capture_rejection,
    should_cache_capture_cloud,
    tracking_allows_target_motion_check,
)
from piper_mobile_manipulation.supervised_cube_workflow_node import (
    SupervisedCubeWorkflowNode,
    target_motion_is_terminal,
)


def p(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def stamped_scene(sec, nanosec):
    return SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def test_occlusion_probe_requires_exact_request_and_image_stamp():
    status = {
        'state': 'published',
        'request_id': 'probe-1',
        'image_stamp': {'sec': 12, 'nanosec': 34},
    }

    assert semantic_scene_correlation_rejection(
        'probe-1', status, stamped_scene(12, 34)) == ''
    assert 'not request-correlated' in semantic_scene_correlation_rejection(
        'probe-2', status, stamped_scene(12, 34))
    assert 'does not match' in semantic_scene_correlation_rejection(
        'probe-1', status, stamped_scene(12, 35))


def test_occlusion_probe_never_treats_target_missing_result_as_clear():
    status = {
        'state': 'worker_result_rejected',
        'request_id': 'probe-1',
        'worker_status': 'target_mask_missing',
        'image_stamp': {'sec': 12, 'nanosec': 34},
    }

    assert 'not a published target result' in semantic_scene_correlation_rejection(
        'probe-1', status, stamped_scene(12, 34))


def test_workflow_rejects_correlated_target_missing_occlusion_probe():
    aborted = []
    workflow = SimpleNamespace(
        state='INITIALIZING',
        occlusion_probe_request_id='probe-1',
        occlusion_probe_status=None,
        occlusion_probe_waiting_retry=False,
        occlusion_probe_attempt=1,
        parse=lambda msg: json.loads(msg.data),
        abort=lambda reason: aborted.append(reason),
        publish_status=lambda _reason: None,
        mark=lambda _key: None,
    )
    message = SimpleNamespace(data=json.dumps({
        'state': 'worker_result_rejected',
        'request_id': 'probe-1',
        'worker_status': 'target_mask_missing',
    }))

    SupervisedCubeWorkflowNode.heavy_refresh_status_cb(workflow, message)

    assert aborted == [
        'dedicated occlusion probe could not re-detect the locked target']


def test_workflow_accepts_only_correlated_published_occlusion_probe():
    marked = []
    workflow = SimpleNamespace(
        state='INITIALIZING',
        occlusion_probe_request_id='probe-1',
        occlusion_probe_status=None,
        occlusion_probe_waiting_retry=False,
        occlusion_probe_attempt=1,
        parse=lambda msg: json.loads(msg.data),
        abort=lambda _reason: None,
        publish_status=lambda _reason: None,
        mark=lambda key: marked.append(key),
        obstacles=None,
        cache_correlated_probe_scene=lambda _scene: None,
    )
    old = SimpleNamespace(data=json.dumps({
        'state': 'published', 'request_id': 'acquisition-old',
        'image_stamp': {'sec': 12, 'nanosec': 33},
    }))
    current = SimpleNamespace(data=json.dumps({
        'state': 'published', 'request_id': 'probe-1',
        'image_stamp': {'sec': 12, 'nanosec': 34},
    }))

    SupervisedCubeWorkflowNode.heavy_refresh_status_cb(workflow, old)
    assert workflow.occlusion_probe_status is None
    SupervisedCubeWorkflowNode.heavy_refresh_status_cb(workflow, current)

    assert workflow.occlusion_probe_status['request_id'] == 'probe-1'
    assert marked == ['occlusion_probe']


def test_correlated_probe_scene_is_cached_against_live_overwrite():
    marked = []
    workflow = SimpleNamespace(
        occlusion_probe_request_id='probe-1',
        occlusion_probe_status={
            'state': 'published',
            'request_id': 'probe-1',
            'image_stamp': {'sec': 12, 'nanosec': 34},
        },
        occlusion_probe_scene=None,
        mark=lambda key: marked.append(key),
    )
    exact = stamped_scene(12, 34)
    newer_live = stamped_scene(12, 35)

    SupervisedCubeWorkflowNode.cache_correlated_probe_scene(workflow, exact)
    SupervisedCubeWorkflowNode.cache_correlated_probe_scene(workflow, newer_live)

    assert workflow.occlusion_probe_scene is exact
    assert marked == ['occlusion_probe_scene']


def test_moving_target_mode_keeps_pre_scan_lock_strict_but_allows_scan_motion():
    assert target_motion_is_terminal(True, 'INITIALIZING')
    assert target_motion_is_terminal(True, 'PLAN_READY')
    assert not target_motion_is_terminal(True, 'SCAN_READY')
    assert not target_motion_is_terminal(True, 'WAIT_CAPTURE')
    assert target_motion_is_terminal(False, 'SCAN_READY')


def obstacle(object_id=2, label='pen', center=(0.4, 0.1, 0.05), size=(0.05, 0.01, 0.01)):
    return SimpleNamespace(
        object_id=object_id, semantic_label=label, valid=True, validity_reason='ok',
        base_centroid=p(*center),
        base_bounds_min=p(*(center[i] - size[i] / 2 for i in range(3))),
        base_bounds_max=p(*(center[i] + size[i] / 2 for i in range(3))),
    )


def config():
    return dict(
        movable_whitelist=['pen'], target_clearance_m=0.04,
        drop_target_clearance_m=0.12, drop_obstacle_clearance_m=0.08,
        drop_search_radius_m=0.18, max_grasp_width_m=0.07,
        approach_height_m=0.10, pre_push_offset_m=0.08, push_distance_m=0.06,
        workspace_x_min=0.10, workspace_x_max=0.70,
        workspace_y_min=-0.40, workspace_y_max=0.40,
        workspace_z_min=0.02, workspace_z_max=0.60,
        enforce_static_workspace=False,
        require_observed_drop_support=False,
    )


def test_prefers_pick_for_graspable_pen():
    item = obstacle()
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert plan['valid']
    assert plan['action'] == 'pick_and_place'
    assert plan['execute'] is False


def test_unknown_label_is_rejected():
    item = obstacle(label='knife')
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert not plan['valid']


def test_large_pen_uses_outward_push():
    item = obstacle(center=(0.45, 0.0, 0.05), size=(0.09, 0.02, 0.02))
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert plan['valid']
    assert plan['action'] == 'push'
    assert plan['push_end'][0] > plan['object_center'][0]


def test_obstacle_inside_target_clearance_is_rejected():
    item = obstacle(center=(0.36, 0.0, 0.05))
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert not plan['valid']


def test_dynamic_planning_does_not_reject_a_static_workspace_box():
    item = obstacle(center=(0.755, 0.16, 0.26))
    plan = choose_removal_plan(item, (0.55, 0.0, 0.05), [item], config())
    assert plan['valid']


def test_pick_requires_and_records_observed_flat_drop_support():
    item = obstacle(center=(0.40, 0.0, 0.05))
    values = config()
    values.update({
        'require_observed_drop_support': True,
        'drop_support_radius_m': 0.04,
        'drop_support_min_points': 8,
        'drop_support_max_stddev_m': 0.004,
    })
    support = [
        (0.52 + x * 0.01, 0.20 + y * 0.01, 0.02)
        for x in range(-6, 7) for y in range(-6, 7)
    ]

    plan = choose_removal_plan(
        item, (0.30, 0.0, 0.05), [item], values, support)

    assert plan['valid']
    assert plan['action'] == 'pick_and_place'
    assert plan['drop_support_verified']


def test_cloud_model_rejects_outlier_for_center():
    points = [(0.3, 0.0, 0.1)] * 50 + [(0.31, 0.01, 0.11)] * 50 + [(10, 10, 10)]
    model = cloud_model(points, 'base_link', 5)
    assert model['valid']
    assert model['center'][0] < 0.32
    assert model['accepted_views'] == 5


def test_capture_cloud_is_cached_before_acceptance_then_processed():
    assert should_cache_capture_cloud('WAIT_CAPTURE', 0, 0)
    assert not capture_cloud_ready(0, 0, True)
    assert capture_cloud_ready(1, 0, True)


def test_capture_cloud_is_processed_when_acceptance_arrives_first():
    assert should_cache_capture_cloud('SCAN_READY', 1, 0)
    assert capture_cloud_ready(1, 0, True)
    assert not should_cache_capture_cloud('SCAN_READY', 1, 1)


def test_saved_rgbd_view_is_accepted_without_cloud_or_quality_gate():
    published = []
    cloud_requests = []
    node = SimpleNamespace(
        state='SCAN_READY',
        accepted_views=0,
        modeled_views=0,
        get_parameter=lambda name: SimpleNamespace(
            value=13 if name == 'max_views' else False),
        cloud_request_pub=SimpleNamespace(
            publish=lambda msg: cloud_requests.append(msg)),
        publish_status=lambda reason: published.append(reason),
        reply=lambda response, success, message: SimpleNamespace(
            success=success, message=message),
    )

    response = SupervisedCubeWorkflowNode.capture_view_cb(
        node, None, SimpleNamespace())

    assert response.success
    assert node.accepted_views == 1
    assert node.modeled_views == 1
    assert cloud_requests == []
    assert 'synchronized RGB-D viewpoint accepted' in published[-1]


def test_capture_requires_fresh_clear_occlusion_evidence():
    assert occlusion_capture_rejection(
        {'occlusion_state': 'CLEAR'}, True) == ''
    assert 'HEAVILY_OCCLUDED' in occlusion_capture_rejection({
        'occlusion_state': 'HEAVILY_OCCLUDED',
        'reason': 'large closer depth region',
    }, True)
    assert 'missing or stale' in occlusion_capture_rejection(
        {'occlusion_state': 'CLEAR'}, False)


def tracking_health(**changes):
    values = dict(
        lifecycle_state='TRACKING',
        arm_moving=False,
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.1,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_target_motion_check_requires_settled_measured_tracking():
    assert tracking_allows_target_motion_check(tracking_health(), 0.75)
    assert not tracking_allows_target_motion_check(
        tracking_health(arm_moving=True), 0.75)
    assert not tracking_allows_target_motion_check(
        tracking_health(camera_settled=False), 0.75)
    assert not tracking_allows_target_motion_check(
        tracking_health(prediction_only=True), 0.75)
    assert not tracking_allows_target_motion_check(
        tracking_health(measurement_age_sec=0.8), 0.75)
    assert not tracking_allows_target_motion_check(
        tracking_health(lifecycle_state='DEGRADED'), 0.75)


def test_heavy_refinement_busy_is_correlated_and_retryable():
    request_id = 'cloud_capture_1'
    assert heavy_refinement_status_action({
        'state': 'request_ignored_busy',
        'request_id': request_id,
    }, request_id) == 'retry'
    assert heavy_refinement_status_action({
        'state': 'request_ignored_busy',
        'request_id': 'another_request',
    }, request_id) == 'ignore'
    assert heavy_refinement_status_action({
        'state': 'worker_result_rejected',
        'request_id': request_id,
    }, request_id) == 'fail'
    assert heavy_refinement_status_action({
        'state': 'queued',
        'request_id': request_id,
    }, request_id) == 'wait'


def test_target_motion_requires_fresh_independent_geometric_corroboration():
    stable_landmark = {
        'state': 'LOCKED',
        'measurement_error_m': 0.008,
    }
    moved_landmark = {
        'state': 'RESCAN_NEEDED',
        'measurement_error_m': 0.024,
    }
    assert corroborated_target_motion_rejection(
        0.010, 0.020, {}, False) == ''
    assert corroborated_target_motion_rejection(
        0.024, 0.020, stable_landmark, True) == ''
    assert 'cube landmark moved' in corroborated_target_motion_rejection(
        0.024, 0.020, moved_landmark, True)
    assert corroborated_target_motion_rejection(
        0.026, 0.020, {
            'state': 'RESCAN_NEEDED',
            'measurement_error_m': 0.605,
        }, True) == ''
    assert 'missing or stale' in corroborated_target_motion_rejection(
        0.024, 0.020, stable_landmark, False)


def test_cube_surface_uncertainty_does_not_raise_physical_motion_threshold():
    surface_shift = {
        'state': 'RESCAN_NEEDED',
        'measurement_error_m': 0.028,
    }
    actual_shift = {
        'state': 'RESCAN_NEEDED',
        'measurement_error_m': 0.041,
    }
    assert corroborated_target_motion_rejection(
        0.027, 0.020, surface_shift, True, 0.015) == ''
    assert 'cube landmark moved' in corroborated_target_motion_rejection(
        0.040, 0.020, actual_shift, True, 0.015)
