"""Probe ownership, mission reset and stale-occlusion capture regressions."""

from types import SimpleNamespace
import time

import numpy as np
import pytest

from piper_mobile_manipulation.perception.occlusion import qualified_probe_assumption
from piper_mobile_manipulation.scan_capture_node import ScanCaptureNode
from piper_mobile_manipulation.supervised_cube_workflow_node import SupervisedCubeWorkflowNode
from piper_mobile_manipulation.occlusion_checker_node import OcclusionCheckerNode


def receipt(**changes):
    result = dict(session_id='mission-a', state='SCAN_READY',
                  occlusion_probe_request_id='mission-a-occlusion-1',
                  mission_occlusion_probe_passed=True,
                  capture_occlusion_probe_pending=False, occlusion_labels=[])
    result.update(changes)
    return result


@pytest.mark.parametrize('changes', [
    {'session_id': 'mission-b'}, {'mission_occlusion_probe_passed': False},
    {'state': 'INITIALIZING'}, {'state': 'ABORTED'}, {'state': 'COMPLETE'},
    {'capture_occlusion_probe_pending': True}, {'occlusion_probe_request_id': ''},
])
def test_assumption_requires_passed_probe_from_same_active_session(changes):
    assert qualified_probe_assumption(receipt(**changes), 'mission-a') is None


def capture_fixture():
    params = dict(dry_run=True, enable_real_arm_motion=False, require_mask=True,
                  require_valid_target=True, require_good_quality_for_service=True,
                  require_clear_occlusion_for_service=True,
                  allow_classified_occlusion_for_service=True,
                  allow_classified_occlusion_for_first_capture=False,
                  diagnostic_timeout_sec=1.0, minimum_accepted_quality_score=0.65)
    return SimpleNamespace(
        prepared_capture=None, frame_index=3, capture_workflow_session_id='mission-a',
        latest_workflow_status=receipt(),
        param_bool=lambda name: params[name], capture_mode=lambda: 'service',
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        latest_mask=np.ones((2, 2)), latest_target=SimpleNamespace(valid=True),
        latest_execution_status=SimpleNamespace(
            execution_mode='MULTIVIEW_SCAN', state='CAPTURING_RGBD', current_view=4),
        latest_scan_quality={'quality_label': 'GOOD', 'quality_score': 0.8,
                             'target_valid': True}, latest_scan_quality_at=time.monotonic(),
        latest_occlusion_status={'occlusion_state': 'UNKNOWN'}, latest_occlusion_status_at=0)


def test_incident_capture_ignores_stale_unknown_occlusion_after_probe():
    node = capture_fixture()
    ready, reason, evidence = ScanCaptureNode.capture_prerequisite_evidence(node)
    assert ready, reason
    assert evidence['occlusion_raw']['occlusion_state'] == 'UNKNOWN'
    assert evidence['occlusion_metadata']['occlusion_policy'] == (
        'assumed_clear_after_qualified_probe')
    assert evidence['occlusion_metadata']['occlusion_policy_session_id'] == 'mission-a'


@pytest.mark.parametrize('change', [
    'first', 'other_session', 'unpassed', 'quality', 'target', 'mask'])
def test_assumption_does_not_bypass_first_probe_or_other_capture_checks(change):
    node = capture_fixture()
    if change == 'first':
        node.frame_index = 0
    elif change == 'other_session':
        node.capture_workflow_session_id = 'other'
    elif change == 'unpassed':
        node.latest_workflow_status['mission_occlusion_probe_passed'] = False
    elif change == 'quality':
        node.latest_scan_quality_at = 0
    elif change == 'target':
        node.latest_target.valid = False
    else:
        node.latest_mask = None
    assert not ScanCaptureNode.capture_prerequisite_evidence(node)[0]


@pytest.mark.parametrize('occlusion', [None, {'occlusion_state': 'UNKNOWN'},
                                       {'occlusion_state': 'HEAVILY_OCCLUDED'}])
def test_workflow_accepts_later_saved_capture_without_rechecking(occlusion):
    node = SimpleNamespace(
        state='SCAN_READY', accepted_views=3, modeled_views=3,
        mission_occlusion_probe_passed=True, first_capture_occlusion_cleared=True,
        occlusion=occlusion, get_parameter=lambda _: SimpleNamespace(value=24),
        accept_saved_view=lambda _: None,
        reply=lambda response, success, message: SimpleNamespace(success=success))
    assert SupervisedCubeWorkflowNode.capture_view_cb(node, None, None).success


def test_new_mission_clears_the_previous_probe_receipt():
    node = SimpleNamespace(state='COMPLETE', mission_occlusion_probe_passed=True,
                           session_id='old', updated={}, publish_status=lambda _: None,
                           reply=lambda *args: None)
    SupervisedCubeWorkflowNode.start_cb(node, None, None)
    assert not node.mission_occlusion_probe_passed
    assert node.session_id != 'old'
    assert node.occlusion_probe_request_id == ''


def test_abort_revokes_probe_assumption():
    node = SimpleNamespace(state='SCAN_READY', mission_occlusion_probe_passed=True,
                           publish_status=lambda _: None,
                           get_logger=lambda: SimpleNamespace(error=lambda _: None))
    SupervisedCubeWorkflowNode.abort(node, 'cancel')
    assert not node.mission_occlusion_probe_passed


def test_occlusion_timer_does_not_evaluate_images_after_qualified_probe():
    published = []
    node = SimpleNamespace(
        last_eval_stamp=0, latest_workflow_status=receipt(),
        get_parameter=lambda _: SimpleNamespace(value=0.4),
        base_payload=lambda state, reason: dict(occlusion_state=state, reason=reason),
        publish_status=published.append, publish_debug=lambda _: None)
    # No evaluate_current_view method: the latched path must never call it.
    OcclusionCheckerNode.timer_cb(node)
    assert published[0]['occlusion_is_live_measurement'] is False


def test_recorder_cannot_reuse_another_missions_passed_probe():
    import threading
    node = SimpleNamespace(
        diagnostic_lock=threading.Lock(), frame_index=0,
        parse_json_msg=lambda message: message, notify_capture_evidence=lambda: None)
    ScanCaptureNode.workflow_status_cb(node, receipt(mission_occlusion_probe_passed=False))
    assert node.capture_workflow_session_id == 'mission-a'
    node.frame_index = 1
    ScanCaptureNode.workflow_status_cb(node, receipt())
    assert qualified_probe_assumption(node.latest_workflow_status,
                                      node.capture_workflow_session_id)
    ScanCaptureNode.workflow_status_cb(node, receipt(
        session_id='mission-b', occlusion_probe_request_id='mission-b-occlusion-1'))
    assert node.capture_workflow_session_id == 'mission-a'
    assert qualified_probe_assumption(node.latest_workflow_status,
                                      node.capture_workflow_session_id) is None


@pytest.mark.parametrize('state', ['CLEAR', 'PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED'])
def test_every_viewpoint_requests_probe_even_after_previous_pass(state):
    probes = []
    node = SimpleNamespace(
        state='SCAN_READY', accepted_views=3, modeled_views=3,
        mission_occlusion_probe_passed=True, first_capture_occlusion_cleared=True,
        occlusion_probe_every_viewpoint=True,
        occlusion={'occlusion_state': state}, fresh=lambda _: True,
        get_parameter=lambda _: SimpleNamespace(value=24),
        begin_capture_occlusion_probe=lambda value, first_capture: probes.append(
            (value, first_capture)),
        reply=lambda response, success, message: SimpleNamespace(success=success))
    assert SupervisedCubeWorkflowNode.capture_view_cb(node, None, None).success
    assert probes == [(state, False)]


def test_every_viewpoint_receipt_cannot_bypass_live_recorder_gate():
    node = capture_fixture()
    node.latest_workflow_status = receipt(occlusion_probe_every_viewpoint=True)
    assert qualified_probe_assumption(node.latest_workflow_status) is None
    assert not ScanCaptureNode.capture_prerequisite_evidence(node)[0]
    node.latest_occlusion_status = {'occlusion_state': 'CLEAR'}
    node.latest_occlusion_status_at = time.monotonic()
    ready, reason, evidence = ScanCaptureNode.capture_prerequisite_evidence(node)
    assert ready, reason
    assert evidence.get('occlusion_metadata', {}).get('occlusion_policy') != (
        'assumed_clear_after_qualified_probe')


def test_every_viewpoint_checker_still_evaluates_live_images():
    evaluated = []
    published = []
    node = SimpleNamespace(
        last_eval_stamp=0,
        latest_workflow_status=receipt(occlusion_probe_every_viewpoint=True),
        get_parameter=lambda _: SimpleNamespace(value=0.4),
        evaluate_current_view=lambda now: evaluated.append(now) or {
            'occlusion_state': 'HEAVILY_OCCLUDED'},
        stabilize_payload=lambda payload: payload,
        publish_status=published.append, publish_debug=lambda _: None)
    OcclusionCheckerNode.timer_cb(node)
    assert len(evaluated) == 1
    assert published == [{'occlusion_state': 'HEAVILY_OCCLUDED'}]
