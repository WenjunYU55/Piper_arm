from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fixtures.piper_gui_automation import (
    ACQUISITION_PLAN_TIMEOUT_SEC,
    ACQUISITION_SERVICE_TIMEOUT_SEC,
    AcquisitionPhase,
    AutomationSession,
    command_publisher_identity_pending,
    command_publisher_ownership_rejection,
    MULTIVIEW_PLAN_TIMEOUT_SEC,
    MULTIVIEW_SCAN,
    PLAN_REQUEST_QUEUE_TIMEOUT_SEC,
    plan_matches_request,
    readiness_rejection,
    retryable_multiview_terminal,
    ROUGH_ACQUISITION,
    Step4Phase,
    STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC,
    STEP45_AUTO_RECOVERY_MAX_ATTEMPTS,
    STEP45_AUTO_RECOVERY_RETRY_SEC,
    step45_auto_recovery_blocker,
    step4_workflow_action,
    tracking_health_rejection,
    tracking_lock_rejection,
    validate_automation_speed,
    validate_rough_coordinates,
    WORKFLOW_ASSESSMENT_TIMEOUT_SEC,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_command_publisher_ownership_requires_exact_executor_identity():
    assert command_publisher_ownership_rejection(
        ['/scan_viewpoint_executor']) == ''
    assert command_publisher_identity_pending(
        ['/_NODE_NAMESPACE_UNKNOWN_/_NODE_NAME_UNKNOWN_'])
    assert command_publisher_ownership_rejection(
        ['/_NODE_NAMESPACE_UNKNOWN_/_NODE_NAME_UNKNOWN_'],
        owned_stack_running=True,
        executor_node_present=True,
    ) == ''
    assert 'found: none' in command_publisher_ownership_rejection([])
    assert '_UNKNOWN_' in command_publisher_ownership_rejection(
        ['/_NODE_NAMESPACE_UNKNOWN_/_NODE_NAME_UNKNOWN_'])
    assert '_UNKNOWN_' in command_publisher_ownership_rejection(
        ['/_NODE_NAMESPACE_UNKNOWN_/_NODE_NAME_UNKNOWN_'],
        owned_stack_running=False,
        executor_node_present=True,
    )
    assert '_UNKNOWN_' in command_publisher_ownership_rejection(
        ['/_NODE_NAMESPACE_UNKNOWN_/_NODE_NAME_UNKNOWN_'],
        owned_stack_running=True,
        executor_node_present=False,
    )
    assert 'other_publisher' in command_publisher_ownership_rejection(
        ['/scan_viewpoint_executor', '/other_publisher'])


def test_consumed_multiview_terminal_is_retryable_with_a_new_session():
    assert retryable_multiview_terminal(
        MULTIVIEW_SCAN, 'ABORTED', scan_approval_used=True)
    assert retryable_multiview_terminal(
        MULTIVIEW_SCAN, 'INVALID', scan_approval_used=True)
    assert not retryable_multiview_terminal(
        MULTIVIEW_SCAN, 'COMPLETE', scan_approval_used=True)
    assert not retryable_multiview_terminal(
        MULTIVIEW_SCAN, 'ABORTED', scan_approval_used=False)
    assert not retryable_multiview_terminal(
        ROUGH_ACQUISITION, 'ABORTED', scan_approval_used=True)


def test_step45_auto_recovery_is_bounded_and_never_hides_operator_blockers():
    assert STEP45_AUTO_RECOVERY_MAX_ATTEMPTS == 2
    assert STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC == 20.0
    assert STEP45_AUTO_RECOVERY_RETRY_SEC == 0.50
    for transient in (
            'proposal expired; refresh viewpoints',
            'workflow diagnostic service timed out',
            'camera timestamp CLOCK_OFFSET',
            'cube landmark moved beyond tolerance',
            'matching RGB-D frame expired'):
        assert step45_auto_recovery_blocker(transient) == ''
    for operator_stop in (
            'movable clutter was detected; clear the workspace',
            'obstacle scene is blocked',
            'Expected exactly one executor command publisher',
            'collision model is not qualified',
            'current joint3 is outside configured limits',
            'arm is not enabled',
            'managed scan stack failed',
            'operator cancelled'):
        assert step45_auto_recovery_blocker(operator_stop)


def test_legacy_recovery_helpers_are_not_used_by_production_gui():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    ros_source = (
        PROJECT_ROOT / 'piper_gui' / 'ros_node.py'
    ).read_text(encoding='utf-8')
    production_source = source + ros_source
    assert 'prepare_scan_from_current_lock' not in production_source
    assert 'step45_auto_recovery' not in production_source
    assert 'scan_approval' not in production_source
    assert 'MissionActionClient' in ros_source


def test_automatic_scan_is_a_separate_one_start_button_tab():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert 'notebook.add(automatic, text="Automatic Scan")' in source
    automatic = source.split('def _build_automatic_scan', 1)[1].split(
        'def _build_manual', 1)[0]
    assert 'textvariable=self.mission_label_var' in automatic
    assert 'open-vocabulary profile' in automatic
    assert 'rough green-cube coordinate' not in automatic
    assert 'text="Start Complete Automated Scan"' in source
    assert 'command=self.start_automated_scan' in source
    assert 'text="Acquire & Scan"' not in source
    assert 'Production mission API simulator' not in source
    assert 'physical obstacle removal is NOT enabled yet' in source
    assert 'Commissioning: Manual' in source
    assert 'Commissioning: 3D Preview' in source
    assert 'PrepareAcquisition' not in source
    assert 'ApproveScanExecution' not in source
    assert 'RequestTesseractPlan' not in source


def plan(
        kind, plan_id='plan', views=None, qualified=True, valid=True,
        target=(0.45, 0.0, 0.2), source_request_id=''):
    if views is None:
        views = 5 if kind == ROUGH_ACQUISITION else 13
    return SimpleNamespace(
        plan_kind=kind,
        valid=valid,
        planner_backend='tesseract',
        collision_model_qualified=qualified,
        plan_id=plan_id,
        source_request_id=source_request_id,
        trajectory_sha256='a' * 64,
        planned_viewpoints=views,
        target_center=SimpleNamespace(
            x=target[0], y=target[1], z=target[2]),
    )


def test_rough_coordinates_require_three_finite_numbers():
    assert validate_rough_coordinates(('0.45', '0', '0.2')) == (0.45, 0.0, 0.2)
    with pytest.raises(ValueError):
        validate_rough_coordinates(('0.45', 'nan', '0.2'))


def test_automation_speed_is_finite_and_within_sdk_range():
    assert validate_automation_speed('100') == 100.0
    for value in ('nan', '0', '101', 'fast'):
        with pytest.raises(ValueError):
            validate_automation_speed(value)


def test_rootless_worker_keeps_shell_parent_for_bubblewrap_lifetime():
    wrapper = (
        PROJECT_ROOT
        / 'motion_planning'
        / 'tesseract'
        / 'run_worker.sh'
    ).read_text(encoding='utf-8')
    assert 'exec "${ROOTLESS_BWRAP[@]}"' not in wrapper
    assert '"${ROOTLESS_BWRAP[@]}" \\\n' in wrapper
    assert '--die-with-parent' in (
        PROJECT_ROOT
        / 'motion_planning'
        / 'tesseract'
        / 'rootless_common.sh'
    ).read_text(encoding='utf-8')


def test_acquisition_and_scan_require_two_separate_exact_confirmations():
    session = AutomationSession()
    session.prepare_acquisition(
        (0.45, 0.0, 0.2),
        plan(ROUGH_ACQUISITION, plan_id='acquire', views=5),
    )
    session.confirm_acquisition()
    assert 'approval has not been accepted' in session.scan_plan_rejection(
        plan(MULTIVIEW_SCAN))
    session.mark_acquisition_approved()
    assert 'has not been acquired' in session.scan_plan_rejection(
        plan(MULTIVIEW_SCAN))
    session.mark_target_acquired()
    scan = plan(MULTIVIEW_SCAN, plan_id='scan', views=13)
    assert session.scan_plan_rejection(scan) == ''
    session.prepare_scan(scan)
    assert not session.scan_confirmed
    session.confirm_scan(scan)
    assert session.scan_confirmed
    assert 'already consumed' in session.scan_plan_rejection(
        plan(MULTIVIEW_SCAN, plan_id='other'))


def test_unapproved_expired_scan_plan_can_be_discarded_and_retried():
    session = AutomationSession()
    session.prepare_acquisition(
        (0.45, 0.0, 0.2),
        plan(ROUGH_ACQUISITION, plan_id='acquire', views=1),
    )
    session.confirm_acquisition()
    session.mark_acquisition_approved()
    session.mark_target_acquired()
    scan = plan(MULTIVIEW_SCAN, plan_id='scan', views=13)
    session.prepare_scan(scan)

    session.discard_scan_plan()

    assert session.scan_plan_id == ''
    assert session.scan_hash == ''
    assert not session.scan_confirmed
    assert not session.scan_approval_used
    session.prepare_scan(scan)


def test_approved_scan_plan_cannot_be_discarded_for_retry():
    session = AutomationSession()
    session.prepare_acquisition(
        (0.45, 0.0, 0.2),
        plan(ROUGH_ACQUISITION, plan_id='acquire', views=1),
    )
    session.confirm_acquisition()
    session.mark_acquisition_approved()
    session.mark_target_acquired()
    scan = plan(MULTIVIEW_SCAN, plan_id='scan', views=13)
    session.prepare_scan(scan)
    session.confirm_scan(scan)

    with pytest.raises(ValueError, match='approved scan plan'):
        session.discard_scan_plan()


def test_session_rejects_wrong_shape_unqualified_and_changed_scan():
    session = AutomationSession()
    session.prepare_acquisition(
        (0.45, 0.0, 0.2), plan(ROUGH_ACQUISITION, views=1))
    session.confirm_acquisition()
    session.mark_acquisition_approved()
    session.mark_target_acquired()
    assert 'exactly 13' in session.scan_plan_rejection(
        plan(MULTIVIEW_SCAN, views=12))
    assert 'not qualified' in session.scan_plan_rejection(
        plan(MULTIVIEW_SCAN, qualified=False))
    scan = plan(MULTIVIEW_SCAN, plan_id='scan')
    session.prepare_scan(scan)
    with pytest.raises(ValueError, match='changed'):
        session.confirm_scan(plan(MULTIVIEW_SCAN, plan_id='other'))


def test_session_rejects_acquisition_plan_for_different_rough_target():
    session = AutomationSession()
    with pytest.raises(ValueError, match='does not match'):
        session.prepare_acquisition(
            (0.45, 0.0, 0.2),
            plan(ROUGH_ACQUISITION, target=(0.80, 0.0, 0.2)),
        )


def test_session_rejects_plan_from_different_atomic_acquisition_request():
    session = AutomationSession()
    with pytest.raises(ValueError, match='current acquisition session'):
        session.prepare_acquisition(
            (0.45, 0.0, 0.2),
            plan(
                ROUGH_ACQUISITION,
                source_request_id='acq-old-session'),
            acquisition_request_id='acq-current-session',
        )


def test_current_lock_can_be_adopted_before_workflow_assessment():
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.10,
    )
    workflow = {
        'state': 'SCAN_READY',
        'measured_lock_ready': True,
        'measured_lock_rejection': '',
    }
    assert tracking_lock_rejection(
        health, workflow, require_scan_ready=True) == ''
    assert tracking_health_rejection(health) == ''
    assert tracking_lock_rejection(
        health,
        {
            'state': 'IDLE',
            'measured_lock_ready': True,
            'measured_lock_rejection': '',
        },
        require_scan_ready=False,
    ) == ''
    assert 'SCAN_READY' in tracking_lock_rejection(
        health,
        {'state': 'INITIALIZING', 'measured_lock_ready': True},
        require_scan_ready=True,
    )
    assert 'prediction-only' in tracking_lock_rejection(
        SimpleNamespace(
            lifecycle_state='TRACKING',
            camera_settled=True,
            prediction_only=True,
            measurement_age_sec=0.10,
        ),
        workflow,
        require_scan_ready=True,
    )
    assert 'stale' in tracking_lock_rejection(
        SimpleNamespace(
            lifecycle_state='TRACKING',
            camera_settled=True,
            prediction_only=False,
            measurement_age_sec=0.80,
        ),
        workflow,
        require_scan_ready=True,
    )


def test_tracking_health_receipt_must_remain_fresh():
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.10,
    )
    assert tracking_health_rejection(
        health, received_at=9.5, now=10.0) == ''
    assert 'message is stale' in tracking_health_rejection(
        health, received_at=8.0, now=10.0)
    assert 'unavailable' in tracking_lock_rejection(
        health, None, received_at=9.5, now=10.0)
    assert 'measured target lock' in tracking_lock_rejection(
        health,
        {'state': 'SCAN_READY', 'measured_lock_ready': False},
        received_at=9.5,
        now=10.0,
    )


def test_tracking_health_reports_every_step4_enablement_gate():
    assert tracking_health_rejection(None) == 'tracking health is unavailable'
    base = {
        'lifecycle_state': 'TRACKING',
        'camera_settled': True,
        'prediction_only': False,
        'measurement_age_sec': 0.10,
    }
    assert 'not TRACKING' in tracking_health_rejection(
        SimpleNamespace(**dict(base, lifecycle_state='LOST')))
    assert 'not settled' in tracking_health_rejection(
        SimpleNamespace(**dict(base, camera_settled=False)))
    assert 'prediction-only' in tracking_health_rejection(
        SimpleNamespace(**dict(base, prediction_only=True)))
    assert 'measurement is stale' in tracking_health_rejection(
        SimpleNamespace(**dict(base, measurement_age_sec=0.76)))


def test_readiness_mode_cannot_substitute_acquisition_for_multiview():
    readiness = SimpleNamespace(
        worker_ready=True,
        acquisition_ready=True,
        multiview_ready=False,
        acquisition_blockers=[],
        multiview_blockers=[
            'reachable viewpoints are stale',
            'joint state is stale',
            'controller motion limits are stale',
            'tracking is not settled TRACKING',
            'camera timestamp health is not healthy',
            'obstacle scene is blocked: movable clutter',
            'Tesseract worker heartbeat is stale',
        ],
    )
    assert readiness_rejection(
        readiness, 9.5, 10.0, 'acquisition') == ''
    assert readiness_rejection(
        readiness, 9.5, 10.0, 'multiview') == (
            'reachable viewpoints are stale; '
            'joint state is stale; '
            'controller motion limits are stale; '
            'tracking is not settled TRACKING; '
            'camera timestamp health is not healthy; '
            'obstacle scene is blocked: movable clutter; '
            'Tesseract worker heartbeat is stale'
        )
    with pytest.raises(ValueError, match='planning_mode'):
        readiness_rejection(readiness, 9.5, 10.0, 'automatic')
    assert 'has not arrived' in readiness_rejection(
        None, None, 10.0, 'multiview')
    assert 'is stale' in readiness_rejection(
        readiness, 8.0, 10.0, 'multiview')


def test_step4_workflow_actions_are_bounded_and_clutter_stops():
    ready = {'state': 'SCAN_READY', 'measured_lock_ready': True}
    assert step4_workflow_action(ready) == ('ready', '')
    assert step4_workflow_action({
        'state': 'IDLE', 'measured_lock_ready': True,
    }) == ('start', '')
    assert step4_workflow_action({
        'state': 'IDLE', 'measured_lock_ready': False,
    }) == ('start', '')
    assert step4_workflow_action({
        'state': 'IDLE', 'measured_lock_ready': True,
    }, workflow_started=True) == ('wait', '')
    for terminal_state in ('COMPLETE', 'ABORTED'):
        assert step4_workflow_action({
            'state': terminal_state, 'measured_lock_ready': True,
        }) == ('start', '')
    assert step4_workflow_action({
        'state': 'INITIALIZING', 'measured_lock_ready': False,
    }) == ('wait', '')
    action, message = step4_workflow_action({
        'state': 'PLAN_READY', 'measured_lock_ready': True,
    })
    assert action == 'fail'
    assert 'clear the workspace' in message
    action, message = step4_workflow_action({
        'state': 'ABORTED',
        'reason': 'obstacle geometry is invalid',
        'measured_lock_ready': True,
    }, workflow_started=True)
    assert action == 'fail'
    assert message == 'obstacle geometry is invalid'
    assert step4_workflow_action({
        'state': 'WAIT_OPERATOR_ACTION', 'measured_lock_ready': True,
    }) == (
        'fail',
        'supervised workflow is in incompatible active state '
        'WAIT_OPERATOR_ACTION',
    )
    assert step4_workflow_action({}) == (
        'fail', 'supervised workflow diagnostic state is missing')
    assert step4_workflow_action({
        'state': 'SCAN_READY',
        'measured_lock_ready': False,
        'measured_lock_rejection': 'measured lock was already consumed',
    }) == ('fail', 'measured lock was already consumed')


def test_gui_phase_and_timeout_contracts_are_explicit():
    assert [phase.value for phase in AcquisitionPhase] == [
        'IDLE', 'REQUESTING_PREPARE', 'WAITING_PLAN', 'PLAN_READY', 'FAILED']
    assert [phase.value for phase in Step4Phase] == [
        'IDLE', 'STARTING_STACK', 'CHECKING_WORKFLOW',
        'WAITING_SCAN_READY', 'REQUESTING_PLAN', 'WAITING_PLAN',
        'PLAN_READY', 'FAILED']
    assert ACQUISITION_SERVICE_TIMEOUT_SEC == 8.0
    assert ACQUISITION_PLAN_TIMEOUT_SEC == 185.0
    assert WORKFLOW_ASSESSMENT_TIMEOUT_SEC == 15.0
    assert PLAN_REQUEST_QUEUE_TIMEOUT_SEC == 12.0
    assert MULTIVIEW_PLAN_TIMEOUT_SEC == 185.0


def test_gui_joint6_fallback_uses_full_turn_limit():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert '("joint6", -math.pi, math.pi, "rad")' in source


def test_production_gui_has_no_step2_service_or_retry_state():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert '_fresh_acquisition_prepare_client' not in source
    assert 'pending_acquisition_session_id' not in source
    assert 'acquisition_plan_deadline' not in source
    assert '/scan_target_acquisition/prepare' not in source


def test_production_gui_does_not_classify_acquisition_failures():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert 'Acquisition proposal invalidated' not in source
    assert '_acquisition_fail' not in source
    assert 'as_failure(' not in source


def test_production_gui_does_not_approve_executor_plans():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert '/scan_viewpoint_executor/approve' not in source
    assert 'ApproveScanExecution' not in source
    assert 'acquisition_approval' not in source
    assert 'scan_approval' not in source


def test_production_action_owns_enable_and_plan_invalidation():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert 'arm_enable_confirmed' not in source
    assert 'prepare_acquisition' not in source
    assert 'update_automation_buttons' not in source
    automatic = source.split('def start_automated_scan', 1)[1].split(
        'def report_tracked_robot_homed', 1)[0]
    assert 'self.ros_node.submit_mission(request)' in automatic


def test_commissioning_disable_is_direct_but_mission_cancel_still_homes():
    source = (
        PROJECT_ROOT / 'piper_gui' / 'native_app.py'
    ).read_text(encoding='utf-8')
    assert 'command=self.request_safe_disable' in source
    safe_disable = source.split(
        'def request_safe_disable(self) -> None:', 1)[1].split(
            'def use_feedback(self) -> None:', 1)[0]
    assert 'self.ros_node.cancel_scan()' not in safe_disable
    assert 'Cancel and Home' in safe_disable
    assert 'self.ros_node.publish_joint_target' not in safe_disable
    assert 'self.fresh_feedback()' not in safe_disable
    assert 'self.ros_node.call_enable_async(False)' in safe_disable
    assert 'Commissioning disable requested directly' in safe_disable

    executor = (
        PROJECT_ROOT
        / 'piper_ros_foxy'
        / 'src'
        / 'piper_mobile_manipulation'
        / 'piper_mobile_manipulation'
        / 'scan_viewpoint_executor_node.py'
    ).read_text(encoding='utf-8')
    inactive_cancel = executor.split(
        'def cancel_cb(self, _request, response):', 1)[1].split(
            'def refresh_cb(self, _request, response):', 1)[0]
    assert 'held = self.publish_hold()' in inactive_cancel
    assert 'dedicated current-state return-home replanning' in inactive_cancel
    assert 'self.try_start_abort_return(' not in inactive_cancel
    assert 'approved-path retrace to plan start started' not in inactive_cancel
    assert 'current joint hold requested before' in inactive_cancel
    assert 'current joint hold requested' in inactive_cancel

    assert 'text="Cancel and Home"' in source


def test_explicit_current_lock_adoption_recovers_terminal_acquisition():
    session = AutomationSession()
    session.prepare_acquisition(
        (0.45, 0.0, 0.2),
        plan(ROUGH_ACQUISITION, plan_id='acquire', views=1),
    )
    session.confirm_acquisition()
    session.mark_acquisition_approved()
    session.finish('time-aligned following error exceeded the sustained limit')
    session.adopt_current_lock()

    assert not session.ended
    assert session.current_lock_adopted
    assert session.target_acquired
    assert session.scan_plan_rejection(
        plan(MULTIVIEW_SCAN, plan_id='scan', views=13)) == ''


def test_plan_request_correlation_requires_full_worker_request_id():
    assert plan_matches_request(
        plan(
            MULTIVIEW_SCAN,
            plan_id='0123456789abcdef0123456789abcdef'),
        '0123456789abcdef0123456789abcdef',
    )
    assert not plan_matches_request(
        plan(MULTIVIEW_SCAN, plan_id='0123456789abcdef'),
        '0123456789abcdef0123456789abcdef',
    )
    assert not plan_matches_request(
        plan(MULTIVIEW_SCAN, plan_id='fedcba9876543210'),
        '0123456789abcdef0123456789abcdef',
    )
