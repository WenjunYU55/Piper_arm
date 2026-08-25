"""ROS-facing operations adapter consumed by the pure mission engine."""

import math
import time

from piper_mobile_manipulation.configuration import configured_value
from piper_mobile_manipulation.scan_motion import motor_driver_states


NON_COMMAND_PROCESS_GROUPS = (
    'vision',
    'hand_eye',
    'tesseract_worker',
)


def previous_generation_cleanup_targets(
        live_processes, all_motors_disabled):
    """Select exact stale handles that are safe to stop before admission."""
    live = tuple(dict.fromkeys(str(name) for name in live_processes))
    if 'driver' in live and not bool(all_motors_disabled):
        return ()
    return live


class MissionNodeOperations:
    """Translate pure MissionEngine operations to the existing ROS node."""

    def __init__(self, node, goal_handle, cancellation, rough_target=None):
        self.node = node
        self.goal_handle = goal_handle
        self.cancellation = cancellation
        self.rough_target = rough_target

    @property
    def session(self):
        """Return the currently bound session for compatibility diagnostics."""
        return getattr(self.node, '_active_engine_session', None)

    def begin_process_generation(self, _context):
        live = self.node.processes.begin_generation()
        if not live:
            return []
        cleanup_targets = previous_generation_cleanup_targets(
            live, self.node.fresh_all_motors_disabled())
        if not cleanup_targets:
            self.node.get_logger().error(
                'retaining previous process generation because its driver is '
                'live and fresh six-disabled feedback is not proved: %s'
                % ', '.join(live))
            return live
        self.node.get_logger().warn(
            'cleaning exact previous-generation process handles before new '
            'mission admission: %s' % ', '.join(cleanup_targets))
        try:
            report = self.node.processes.shutdown(cleanup_targets)
        except Exception as error:
            self.node.get_logger().error(
                'previous-generation cleanup raised %s: %s'
                % (type(error).__name__, error))
            return live
        if not report.complete:
            self.node.get_logger().error(
                'previous-generation cleanup remains incomplete: %s'
                % ', '.join(report.still_running))
            return list(report.still_running)
        return self.node.processes.begin_generation()

    def snapshot_target(self, _context):
        return self.node.snapshot_target(self.rough_target)

    def transition(self, context, phase, reason):
        self.node.transition(self.goal_handle, context.session, phase, reason)

    def progress(self, context, reason):
        self.node.startup_progress(
            self.goal_handle, context.session, reason)

    def selected_home_profile(self, _context):
        return self.node.selected_home_profile()

    def bind_home_profile(self, _context, profile):
        self.node.current_home_profile = profile

    def current_home_profile(self, _context):
        return self.node.current_home_profile

    def start_processes(self, context):
        self.node.start_processes(self.goal_handle, context.session)

    def wait_for_enable_service(self, context, timeout):
        self.node.wait_for(
            self.goal_handle, context.session,
            lambda: self.node.enable_client.service_is_ready(), timeout,
            'PiPER enable service did not become ready')

    def wait_for_stable_readiness(
            self, context, mode, stable_sec, timeout_sec):
        self.node.wait_for_stable_readiness(
            self.goal_handle, context.session, mode, stable_sec, timeout_sec)

    def wait_for_stable_joint_stream(
            self, context, stable_sec, timeout_sec, label):
        self.node.wait_for_stable_joint_stream(
            stable_sec, timeout_sec, label,
            self.goal_handle, context.session)

    def require_fresh_joint_feedback(self, _context):
        self.node.require_fresh_joint_feedback()

    def current_joint_positions(self, _context):
        return self.node.latest_joints.position[:6]

    def boolean_option(self, _context, name):
        return self.node.param_bool(name)

    def numeric_option(self, _context, name):
        return configured_value(self.node, name)

    def authorize_mission(self, context, revoke=False):
        return self.node.authorize_mission(context.session, revoke=revoke)

    def enable_arm(self, _context, enabled):
        return self.node.call_enable(enabled)

    def arm_enable_guard_started(self, _context):
        self.node.motor_enable_guard_after = time.monotonic() + 0.5

    def prove_current_hold(self, context):
        return self.node.prove_current_hold(
            self.goal_handle, context.session)

    def hold_diagnostic(self, _context):
        return str(self.node.last_hold_diagnostic)

    def prove_home(
            self, context, startup=False, target_positions=None,
            home_stage='ROUGH_HOME', interruptible=False):
        kwargs = {}
        if startup:
            kwargs['startup'] = True
        if interruptible:
            kwargs['goal_handle'] = self.goal_handle
        if target_positions is not None:
            kwargs['target_positions'] = target_positions
        if str(home_stage) != 'ROUGH_HOME':
            kwargs['home_stage'] = home_stage
        return self.node.prove_return_home_for_shutdown(
            context.session, **kwargs)

    def return_home_diagnostic(self, _context):
        return str(getattr(self.node, 'last_return_home_diagnostic', ''))

    def clear_plan_cache(self, _context):
        return self.node.clear_plan_cache()

    def prepare_acquisition(self, context):
        return self.node.prepare_acquisition(
            context.session, context.target)

    def wait_for_plan(self, context, kind, request_id, timeout):
        return self.node.wait_for_plan(
            self.goal_handle, context.session, kind, request_id, timeout)

    def approve_plan(self, context, plan):
        return self.node.approve_plan(
            self.goal_handle, context.session, plan)

    def wait_for_execution(
            self, context, successes, timeout, failures):
        return self.node.wait_for_execution(
            self.goal_handle, context.session, successes, timeout, failures)

    def start_and_wait_workflow(self, context):
        return self.node.start_and_wait_workflow(
            self.goal_handle, context.session)

    def readiness_rejection(self, _context, mode):
        return self.node.readiness_rejection(mode)

    def capture_count(self, _context):
        return int(self.node.latest_capture.get('captured_frame_count', 0))

    def current_feature_coverage(self, _context):
        return self.node.current_scan_feature_coverage()

    def request_multiview_plan(self, context):
        return self.node.request_multiview_plan(
            self.goal_handle, context.session)

    def wait_for_view_generation(self, context, accepted_views, timeout):
        return self.node.wait_for_view_generation(
            self.goal_handle, context.session, accepted_views, timeout)

    def remaining_time(self, context):
        return context.session.remaining()

    def wait_for_scan_history(self, context, timeout):
        self.node.wait_for(
            self.goal_handle, context.session,
            lambda: int((self.node.latest_scan_history or {}).get(
                'accepted_views', 0)) >= context.session.accepted_captures,
            timeout,
            'scan history did not catch up with the final accepted capture')

    def wait_for_all_motors_disabled(self, _context, timeout):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            telemetry_store = getattr(self.node, 'telemetry_store', None)
            if telemetry_store is None:
                status = self.node.latest_arm_status
                age = time.monotonic() - self.node.latest_arm_status_at
            else:
                snapshot = telemetry_store.snapshot()
                observation = snapshot.arm.status
                status = None if observation is None else observation.value
                age = (
                    math.inf if observation is None else
                    observation.age_at(snapshot.captured_at))
            if (
                    status is not None
                    and age <= 0.5
                    and bool(getattr(status, 'motor_feedback_valid', False))
                    and not any(motor_driver_states(status))):
                return True
            time.sleep(0.02)
        return False

    def stop_processes(self, _context):
        return self.node.processes.stop_all()

    def stop_processing_processes(self, _context):
        try:
            report = self.node.processes.shutdown(NON_COMMAND_PROCESS_GROUPS)
        except Exception as error:
            self.node.get_logger().error(
                'non-command process cleanup raised %s: %s'
                % (type(error).__name__, error))
            return False
        return report.complete

    def prove_shutdown_hold(self, context):
        return self.node.prove_current_hold_for_shutdown(context.session)
