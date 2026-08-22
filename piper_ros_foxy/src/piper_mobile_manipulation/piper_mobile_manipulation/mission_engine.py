"""Pure application orchestration for one bounded target-scan mission."""

from dataclasses import dataclass, field
import threading
import time

from piper_mobile_manipulation.configuration import (
    MissionCaptureConfig,
    MissionMotionConfig,
    MissionWorkflowConfig,
)
from piper_mobile_manipulation.failure_model import (
    as_failure,
    Failure,
    FailureTag,
    legacy_failure_adapter,
)
from piper_mobile_manipulation.home_pose import (
    staged_home_targets,
    validate_staged_wrist_direction,
)
from piper_mobile_manipulation.mission_core import (
    MissionPhase,
    REQUIRED_CAPTURES,
)


_WORKFLOW_DEFAULTS = MissionWorkflowConfig()
ACQUISITION_SERVICE_TIMEOUT_SEC = \
    _WORKFLOW_DEFAULTS.acquisition_service_timeout_sec
WORKFLOW_ASSESSMENT_TIMEOUT_SEC = \
    _WORKFLOW_DEFAULTS.workflow_assessment_timeout_sec
PLAN_REQUEST_QUEUE_TIMEOUT_SEC = \
    _WORKFLOW_DEFAULTS.plan_request_queue_timeout_sec
PLAN_RESULT_TIMEOUT_SEC = _WORKFLOW_DEFAULTS.plan_result_timeout_sec
PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC = \
    _WORKFLOW_DEFAULTS.plan_approval_transient_timeout_sec
SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC = \
    _WORKFLOW_DEFAULTS.scan_visual_reacquisition_timeout_sec
MAX_SCAN_QUALITY_REPLANS = _WORKFLOW_DEFAULTS.max_scan_quality_replans
MAX_SCAN_TARGET_DRIFT_REPLANS = \
    _WORKFLOW_DEFAULTS.max_scan_target_drift_replans


class MissionFailure(RuntimeError):
    """Typed application failure retained at the legacy exception boundary."""

    def __init__(self, reason, needs_operator=False, outcome='FAILED',
                 failure_code='', retryable=None, failure=None):
        typed = failure
        if typed is None and isinstance(reason, Failure):
            typed = reason
        if typed is None:
            typed = legacy_failure_adapter(
                reason,
                code=(failure_code or None),
                retryable=retryable,
                needs_operator=needs_operator,
                outcome=outcome,
            )
        elif not isinstance(reason, Failure) and str(reason) != typed.detail:
            typed = typed.with_detail(str(reason))
        super().__init__(typed.detail)
        self.failure = typed
        self.needs_operator = typed.needs_operator
        self.outcome = typed.outcome
        self.failure_code = typed.code.value
        self.retryable = typed.retryable


class CancellationToken:
    """Thread-safe application cancellation state with a stable reason."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancelled = False
        self._reason = ''

    def cancel(self, reason='tracked robot cancelled the task'):
        """Request cancellation once without replacing its first cause."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                self._reason = str(reason)

    @property
    def cancelled(self):
        """Return whether cancellation has been requested."""
        with self._lock:
            return bool(self._cancelled)

    @property
    def reason(self):
        """Return the first cancellation reason."""
        with self._lock:
            return str(self._reason)


@dataclass
class MissionContext:
    """Mutable state belonging exclusively to one engine invocation."""

    session: object
    cancellation: CancellationToken
    target: object = None
    owns_process_generation: bool = False
    phase_sequence: list = field(default_factory=list)


@dataclass(frozen=True)
class MissionResult:
    """Application outcome before conversion to the public ROS result."""

    outcome: str
    reason: str
    failure: object
    phase_sequence: tuple
    owns_process_generation: bool

    @property
    def succeeded(self):
        """Return whether the application mission and shutdown succeeded."""
        return self.failure is None and self.outcome == 'SUCCEEDED'


def failure_code_for_reason(reason):
    """Return one stable tracked-robot failure code for legacy exceptions."""
    return as_failure(reason).code.value


def retryable_plan_approval_rejection(reason):
    """Retry only an unchanged plan's transient live-state gate."""
    return as_failure(reason).has(FailureTag.PLAN_APPROVAL_RETRY)


def visual_reacquisition_plan_approval_rejection(reason):
    """Recognize a no-motion scan approval wait for a measured lock."""
    return as_failure(reason).has(
        FailureTag.PLAN_APPROVAL_VISUAL_REACQUISITION)


def visual_reacquisition_plan_request_rejection(reason):
    """Recognize a command-free planning snapshot that lacks a lock."""
    return as_failure(reason).has(
        FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION)


def runtime_freshness_plan_request_rejection(reason):
    """Recognize a command-free planning snapshot with transient telemetry."""
    return as_failure(reason).has(FailureTag.RUNTIME_FRESHNESS_GAP)


def shutdown_uses_startup_home(session):
    """Use static home authority only before perception owns a scene."""
    return bool(
        not getattr(session, 'perception_scene_established', False)
        and int(getattr(session, 'accepted_captures', 0)) == 0)


def target_drift_requires_replan(reason):
    """Recognize the executor's no-motion stale-target-plan rejection."""
    return as_failure(reason).has(FailureTag.TARGET_DRIFT_REPLAN)


def safe_view_exhaustion_after_capture(
        reason, accepted_captures, feature_coverage=None):
    """Recognize a proved end of the adaptive safe-view frontier."""
    accepted = int(accepted_captures)
    proved_accepted = int(
        feature_coverage.get('accepted_achieved_views', 0)
        if isinstance(feature_coverage, dict) else 0)
    return (
        accepted >= 1
        and isinstance(feature_coverage, dict)
        and proved_accepted >= accepted
        and as_failure(reason).has(FailureTag.SAFE_VIEW_EXHAUSTED)
    )


def feature_capture_decision(
        accepted_captures, required_captures, maximum_captures,
        feature_coverage):
    """Choose continue, complete, or exhausted from achieved feature proof."""
    accepted = int(accepted_captures)
    required = int(required_captures)
    maximum = int(maximum_captures)
    if required < 1 or maximum < required or accepted < 0:
        raise ValueError('feature capture bounds are invalid')
    sufficient = bool(
        isinstance(feature_coverage, dict)
        and feature_coverage.get('sufficient') is True)
    if accepted >= required and sufficient:
        return 'COMPLETE'
    if accepted >= maximum:
        return 'EXHAUSTED'
    return 'CONTINUE'


def planning_rejection_allows_current_state_home(reason):
    """Identify failures that may be re-qualified by a fresh home plan."""
    return as_failure(reason).has(FailureTag.PLAN_REJECTION_HOME_ALLOWED)


class MissionEngine:
    """
    Run the existing mission workflow through injected operations.

    ``operations`` is deliberately a straightforward application adapter, not
    a hierarchy of subsystem base classes.  Production supplies the ROS node
    adapter; tests supply a small fake.  The engine itself imports no ROS code.
    """

    PIPELINE_HANDLERS = (
        MissionPhase.STARTING,
        MissionPhase.PREFLIGHT,
        MissionPhase.ENABLE_AND_HOLD,
        MissionPhase.RETURNING_HOME,
        MissionPhase.ROUGH_ACQUISITION,
        MissionPhase.OCCLUSION_PROBE,
        MissionPhase.VIEW_PLANNING,
    )

    def __init__(
            self, operations, clock=time.monotonic, motion_config=None,
            capture_config=None, workflow_config=None):
        self.operations = operations
        self.clock = clock
        # Production supplies immutable typed configuration. The fallback
        # preserves Phase 1-7 pure test doubles that expose the former option
        # adapter without constructing a ROS node.
        self.motion_config = motion_config or MissionMotionConfig(
            enable_real_arm_motion=bool(operations.boolean_option(
                None, 'enable_real_arm_motion')),
            motion_speed_profile_qualified=bool(operations.boolean_option(
                None, 'motion_speed_profile_qualified')),
            free_motion_speed_percent=float(operations.numeric_option(
                None, 'free_motion_speed_percent')),
            contact_speed_percent=float(operations.numeric_option(
                None, 'contact_speed_percent')),
            home_pose_path='',
            require_staged_home_profile=True,
        )
        self.capture_config = capture_config or MissionCaptureConfig(
            required_captures=int(operations.numeric_option(
                None, 'required_captures')),
            maximum_captures=int(operations.numeric_option(
                None, 'maximum_captures')),
        )
        self.workflow_config = workflow_config or MissionWorkflowConfig()
        self.handlers = {
            MissionPhase.STARTING: self._handle_starting,
            MissionPhase.PREFLIGHT: self._handle_preflight,
            MissionPhase.ENABLE_AND_HOLD: self._handle_enable_and_hold,
            MissionPhase.RETURNING_HOME: self._handle_startup_home,
            MissionPhase.ROUGH_ACQUISITION: self._handle_acquisition,
            MissionPhase.OCCLUSION_PROBE: self._handle_occlusion_probe,
            MissionPhase.VIEW_PLANNING: self._handle_scan,
        }

    def execute(self, context):
        """Run one admitted mission and its established terminal policy."""
        session = context.session
        failure = None
        try:
            live_processes = self.operations.begin_process_generation(context)
            if live_processes:
                raise MissionFailure(
                    'previous mission still owns live process groups: %s'
                    % ', '.join(live_processes),
                    needs_operator=True,
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
            context.owns_process_generation = True
            if context.target is None:
                context.target = self.operations.snapshot_target(context)
            self.run_pipeline(context)
        except MissionFailure as exc:
            failure = exc
        except Exception as exc:
            failure = MissionFailure('mission exception: %s' % exc)

        if failure is None:
            shutdown_failure = self.shutdown(
                context, normal_completion=True)
            if shutdown_failure is not None:
                failure = shutdown_failure
        elif context.owns_process_generation:
            shutdown_failure = self.shutdown(
                context, normal_completion=False, failure=failure)
            if shutdown_failure is not None:
                failure = MissionFailure(
                    '%s; safe shutdown also failed: %s'
                    % (failure, shutdown_failure),
                    needs_operator=True,
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
            elif failure.outcome == 'CANCELLED':
                failure = MissionFailure(
                    'task failed: cancelled; arm returned to configured home, '
                    'disabled, and pipeline stopped; please retry',
                    outcome='CANCELLED', failure_code='CANCELLED',
                    retryable=True)

        if failure is None:
            session.phase = MissionPhase.SUCCEEDED
            session.reason = (
                'distinctive-feature target scan completed with %d accepted '
                'diverse views and PiPER shut down safely'
                % session.accepted_captures)
            outcome = 'SUCCEEDED'
        else:
            session.phase = (
                MissionPhase.NEEDS_OPERATOR
                if failure.needs_operator else MissionPhase.FAILED)
            session.reason = str(failure)
            outcome = (
                'NEEDS_OPERATOR' if failure.needs_operator
                else failure.outcome)
        return MissionResult(
            outcome=outcome,
            reason=session.reason,
            failure=failure,
            phase_sequence=tuple(context.phase_sequence),
            owns_process_generation=context.owns_process_generation,
        )

    def run_pipeline(self, context):
        """Dispatch the existing sequence through small phase handlers."""
        for phase in self.PIPELINE_HANDLERS:
            self.handlers[phase](context)

    def _transition(self, context, phase, reason):
        self.operations.transition(context, phase, reason)
        context.phase_sequence.append(MissionPhase(phase).value)

    def _handle_starting(self, context):
        session = context.session
        profile = self.operations.selected_home_profile(context)
        self.operations.bind_home_profile(context, profile)
        session.home_positions_rad = tuple(profile['positions_rad'])
        session.pre_home_positions_rad = tuple(
            profile.get('pre_home_positions_rad', ()))
        session.mission_ready_joint6_rad = float(
            profile['mission_ready_joint6_rad'])
        session.storage_joint6_rad = float(profile['storage_joint6_rad'])
        session.storage_positions_rad = tuple(
            list(session.home_positions_rad[:5])
            + [session.storage_joint6_rad])
        self._transition(
            context, MissionPhase.STARTING,
            'starting PiPER-owned process groups')
        self.operations.start_processes(context)
        self.operations.progress(
            context,
            'scan stack started; waiting for typed acquisition readiness')
        self.operations.wait_for_enable_service(
            context, self.workflow_config.enable_service_timeout_sec)
        self.operations.wait_for_stable_readiness(
            context, 'acquisition',
            self.workflow_config.startup_readiness_stable_sec,
            self.workflow_config.startup_readiness_timeout_sec)
        self.operations.progress(
            context,
            'acquisition ready; proving final settled joint feedback')
        self.operations.wait_for_stable_joint_stream(
            context, self.workflow_config.startup_joint_stable_sec,
            self.workflow_config.startup_joint_timeout_sec,
            'pre-enable joint feedback')

    def _handle_preflight(self, context):
        profile = self.operations.current_home_profile(context)
        self._transition(
            context, MissionPhase.PREFLIGHT,
            'validating current feedback and mission authority')
        self.operations.require_fresh_joint_feedback(context)
        try:
            validate_staged_wrist_direction(
                profile, self.operations.current_joint_positions(context))
        except (TypeError, ValueError) as exc:
            raise MissionFailure(
                'configured startup wrist direction is unsafe: %s; arm '
                'remained disabled' % exc,
                failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
        if not self.motion_config.enable_real_arm_motion:
            raise MissionFailure(
                'mission node is proposal-only; real arm motion was not enabled')
        if not self.motion_config.motion_speed_profile_qualified:
            raise MissionFailure(
                'configured %.1f-percent transit and %.1f-percent contact '
                'speed profile is not physically qualified; arm remained '
                'disabled'
                % (
                    self.motion_config.free_motion_speed_percent,
                    self.motion_config.contact_speed_percent))
        self.operations.authorize_mission(context)

    def _handle_enable_and_hold(self, context):
        session = context.session
        self._transition(
            context, MissionPhase.ENABLE_AND_HOLD,
            'enabling arm; controller retains its current joint target')
        self.operations.enable_arm(context, True)
        session.arm_enabled = True
        self.operations.arm_enable_guard_started(context)
        # PiPER position control holds the current controller target when the
        # motors enable.  The next direct startup-home transaction re-reads
        # fresh joints and live all-six motor state before sending its first
        # endpoint, so a second hold service is not a readiness authority.
        session.current_hold_proved = True

    def _handle_startup_home(self, context):
        session = context.session
        targets = staged_home_targets(
            self.operations.current_home_profile(context),
            self.operations.current_joint_positions(context))
        self._transition(
            context, MissionPhase.RETURNING_HOME,
            'rotating J6 from the measured powered start to the configured '
            'mission-ready wrist angle')
        if not self.operations.prove_home(
                context, startup=True,
                target_positions=targets['startup_wrist_positions_rad'],
                home_stage='STARTUP_WRIST', interruptible=True):
            raise MissionFailure(
                'startup wrist normalization was not proved; arm remains in '
                'a current-position hold: '
                + self.operations.return_home_diagnostic(context),
                True, failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
        session.startup_wrist_completed = True
        session.return_home_proved = False
        self._transition(
            context, MissionPhase.RETURNING_HOME,
            'normalizing joints 1-6 to the configured rough mission home')
        if not self.operations.prove_home(
                context, startup=True,
                target_positions=targets['rough_home_positions_rad'],
                home_stage='ROUGH_HOME', interruptible=True):
            raise MissionFailure(
                'startup configured-home normalization was not proved; '
                'arm remains in a current-position hold: '
                + self.operations.return_home_diagnostic(context),
                True, failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
        session.startup_home_completed = True

    def _handle_acquisition(self, context):
        session = context.session
        self._transition(
            context, MissionPhase.ROUGH_ACQUISITION,
            'starting closed-loop rough-target acquisition')
        acquired = False
        maximum_looks = self.workflow_config.acquisition_max_looks
        for look_index in range(maximum_looks):
            session.acquisition_attempt = look_index + 1
            self.operations.clear_plan_cache(context)
            request_id = self.operations.prepare_acquisition(context)
            plan = self.operations.wait_for_plan(
                context, 'ROUGH_ACQUISITION', request_id,
                self.workflow_config.plan_result_timeout_sec)
            self.operations.approve_plan(context, plan)
            self._transition(
                context, MissionPhase.TARGET_LOCK,
                'settling and measuring acquisition look %d/%d'
                % (look_index + 1, maximum_looks))
            execution = self.operations.wait_for_execution(
                context,
                (
                    'ACQUIRED', 'ACQUISITION_LOOK_COMPLETE',
                    'ACQUISITION_TARGET_TOO_FAR',
                ),
                self.workflow_config.acquisition_execution_timeout_sec,
                ('ACQUISITION_FAILED', 'ABORTED', 'INVALID'))
            session.perception_scene_established = True
            if str(execution.state) == 'ACQUISITION_TARGET_TOO_FAR':
                target = [float(value) for value in context.target]
                raise MissionFailure(
                    'target is outside the qualified sensor depth at rough '
                    'base_link coordinates x=%.3f, y=%.3f, z=%.3f: %s'
                    % (
                        target[0], target[1], target[2],
                        str(execution.reason)),
                    outcome='REPOSITION_REQUIRED',
                    failure_code='TARGET_TOO_FAR', retryable=True)
            if str(execution.state) == 'ACQUIRED':
                acquired = True
                break
            self._transition(
                context, MissionPhase.ROUGH_ACQUISITION,
                'target absent after look %d/%d; replanning once from fresh '
                'measured arm and camera state'
                % (look_index + 1, maximum_looks))
        if not acquired:
            target = [float(value) for value in context.target]
            raise MissionFailure(
                'target not found after %d distinct closed-loop looks near '
                'rough base_link coordinates x=%.3f, y=%.3f, z=%.3f'
                % (
                    maximum_looks, target[0], target[1], target[2]),
                failure_code='TARGET_NOT_FOUND', retryable=True)

    def _handle_occlusion_probe(self, context):
        self._transition(
            context, MissionPhase.OCCLUSION_PROBE,
            'assessing the measured target and occluder scene')
        workflow = self.operations.start_and_wait_workflow(context)
        if workflow.get('state') == 'PLAN_READY':
            readiness = self.operations.readiness_rejection(
                context, 'manipulation')
            raise MissionFailure(
                'beneficial occluder removal is required but unavailable: '
                + (readiness or 'contact planner is not implemented'),
                needs_operator=True)
        self.operations.progress(
            context,
            'measured target lock ready; waiting for stable multiview readiness')
        self.operations.wait_for_stable_readiness(
            context, 'multiview',
            self.workflow_config.multiview_readiness_stable_sec,
            self.workflow_config.multiview_readiness_timeout_sec)

    def _handle_scan(self, context):
        session = context.session
        quality_replans = 0
        target_drift_replans = 0
        execution = None
        required = self.capture_config.required_captures
        maximum = self.capture_config.maximum_captures
        if required < 1 or maximum < required:
            raise MissionFailure(
                'capture bounds are invalid: minimum %d maximum %d'
                % (required, maximum),
                failure_code='MISSION_FAILED', retryable=False)
        adaptive_completion = False
        while True:
            accepted = self.operations.capture_count(context)
            coverage = self.operations.current_feature_coverage(context)
            history_count = int(coverage.get('accepted_achieved_views', 0))
            decision = feature_capture_decision(
                accepted, required, maximum,
                coverage if history_count >= accepted else {})
            if decision == 'COMPLETE':
                session.accepted_captures = accepted
                adaptive_completion = True
                self.operations.progress(
                    context,
                    'distinctive feature floors are complete after %d '
                    'accepted views; holding for a fresh current-state home '
                    'plan' % accepted)
                break
            if decision == 'EXHAUSTED':
                raise MissionFailure(
                    '%d-view bounded scan limit reached but distinctive '
                    'feature coverage remained insufficient: %s'
                    % (accepted, '; '.join(coverage.get('blockers', []))),
                    failure_code='INSUFFICIENT_CAPTURE_QUALITY',
                    retryable=True)
            remaining = maximum - accepted
            self._transition(
                context, MissionPhase.VIEW_PLANNING,
                'requesting one correlated feature-driven view; up to %d '
                'bounded views remain (model seed floor %d)'
                % (remaining, required))
            self.operations.clear_plan_cache(context)
            self.operations.wait_for_view_generation(
                context, accepted,
                self.workflow_config.plan_result_timeout_sec)
            try:
                request_id = self.operations.request_multiview_plan(context)
                plan = self.operations.wait_for_plan(
                    context, 'MULTIVIEW_SCAN', request_id,
                    self.workflow_config.plan_result_timeout_sec)
            except MissionFailure as exc:
                coverage = self.operations.current_feature_coverage(context)
                if as_failure(exc).has(
                        FailureTag.RAY_SHORTLIST_EXHAUSTED):
                    self.operations.progress(
                        context,
                        'informative ray shortlist was infeasible; '
                        'requesting the next untried bounded ray shortlist')
                    continue
                if not safe_view_exhaustion_after_capture(
                        exc, accepted, coverage):
                    if (
                            as_failure(exc).has(FailureTag.EMPTY_VIEW_FRONTIER)
                            and coverage.get('blockers')):
                        raise MissionFailure(
                            '%s; safe-view frontier ended before distinctive '
                            'feature coverage was sufficient: %s'
                            % (exc, '; '.join(coverage['blockers'])),
                            failure_code='INSUFFICIENT_CAPTURE_QUALITY',
                            retryable=True)
                    raise
                session.accepted_captures = accepted
                adaptive_completion = True
                self.operations.progress(
                    context,
                    'adaptive scan complete after %d accepted diverse views; '
                    'Tesseract proved no meaningfully different collision-free '
                    'view remains' % accepted)
                break
            try:
                self.operations.approve_plan(context, plan)
            except MissionFailure as exc:
                if (
                        not target_drift_requires_replan(exc)
                        or target_drift_replans
                        >= self.workflow_config.max_scan_target_drift_replans):
                    raise
                target_drift_replans += 1
                self.operations.progress(
                    context,
                    'measured target changed after planning; no motion was '
                    'authorized, replanning from the fresh lock (%d/%d)'
                    % (target_drift_replans,
                       self.workflow_config.max_scan_target_drift_replans))
                # The next loop is command-free until plan approval. Let the
                # plan-request adapter own its existing bounded visual
                # reacquisition path instead of failing first in a duplicate
                # generic readiness wait.
                continue
            self._transition(
                context, MissionPhase.CAPTURING,
                'executing one settled quality-gated viewpoint')
            execution = self.operations.wait_for_execution(
                context,
                ('VIEW_COMPLETE', 'VIEW_REJECTED', 'COMPLETE'),
                self.operations.remaining_time(context),
                ('ABORTED', 'INVALID'))
            if str(execution.state) == 'COMPLETE':
                break
            session.accepted_captures = self.operations.capture_count(context)
            if str(execution.state) == 'VIEW_REJECTED':
                if quality_replans >= \
                        self.workflow_config.max_scan_quality_replans:
                    raise MissionFailure(
                        'visual replacement budget exhausted: '
                        + str(execution.reason))
                quality_replans += 1
                self.operations.progress(
                    context,
                    'view rejected by fresh visual gates; executor is holding, '
                    'excluding that pose and replanning (%d/%d)'
                    % (quality_replans,
                       self.workflow_config.max_scan_quality_replans))
                # Re-enter the command-free plan request. Its typed visual
                # rejection path waits for the existing bounded SAM2/heavy
                # reacquisition before any proposal can be approved.
                continue
            self.operations.progress(
                context,
                'accepted view %d (minimum %d, bounded maximum %d); '
                'replanning one next view from measured pose and achieved '
                'feature coverage'
                % (session.accepted_captures, required, maximum))
            # Planning is command-free and the request/approval boundaries
            # already own visual reacquisition plus every motion-safety gate.
            # Avoid a second readiness owner between accepted NBV iterations.
        session.accepted_captures = self.operations.capture_count(context)
        if adaptive_completion:
            return
        if not required <= session.accepted_captures <= maximum:
            raise MissionFailure(
                'executor completed with %d captures outside the bounded '
                '%d-%d contract'
                % (session.accepted_captures, required, maximum))
        if not as_failure(execution.reason).has(FailureTag.HOME_REACHED):
            raise MissionFailure(
                'captures completed but return-home was not proved: %s'
                % execution.reason)
        session.return_home_proved = True
        self.operations.wait_for_scan_history(
            context, self.workflow_config.final_history_timeout_sec)
        coverage = self.operations.current_feature_coverage(context)
        if not coverage.get('sufficient'):
            raise MissionFailure(
                '%d captures completed but distinctive feature coverage was '
                'insufficient: %s'
                % (session.accepted_captures,
                   '; '.join(coverage.get('blockers', []))),
                failure_code='INSUFFICIENT_CAPTURE_QUALITY', retryable=True)

    def shutdown(self, context, normal_completion, failure=None):
        """Run the existing fail-closed home/hold/disable/cleanup sequence."""
        session = context.session
        try:
            if session.motor_control_lost_reason:
                if not self.operations.wait_for_all_motors_disabled(
                        context,
                        self.workflow_config.motor_disabled_proof_timeout_sec):
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'motor control was lost and six-disabled feedback '
                            'was not proved; automatic home was forbidden and '
                            'the driver remains available for operator '
                            'recovery', True))
                session.disabled_proved = True
                session.arm_enabled = False
                self._revoke_ignoring_failure(context)
                self._transition(
                    context, MissionPhase.STOPPING,
                    'motor watchdog proved all six axes disabled; skipping '
                    'automatic home and stopping mission-owned processes')
                session.processes_stopped = self.operations.stop_processes(
                    context)
                if not session.processes_stopped:
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'motor control was lost; all axes are disabled but '
                            'one or more PiPER-owned processes remain alive',
                            True))
                return MissionFailure(
                    'motor control was lost before configured home; no home '
                    'command was attempted, all six motors are disabled, and '
                    'mission-owned processes are stopped: '
                    + session.motor_control_lost_reason, True)
            if not session.arm_enabled:
                session.current_hold_proved = True
                session.pre_home_completed = True
                session.return_home_proved = True
                session.storage_wrist_proved = True
                session.disabled_proved = True
                self._revoke_ignoring_failure(context)
                self._transition(
                    context, MissionPhase.STOPPING,
                    'stopping never-enabled PiPER process groups')
                session.processes_stopped = self.operations.stop_processes(
                    context)
                if not session.processes_stopped:
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'one or more PiPER-owned processes remain alive',
                            True))
                return None
            if not session.pre_home_completed:
                # The failure that ended scanning is not authority for the
                # shutdown motion.  Re-qualify the dedicated direct-home
                # request from current telemetry instead.  Confirmed motor
                # authority loss is handled above and remains the sole
                # reason to skip all automatic home commands.
                self._transition(
                    context, MissionPhase.RETURNING_HOME,
                    'cancellation/failure entered terminal recovery; requesting '
                    'the configured direct pre-home joint target from current '
                    'feedback; camera-holder floor/external clearance remains '
                    'mandatory')
                startup_home = shutdown_uses_startup_home(session)
                pre_home_target = list(session.pre_home_positions_rad)
                if len(pre_home_target) != 6:
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'pre-home target is missing; arm remains enabled '
                            'in a current-position hold', True))
                if not self.operations.prove_home(
                        context, startup=startup_home,
                        target_positions=pre_home_target,
                        home_stage='PRE_HOME'):
                    if session.motor_control_lost_reason:
                        return self.shutdown(
                            context, normal_completion=False, failure=failure)
                    diagnostic = self.operations.return_home_diagnostic(
                        context).strip()
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'configured pre-home was not proved; arm remains '
                            'enabled in a current-position hold'
                            + (': ' + diagnostic if diagnostic else ''), True))
                session.pre_home_completed = True
            if not session.return_home_proved:
                self._transition(
                    context, MissionPhase.RETURNING_HOME,
                    'pre-home proved; moving directly to configured rough home')
                startup_home = shutdown_uses_startup_home(session)
                if not self.operations.prove_home(
                        context, startup=startup_home,
                        target_positions=list(session.home_positions_rad),
                        home_stage='ROUGH_HOME'):
                    if session.motor_control_lost_reason:
                        return self.shutdown(
                            context, normal_completion=False, failure=failure)
                    diagnostic = self.operations.return_home_diagnostic(
                        context).strip()
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'configured rough home was not proved; arm remains '
                            'enabled in a current-position hold'
                            + (': ' + diagnostic if diagnostic else ''), True))
            if not session.storage_wrist_proved:
                self._transition(
                    context, MissionPhase.RETURNING_HOME,
                    'rough home proved; rotating J6 to the configured storage '
                    'angle before disable')
                storage_target = list(session.storage_positions_rad)
                if len(storage_target) != 6:
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'storage wrist target is missing; arm remains '
                            'enabled at rough home', True))
                startup_home = shutdown_uses_startup_home(session)
                if not self.operations.prove_home(
                        context, startup=startup_home,
                        target_positions=storage_target,
                        home_stage='STORAGE_WRIST'):
                    if session.motor_control_lost_reason:
                        return self.shutdown(
                            context, normal_completion=False, failure=failure)
                    diagnostic = self.operations.return_home_diagnostic(
                        context).strip()
                    return self._retain_command_owner_for_recovery(
                        context, MissionFailure(
                            'storage J6 rotation was not proved; arm remains '
                            'enabled in a current-position hold'
                            + (': ' + diagnostic if diagnostic else ''), True))
                session.storage_wrist_proved = True
            self._transition(
                context, MissionPhase.HOLDING,
                'configured home proved; retaining final target until disable')
            # STORAGE_WRIST endpoint/settling is already feedback-proved and
            # the position controller retains that target.  Do not introduce
            # another hold service or noise-window gate between home and the
            # feedback-confirmed all-axis disable.
            session.current_hold_proved = True
            self._transition(
                context, MissionPhase.DISABLING,
                'disabling arm with feedback-confirmed service')
            self.operations.enable_arm(context, False)
            session.disabled_proved = True
            session.arm_enabled = False
            self._revoke_ignoring_failure(context)
            self._transition(
                context, MissionPhase.STOPPING,
                'stopping PiPER-owned process groups')
            session.processes_stopped = self.operations.stop_processes(context)
            if not session.processes_stopped:
                return self._retain_command_owner_for_recovery(
                    context, MissionFailure(
                        'one or more PiPER-owned processes remain alive', True))
            return None
        except MissionFailure as exc:
            if session.motor_control_lost_reason:
                return self.shutdown(
                    context, normal_completion=False, failure=failure)
            return self._retain_command_owner_for_recovery(context, exc)

    def _retain_command_owner_for_recovery(self, context, failure):
        """Release perception while preserving a possibly powered owner."""
        operator_failure = MissionFailure(
            str(failure),
            needs_operator=True,
            outcome=failure.outcome,
            failure_code=failure.failure_code,
            retryable=failure.retryable)
        if not context.owns_process_generation:
            return operator_failure
        self._revoke_ignoring_failure(context)
        if (
                not context.phase_sequence
                or context.phase_sequence[-1] != MissionPhase.STOPPING.value):
            self._transition(
                context, MissionPhase.STOPPING,
                'operator recovery required; revoking mission authority and '
                'stopping non-command processing groups while retaining any '
                'live arm command owner')
        processing_stopped = self.operations.stop_processing_processes(context)
        if processing_stopped:
            return operator_failure
        return MissionFailure(
            '%s; one or more non-command processing groups also remain alive'
            % operator_failure,
            needs_operator=True,
            failure_code='CONTROL_UNTRUSTWORTHY',
            retryable=False)

    def _revoke_ignoring_failure(self, context):
        try:
            self.operations.authorize_mission(context, revoke=True)
        except MissionFailure:
            pass
