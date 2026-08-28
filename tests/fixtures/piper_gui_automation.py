"""Archived Phase 1 characterization helpers for the retired GUI workflow.

Phase 8 removed this module from the production GUI import graph.  The
production GUI now submits ``RunTargetScan`` and does not execute these rules;
they remain temporarily so the pre-refactor behavior record stays testable.
"""

from dataclasses import dataclass
from enum import Enum
import math

from piper_mobile_manipulation.failure_model import as_failure


ROUGH_ACQUISITION = 'ROUGH_ACQUISITION'
MULTIVIEW_SCAN = 'MULTIVIEW_SCAN'
EXECUTOR_COMMAND_PUBLISHER = '/scan_viewpoint_executor'
ACQUISITION_SERVICE_TIMEOUT_SEC = 8.0
ACQUISITION_PLAN_TIMEOUT_SEC = 185.0
WORKFLOW_ASSESSMENT_TIMEOUT_SEC = 15.0
PLAN_REQUEST_QUEUE_TIMEOUT_SEC = 12.0
MULTIVIEW_PLAN_TIMEOUT_SEC = 185.0
STEP45_AUTO_RECOVERY_MAX_ATTEMPTS = 2
STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC = 20.0
STEP45_AUTO_RECOVERY_RETRY_SEC = 0.50


class AcquisitionPhase(str, Enum):
    IDLE = 'IDLE'
    REQUESTING_PREPARE = 'REQUESTING_PREPARE'
    WAITING_PLAN = 'WAITING_PLAN'
    PLAN_READY = 'PLAN_READY'
    FAILED = 'FAILED'


class Step4Phase(str, Enum):
    IDLE = 'IDLE'
    STARTING_STACK = 'STARTING_STACK'
    CHECKING_WORKFLOW = 'CHECKING_WORKFLOW'
    WAITING_SCAN_READY = 'WAITING_SCAN_READY'
    REQUESTING_PLAN = 'REQUESTING_PLAN'
    WAITING_PLAN = 'WAITING_PLAN'
    PLAN_READY = 'PLAN_READY'
    FAILED = 'FAILED'


STEP4_BUSY_PHASES = frozenset((
    Step4Phase.STARTING_STACK,
    Step4Phase.CHECKING_WORKFLOW,
    Step4Phase.WAITING_SCAN_READY,
    Step4Phase.REQUESTING_PLAN,
    Step4Phase.WAITING_PLAN,
))


def command_publisher_identity_pending(names):
    """Return whether ROS discovery has not resolved an endpoint identity yet."""
    return any('UNKNOWN' in str(name).upper() for name in names)


def retryable_multiview_terminal(
        execution_mode, state, scan_approval_used):
    """Return whether a consumed scan approval ended without completion."""
    return (
        str(execution_mode) == MULTIVIEW_SCAN
        and str(state) in ('ABORTED', 'INVALID')
        and bool(scan_approval_used)
    )


def step45_auto_recovery_blocker(message):
    """Return an operator-action blocker, or empty for a bounded auto retry.

    Automatic recovery is plan-only: it may reassess the workflow and prepare
    a new proposal, but it never approves motion.  Explicit workspace,
    ownership, hardware and model blockers therefore remain operator stops.
    """
    return as_failure(message).recovery_blocker


def command_publisher_ownership_rejection(
        names, owned_stack_running=False, executor_node_present=False):
    """Require the motion-enabled executor to be the sole command publisher."""
    publishers = [str(name) for name in names]
    if publishers == [EXECUTOR_COMMAND_PUBLISHER]:
        return ''
    if (
            len(publishers) == 1
            and command_publisher_identity_pending(publishers)
            and bool(owned_stack_running)
            and bool(executor_node_present)):
        # Foxy/Fast DDS can retain UNKNOWN endpoint identity in this node's
        # local graph cache after an independent graph query has resolved it.
        # The GUI proved zero publishers before launching its own stack; with
        # that owned process still live, the exact executor node present, and
        # exactly one endpoint, the endpoint is the owned executor. Any second
        # publisher still changes the count and fails closed.
        return ''
    return (
        'Expected exactly one executor command publisher; found: '
        + (', '.join(publishers) or 'none')
    )


def validate_rough_coordinates(values):
    try:
        coordinates = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError('rough target XYZ must be numeric')
    if len(coordinates) != 3 or not all(math.isfinite(value) for value in coordinates):
        raise ValueError('rough target XYZ must contain three finite values')
    return coordinates


def validate_automation_speed(value):
    try:
        speed = float(value)
    except (TypeError, ValueError):
        raise ValueError('automation speed must be numeric')
    if not math.isfinite(speed) or speed < 1.0 or speed > 100.0:
        raise ValueError('automation speed must be between 1 and 100 percent')
    return speed


def plan_rejection(
        plan, expected_kind, expected_views=None, expected_target=None,
        target_tolerance_m=0.001, expected_source_request_id=None):
    if plan is None:
        return 'no plan is available'
    if str(getattr(plan, 'plan_kind', '')) != expected_kind:
        return 'expected %s plan' % expected_kind
    if not bool(getattr(plan, 'valid', False)):
        return 'plan is invalid'
    if str(getattr(plan, 'planner_backend', '')) != 'tesseract':
        return 'plan backend is not Tesseract'
    if not bool(getattr(plan, 'collision_model_qualified', False)):
        return 'collision model is not qualified'
    if not str(getattr(plan, 'plan_id', '')):
        return 'plan ID is missing'
    if (
            expected_source_request_id is not None
            and str(getattr(plan, 'source_request_id', ''))
            != str(expected_source_request_id)):
        return 'plan does not belong to the current acquisition session'
    if len(str(getattr(plan, 'trajectory_sha256', ''))) != 64:
        return 'trajectory hash is invalid'
    views = int(getattr(plan, 'planned_viewpoints', 0))
    if expected_kind == ROUGH_ACQUISITION and not 1 <= views <= 5:
        return 'acquisition plan must contain one to five poses'
    if expected_views is not None and views != int(expected_views):
        return 'plan must contain exactly %d views' % int(expected_views)
    if expected_target is not None:
        target = getattr(plan, 'target_center', None)
        if target is None:
            return 'plan target center is missing'
        try:
            delta = math.sqrt(sum(
                (float(actual) - float(expected)) ** 2
                for actual, expected in zip(
                    (target.x, target.y, target.z), expected_target)))
        except (AttributeError, TypeError, ValueError):
            return 'plan target center is invalid'
        if not math.isfinite(delta) or delta > float(target_tolerance_m):
            return 'plan target does not match the current rough coordinates'
    return ''


def tracking_health_rejection(
        health, received_at=None, now=None, maximum_receive_age_sec=1.0):
    """Validate direct live tracking evidence without workflow state."""
    if health is None:
        return 'tracking health is unavailable'
    if received_at is not None:
        current = float(now) if now is not None else 0.0
        try:
            receive_age = current - float(received_at)
        except (TypeError, ValueError):
            return 'tracking health receipt time is invalid'
        if (
                not math.isfinite(receive_age)
                or receive_age < 0.0
                or receive_age > float(maximum_receive_age_sec)):
            return 'tracking health message is stale'
    if str(getattr(health, 'lifecycle_state', '')) != 'TRACKING':
        return 'tracking lifecycle is not TRACKING'
    if not bool(getattr(health, 'camera_settled', False)):
        return 'camera is not settled'
    if bool(getattr(health, 'prediction_only', True)):
        return 'tracking is prediction-only'
    try:
        age = float(health.measurement_age_sec)
    except (AttributeError, TypeError, ValueError):
        return 'tracking measurement age is invalid'
    if not math.isfinite(age) or age > 0.75:
        return 'tracking measurement is stale'
    return ''


def tracking_lock_rejection(
        health, workflow, require_scan_ready=True, received_at=None, now=None):
    """Require a fresh measured lock, optionally after workflow assessment."""
    health_rejection = tracking_health_rejection(
        health, received_at=received_at, now=now)
    if health_rejection:
        return health_rejection
    if not isinstance(workflow, dict):
        return 'supervised workflow status is unavailable'
    if not bool(workflow.get('measured_lock_ready', False)):
        return str(
            workflow.get(
                'measured_lock_rejection',
                'workflow does not report a measured target lock'))
    if (
            require_scan_ready
            and str(workflow.get('state', '')) != 'SCAN_READY'):
        return 'supervised workflow is not SCAN_READY'
    return ''


def plan_matches_request(plan, request_id):
    """Require the worker plan to carry the complete bridge request ID."""
    plan_id = str(getattr(plan, 'plan_id', ''))
    request = str(request_id)
    return bool(plan_id and request and request == plan_id)


def readiness_rejection(
        readiness, received_at, now, planning_mode, maximum_age_sec=1.0):
    """Validate one explicit Tesseract readiness mode and preserve blockers."""
    if planning_mode not in ('acquisition', 'multiview'):
        raise ValueError('planning_mode must be acquisition or multiview')
    if readiness is None or received_at is None:
        return 'Tesseract readiness has not arrived'
    age = float(now) - float(received_at)
    if not math.isfinite(age) or age < 0.0 or age > float(maximum_age_sec):
        return 'Tesseract readiness is stale (%.2fs)' % age
    if not bool(getattr(readiness, 'worker_ready', False)):
        return 'Tesseract worker is not ready'
    ready = bool(getattr(readiness, planning_mode + '_ready', False))
    if ready:
        return ''
    blockers = list(getattr(readiness, planning_mode + '_blockers', ()))
    return '; '.join(str(blocker) for blocker in blockers) or (
        planning_mode + ' inputs are not ready')


def step4_workflow_action(workflow, workflow_started=False):
    """Classify an authoritative workflow diagnostic for one Step-4 attempt."""
    if not isinstance(workflow, dict) or not str(workflow.get('state', '')):
        return 'fail', 'supervised workflow diagnostic state is missing'
    state = str(workflow['state'])
    if state == 'PLAN_READY':
        return (
            'fail',
            'movable clutter was detected; clear the workspace and retry Step 4',
        )
    if state == 'ABORTED' and workflow_started:
        return (
            'fail',
            str(workflow.get('reason', 'workflow assessment aborted')),
        )
    if state in ('IDLE', 'COMPLETE', 'ABORTED'):
        if workflow_started:
            return 'wait', ''
        return 'start', ''
    if state == 'INITIALIZING':
        return 'wait', ''
    if state == 'SCAN_READY':
        if not bool(workflow.get('measured_lock_ready', False)):
            return (
                'fail',
                str(workflow.get(
                    'measured_lock_rejection',
                    'workflow does not report a measured target lock')),
            )
        return 'ready', ''
    return (
        'fail',
        'supervised workflow is in incompatible active state %s' % state,
    )


@dataclass
class AutomationSession:
    rough_coordinates: tuple = ()
    acquisition_request_id: str = ''
    acquisition_plan_id: str = ''
    acquisition_hash: str = ''
    acquisition_confirmed: bool = False
    scan_approval_used: bool = False
    acquisition_approved: bool = False
    target_acquired: bool = False
    current_lock_adopted: bool = False
    scan_plan_id: str = ''
    scan_hash: str = ''
    scan_confirmed: bool = False
    scan_expected_views: int = 13
    ended: bool = False
    end_reason: str = ''

    def prepare_acquisition(
            self, rough_coordinates, acquisition_plan,
            acquisition_request_id=''):
        coordinates = validate_rough_coordinates(rough_coordinates)
        rejection = plan_rejection(
            acquisition_plan,
            ROUGH_ACQUISITION,
            expected_target=coordinates,
            expected_source_request_id=acquisition_request_id or None,
        )
        if rejection:
            raise ValueError(rejection)
        self.rough_coordinates = coordinates
        self.acquisition_request_id = str(acquisition_request_id)
        self.acquisition_plan_id = str(acquisition_plan.plan_id)
        self.acquisition_hash = str(acquisition_plan.trajectory_sha256)
        self.acquisition_confirmed = False
        self.scan_approval_used = False
        self.acquisition_approved = False
        self.target_acquired = False
        self.current_lock_adopted = False
        self.scan_plan_id = ''
        self.scan_hash = ''
        self.scan_confirmed = False
        self.ended = False
        self.end_reason = ''

    def confirm_acquisition(self):
        if not self.acquisition_plan_id or len(self.acquisition_hash) != 64:
            raise ValueError('an exact acquisition plan must be prepared first')
        if self.ended:
            raise ValueError('automation session has ended')
        self.acquisition_confirmed = True

    def scan_plan_rejection(self, plan):
        if self.ended:
            return 'automation session has ended'
        if not self.acquisition_approved and not self.current_lock_adopted:
            return 'the exact acquisition approval has not been accepted'
        if not self.target_acquired:
            return 'a matching measured target has not been acquired or adopted'
        if self.scan_approval_used:
            return 'the session has already consumed its one scan approval'
        return plan_rejection(
            plan, MULTIVIEW_SCAN, expected_views=self.scan_expected_views)

    def prepare_scan(self, plan):
        rejection = self.scan_plan_rejection(plan)
        if rejection:
            raise ValueError(rejection)
        self.scan_plan_id = str(plan.plan_id)
        self.scan_hash = str(plan.trajectory_sha256)
        self.scan_confirmed = False

    def discard_scan_plan(self):
        """Forget an unapproved proposal so Step 4 can request a fresh one."""
        if self.scan_approval_used:
            raise ValueError('an approved scan plan cannot be discarded for retry')
        self.scan_plan_id = ''
        self.scan_hash = ''
        self.scan_confirmed = False

    def confirm_scan(self, plan):
        rejection = self.scan_plan_rejection(plan)
        if rejection:
            raise ValueError(rejection)
        if (
                str(plan.plan_id) != self.scan_plan_id
                or str(plan.trajectory_sha256) != self.scan_hash):
            raise ValueError('displayed scan plan changed after preparation')
        self.scan_confirmed = True
        self.scan_approval_used = True

    def mark_acquisition_approved(self):
        if not self.ended and self.acquisition_confirmed:
            self.acquisition_approved = True

    def mark_target_acquired(self):
        if not self.ended and self.acquisition_approved:
            self.target_acquired = True

    def adopt_current_lock(self):
        """Begin a new scan phase from a separately verified measured lock."""
        if self.scan_approval_used:
            raise ValueError('the session has already consumed its one scan approval')
        self.current_lock_adopted = True
        self.target_acquired = True
        self.scan_plan_id = ''
        self.scan_hash = ''
        self.scan_confirmed = False
        self.ended = False
        self.end_reason = ''

    def finish(self, reason):
        self.ended = True
        self.end_reason = str(reason)
