"""Pure contracts for one bounded autonomous target-scan mission."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
import time


DEFAULT_DEADLINE_SEC = 1200.0
MAX_DEADLINE_SEC = 1200.0
# Automatic scanning is not a fixed-view sweep. This is only the minimum
# number of accepted observations needed to seed a measured object model;
# feature/side coverage and measured surface-gain convergence decide when the
# mission actually completes, up to MAX_FEATURE_CAPTURES.
REQUIRED_CAPTURES = 8
MAX_FEATURE_CAPTURES = 24
MAX_OCCLUSION_ACTIONS = 6
MAX_PENDING_MISSIONS = 8
MISSION_QUEUE_COALESCE_SEC = 1.0
HEARTBEAT_TIMEOUT_SEC = 5.0
TASK_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.:-]{8,128}$')
TARGET_WORD_PATTERN = re.compile(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?')


class MissionPhase(str, Enum):
    LISTENING = 'LISTENING'
    QUEUED = 'QUEUED'
    GOAL_LATCHED = 'GOAL_LATCHED'
    STARTING = 'STARTING'
    PREFLIGHT = 'PREFLIGHT'
    ENABLE_AND_HOLD = 'ENABLE_AND_HOLD'
    ROUGH_ACQUISITION = 'ROUGH_ACQUISITION'
    TARGET_LOCK = 'TARGET_LOCK'
    OCCLUSION_PROBE = 'OCCLUSION_PROBE'
    OCCLUSION_CLEARANCE = 'OCCLUSION_CLEARANCE'
    VIEW_PLANNING = 'VIEW_PLANNING'
    CAPTURING = 'CAPTURING'
    RETURNING_HOME = 'RETURNING_HOME'
    HOLDING = 'HOLDING'
    DISABLING = 'DISABLING'
    STOPPING = 'STOPPING'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'
    NEEDS_OPERATOR = 'NEEDS_OPERATOR'


TERMINAL_PHASES = frozenset((
    MissionPhase.SUCCEEDED,
    MissionPhase.FAILED,
    MissionPhase.NEEDS_OPERATOR,
))


@dataclass(frozen=True)
class TargetProfile:
    name: str
    prompt: str
    minimum_confidence: float
    semantic_hints: tuple


TARGET_PROFILES = {
    'green_cube': TargetProfile(
        name='green_cube',
        prompt='green cube .',
        minimum_confidence=0.60,
        semantic_hints=('leaf', 'branch', 'hand', 'finger'),
    ),
    'generic_open_vocab': TargetProfile(
        name='generic_open_vocab',
        prompt='',
        minimum_confidence=0.60,
        semantic_hints=('leaf', 'branch', 'hand', 'finger'),
    ),
}


def target_prompt(label):
    """Build one bounded GroundingDINO phrase from operator text."""
    words = TARGET_WORD_PATTERN.findall(str(label))
    if not words:
        raise ValueError('target_label must contain letters or numbers')
    normalized = ' '.join(words[:12]).lower()
    if len(normalized) > 96:
        normalized = normalized[:96].rstrip()
    return normalized + ' .'


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode('utf-8')


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_goal_payload(payload, now_sec=None):
    """Return a normalized immutable goal or raise a precise ValueError."""
    if not isinstance(payload, dict):
        raise ValueError('goal is not an object')
    task_id = str(payload.get('task_id', '')).strip()
    if TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError('task_id must contain 8-128 safe characters')
    if str(payload.get('task_type', '')).strip().upper() != 'SCAN_3D':
        raise ValueError('task_type must be SCAN_3D')
    label = str(payload.get('target_label', '')).strip()
    if not label:
        raise ValueError('target_label is missing')
    normalized_label = ' '.join(label.lower().replace('_', ' ').split())
    requested_profile = str(payload.get('target_profile', '')).strip()
    profile_name = requested_profile or (
        'green_cube' if normalized_label == 'green cube'
        else 'generic_open_vocab')
    profile = TARGET_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError('unsupported target profile: %s' % profile_name)
    if profile_name == 'green_cube' and normalized_label != 'green cube':
        raise ValueError(
            'green_cube profile requires target_label "green cube"')
    confidence = float(payload.get('target_confidence', 0.0))
    if not math.isfinite(confidence) or confidence < profile.minimum_confidence:
        raise ValueError(
            'target confidence %.3f is below %s minimum %.3f'
            % (confidence, profile_name, profile.minimum_confidence))
    target = payload.get('rough_target')
    if not isinstance(target, dict):
        raise ValueError('rough_target is missing')
    frame_id = str(target.get('frame_id', '')).strip()
    if frame_id not in ('odom', 'base_link'):
        raise ValueError('rough_target frame must be odom or base_link')
    stamp_sec = float(target.get('stamp_sec', 0.0))
    if not math.isfinite(stamp_sec) or stamp_sec <= 0.0:
        raise ValueError('rough_target timestamp is invalid')
    current = time.time() if now_sec is None else float(now_sec)
    if current - stamp_sec > 5.0 or stamp_sec - current > 0.5:
        raise ValueError('rough_target timestamp is stale or in the future')
    position = target.get('position')
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError('rough_target position must contain XYZ')
    position = [float(value) for value in position]
    if not all(math.isfinite(value) for value in position):
        raise ValueError('rough_target position is not finite')
    covariance = target.get('covariance', [])
    if not isinstance(covariance, (list, tuple)) or len(covariance) != 36:
        raise ValueError('rough_target covariance must contain 36 values')
    covariance = [float(value) for value in covariance]
    if not all(math.isfinite(value) for value in covariance):
        raise ValueError('rough_target covariance is not finite')
    for diagonal in (covariance[0], covariance[7], covariance[14]):
        if diagonal < 0.0 or math.sqrt(diagonal) > 0.30:
            raise ValueError('rough_target position uncertainty exceeds 0.30m')
    requested_deadline = float(payload.get('deadline_sec', 0.0))
    deadline = requested_deadline if requested_deadline > 0.0 else DEFAULT_DEADLINE_SEC
    if not math.isfinite(deadline) or deadline < 60.0 or deadline > MAX_DEADLINE_SEC:
        raise ValueError('deadline_sec must be between 60 and 1200 seconds')
    normalized = {
        'task_id': task_id,
        'task_type': 'SCAN_3D',
        'target_label': label,
        'target_profile': profile_name,
        'target_prompt': (
            profile.prompt if profile.prompt else target_prompt(label)),
        'target_confidence': confidence,
        'deadline_sec': deadline,
        'rough_target': {
            'frame_id': frame_id,
            'stamp_sec': stamp_sec,
            'position': position,
            'covariance': covariance,
        },
    }
    normalized['mission_sha256'] = sha256_value(normalized)
    return normalized


def mission_target_distance_m(normalized_goal):
    """Return the rough-target Euclidean distance in the admitted base frame."""
    try:
        position = normalized_goal['rough_target']['position']
        values = tuple(float(value) for value in position)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('queued mission rough target is invalid') from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError('queued mission rough target must contain finite XYZ')
    return math.sqrt(sum(value * value for value in values))


def closest_pending_mission(records):
    """Choose the nearest pending mission with stable arrival-order ties."""
    candidates = list(records)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: (
            mission_target_distance_m(record['normalized']),
            int(record['sequence']),
            str(record['normalized']['task_id']),
        ),
    )


def mission_queue_ready(records, now_monotonic, coalesce_sec):
    """Return whether the closest-first queue may start its next mission."""
    candidates = list(records)
    if not candidates:
        return False
    delay = float(coalesce_sec)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError('mission queue coalescing delay is invalid')
    oldest = min(float(record['admitted_monotonic']) for record in candidates)
    return float(now_monotonic) - oldest >= delay


def queued_cancel_result(normalized_goal, reason='queued mission cancelled'):
    """Build a terminal result for a mission that never owned arm resources."""
    payload = {
        'task_id': str(normalized_goal['task_id']),
        'mission_sha256': str(normalized_goal['mission_sha256']),
        'outcome': 'CANCELLED',
        'reason': str(reason),
        'failure_code': 'CANCELLED',
        'retryable': True,
        'safe_shutdown': True,
        'dataset_path': '',
        'manifest_sha256': '',
        'capture_count': 0,
        'mesh_job_id': '',
        'action_summary': {
            'queue_cancelled_before_start': True,
            'arm_resources_started': False,
        },
    }
    payload['result_sha256'] = sha256_value(payload)
    return payload


@dataclass
class MissionSession:
    goal: dict
    phase: MissionPhase = MissionPhase.GOAL_LATCHED
    started_monotonic: float = field(default_factory=time.monotonic)
    phase_started_monotonic: float = field(default_factory=time.monotonic)
    reason: str = 'task accepted'
    acquisition_attempt: int = 0
    occlusion_action: int = 0
    accepted_captures: int = 0
    heartbeat_monotonic: float = field(default_factory=time.monotonic)
    return_home_proved: bool = False
    pre_home_completed: bool = False
    storage_wrist_proved: bool = False
    startup_wrist_completed: bool = False
    startup_home_completed: bool = False
    perception_scene_established: bool = False
    current_hold_proved: bool = False
    disabled_proved: bool = False
    processes_stopped: bool = False
    arm_enabled: bool = False
    motor_control_lost_reason: str = ''
    home_positions_rad: tuple = ()
    pre_home_positions_rad: tuple = ()
    storage_positions_rad: tuple = ()
    mission_ready_joint6_rad: float = 0.0
    storage_joint6_rad: float = 0.0

    @property
    def task_id(self):
        return self.goal['task_id']

    @property
    def mission_sha256(self):
        return self.goal['mission_sha256']

    @property
    def deadline_sec(self):
        return float(self.goal['deadline_sec'])

    def elapsed(self, now=None):
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.started_monotonic)

    def remaining(self, now=None):
        return max(0.0, self.deadline_sec - self.elapsed(now))

    def deadline_expired(self, now=None):
        return self.remaining(now) <= 0.0

    def heartbeat_stale(self, now=None):
        current = time.monotonic() if now is None else float(now)
        return current - self.heartbeat_monotonic > HEARTBEAT_TIMEOUT_SEC

    def heartbeat(self, now=None):
        self.heartbeat_monotonic = time.monotonic() if now is None else float(now)

    def transition(self, phase, reason, now=None):
        next_phase = MissionPhase(phase)
        if self.phase in TERMINAL_PHASES:
            raise ValueError('terminal mission cannot transition')
        self.phase = next_phase
        self.reason = str(reason)
        self.phase_started_monotonic = (
            time.monotonic() if now is None else float(now))

    def shutdown_outcome(self):
        """Classify shutdown without ever equating process exit with safety."""
        if (
                not self.return_home_proved
                or not self.storage_wrist_proved
                or not self.disabled_proved):
            return MissionPhase.NEEDS_OPERATOR, (
                'verified home return and feedback-confirmed disable were not '
                'both proved')
        if not self.processes_stopped:
            return MissionPhase.FAILED, 'PiPER-owned processes did not all stop'
        return MissionPhase.FAILED, self.reason

    def result_payload(self, outcome, reason, dataset_path='', manifest_sha256='',
                       mesh_job_id='', failure_code='', retryable=False,
                       action_summary=None):
        safe_shutdown = bool(
            self.return_home_proved and self.storage_wrist_proved
            and self.disabled_proved
            and self.processes_stopped)
        payload = {
            'task_id': self.task_id,
            'mission_sha256': self.mission_sha256,
            'outcome': str(outcome),
            'reason': str(reason),
            'failure_code': str(failure_code),
            'retryable': bool(retryable),
            'safe_shutdown': safe_shutdown,
            'dataset_path': str(dataset_path),
            'manifest_sha256': str(manifest_sha256),
            'capture_count': int(self.accepted_captures),
            'mesh_job_id': str(mesh_job_id),
            'action_summary': action_summary or {},
        }
        payload['result_sha256'] = sha256_value(payload)
        return payload


class MissionRegistry:
    """Enforce one active task and idempotent task IDs."""

    def __init__(self):
        self.active = None
        self.results = {}

    def admit(self, normalized_goal):
        task_id = normalized_goal['task_id']
        cached = self.results.get(task_id)
        if cached is not None:
            if cached.get('mission_sha256') != normalized_goal['mission_sha256']:
                return 'CONFLICT', cached
            return 'CACHED', cached
        if self.active is not None:
            if self.active.task_id == task_id:
                if self.active.mission_sha256 == normalized_goal['mission_sha256']:
                    return 'ACTIVE', self.active
                return 'CONFLICT', self.active
            return 'BUSY', self.active
        self.active = MissionSession(normalized_goal)
        return 'ACCEPTED', self.active

    def finish(self, result):
        task_id = str(result['task_id'])
        if self.active is not None and self.active.task_id == task_id:
            self.active = None
        self.results[task_id] = dict(result)
