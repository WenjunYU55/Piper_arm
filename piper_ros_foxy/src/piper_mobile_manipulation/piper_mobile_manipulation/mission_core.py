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
REQUIRED_CAPTURES = 13
MAX_OCCLUSION_ACTIONS = 6
HEARTBEAT_TIMEOUT_SEC = 5.0
TASK_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.:-]{8,128}$')


class MissionPhase(str, Enum):
    LISTENING = 'LISTENING'
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
}


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
    profile_name = str(payload.get('target_profile', '')).strip() or 'green_cube'
    profile = TARGET_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError('unsupported target profile: %s' % profile_name)
    normalized_label = ' '.join(label.lower().replace('_', ' ').split())
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
    current_hold_proved: bool = False
    disabled_proved: bool = False
    processes_stopped: bool = False
    arm_enabled: bool = False
    home_positions_rad: tuple = ()

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
                not self.current_hold_proved
                or not self.return_home_proved
                or not self.disabled_proved):
            return MissionPhase.NEEDS_OPERATOR, (
                'settled current-position hold, verified home return, and '
                'feedback-confirmed disable were not all proved')
        if not self.processes_stopped:
            return MissionPhase.FAILED, 'PiPER-owned processes did not all stop'
        return MissionPhase.FAILED, self.reason

    def result_payload(self, outcome, reason, dataset_path='', manifest_sha256='',
                       mesh_job_id='', action_summary=None):
        safe_shutdown = bool(
            self.current_hold_proved and self.return_home_proved
            and self.disabled_proved
            and self.processes_stopped)
        payload = {
            'task_id': self.task_id,
            'mission_sha256': self.mission_sha256,
            'outcome': str(outcome),
            'reason': str(reason),
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
