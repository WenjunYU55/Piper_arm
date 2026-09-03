"""Filesystem-only campaign schedule and provenance management.

This module deliberately has no ROS imports and no robot-facing operations.
It records what the operator asked the existing GUI to submit; it never
authorizes, blocks, retries, or otherwise changes a mission.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


CAMPAIGN_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGN_ID = 'piper-poster-blocked-20260902'
LEGACY_ALTERNATING_CAMPAIGN_ID = 'piper-poster-20260902'
DEFAULT_X_M = (0.30, 0.50, 0.70, 0.90, 1.10)
DEFAULT_Y_M = 0.0
DEFAULT_Z_M = (0.0, 0.12, 0.30)
DEFAULT_BACKENDS = ('tesseract', 'curobo')
DEFAULT_CUBE_DIMENSIONS_MM = (35.0, 35.0, 35.0)
DEFAULT_RUNTIME_REFERENCE_SEC = 120.0

_SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')


@dataclass(frozen=True)
class TrialDefinition:
    """One predeclared physical mission in a comparison campaign."""

    sequence: int
    pair_index: int
    trial_id: str
    backend: str
    x_m: float
    y_m: float
    z_m: float

    @property
    def coordinates(self):
        return (self.x_m, self.y_m, self.z_m)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_campaign_id(value: object) -> str:
    result = str(value).strip()
    if not _SAFE_ID.fullmatch(result):
        raise ValueError(
            'campaign ID must be 1-80 letters, numbers, dots, dashes or '
            'underscores and must start with a letter or number')
    return result


def _coordinate_token(value: float) -> str:
    sign = 'm' if value < 0 else ''
    return sign + ('%.2f' % abs(float(value))).replace('.', 'p')


def _position_grid():
    for z_m in DEFAULT_Z_M:
        for x_m in DEFAULT_X_M:
            yield float(x_m), float(DEFAULT_Y_M), float(z_m)


def _make_trial(sequence: int, pair_index: int, backend: str,
                coordinates) -> TrialDefinition:
    x_m, y_m, z_m = coordinates
    return TrialDefinition(
        sequence=sequence,
        pair_index=pair_index,
        trial_id='p%02d_x%s_y%s_z%s_%s' % (
            pair_index,
            _coordinate_token(x_m),
            _coordinate_token(y_m),
            _coordinate_token(z_m),
            backend,
        ),
        backend=backend,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
    )


def default_trial_schedule() -> List[TrialDefinition]:
    """Return 15 cuRobo missions followed by the same 15 Tesseract cells."""
    positions = list(_position_grid())
    trials: List[TrialDefinition] = []
    sequence = 1
    for backend in ('curobo', 'tesseract'):
        for pair_index, coordinates in enumerate(positions, start=1):
            trials.append(_make_trial(
                sequence, pair_index, backend, coordinates))
            sequence += 1
    return trials


def alternating_trial_schedule() -> List[TrialDefinition]:
    """Return the original alternating schedule for existing campaigns."""
    trials: List[TrialDefinition] = []
    sequence = 1
    for pair_index, coordinates in enumerate(_position_grid(), start=1):
        order = (
            DEFAULT_BACKENDS
            if pair_index % 2 == 1 else tuple(reversed(DEFAULT_BACKENDS)))
        for backend in order:
            trials.append(_make_trial(
                sequence, pair_index, backend, coordinates))
            sequence += 1
    return trials


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace one generated JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + '.', suffix='.partial',
        dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with Path(path).open('r', encoding='utf-8') as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(project_root: Path) -> Dict[str, Any]:
    """Read repository identity without changing Git state."""
    root = Path(project_root).resolve()
    try:
        branch = subprocess.check_output(
            ['git', 'branch', '--show-current'], cwd=str(root), text=True,
            stderr=subprocess.DEVNULL).strip()
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ['git', 'status', '--short'], cwd=str(root), text=True,
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {'branch': '', 'commit': '', 'dirty': None, 'status': ''}
    return {
        'branch': branch,
        'commit': commit,
        'dirty': bool(status.strip()),
        'status': status,
    }


def configuration_snapshot(project_root: Path) -> Dict[str, Any]:
    """Hash relevant configuration/model sources without interpreting them."""
    root = Path(project_root).resolve()
    relative_paths = (
        'piper_ros_foxy/src/piper_mobile_manipulation/config/planner_backend.yaml',
        'piper_ros_foxy/src/piper_mobile_manipulation/config/collision_environment.yaml',
        'piper_ros_foxy/src/piper_mobile_manipulation/config/camera_params.yaml',
        'piper_ros_foxy/src/piper_mobile_manipulation/config/scan_planning_params.yaml',
        'piper_ros_foxy/src/piper_mobile_manipulation/config/scan_execution_params.yaml',
        'piper_ros_foxy/src/piper_mobile_manipulation/config/scan_capture_params.yaml',
        'piper_ros_foxy/src/piper_mobile_manipulation/config/piper_camera_capability_map.npz',
        'piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml',
        'piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model_ground.yaml',
        'piper_ros_foxy/src/piper_tesseract_foxy/model/piper.srdf',
        'piper_ros_foxy/src/piper_tesseract_foxy/model/piper_bunker.srdf',
        'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro',
        'piper_ros_foxy/src/piper_description/urdf/piper_description.urdf',
        'motion_planning/curobo/model/piper_collision_spheres.yaml',
        'motion_planning/curobo/constraints.txt',
        'piper_home_pose.json',
        'piper_joint_bounds.json',
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            files.append({
                'path': relative,
                'available': False,
                'sha256': '',
                'size_bytes': 0,
            })
            continue
        files.append({
            'path': relative,
            'available': True,
            'sha256': sha256_file(path),
            'size_bytes': int(path.stat().st_size),
        })
    environment = {}
    for name in (
            'PIPER_MISSION_ENABLE_REAL_MOTION',
            'PIPER_MISSION_SPEEDS_QUALIFIED',
            'PIPER_MISSION_FREE_MOTION_SPEED_PERCENT',
            'PIPER_MISSION_CONTACT_SPEED_PERCENT',
            'PIPER_FLOOR_PROFILE',
            'PIPER_CUROBO_COLLISION_MODEL_QUALIFIED',
            'PIPER_CUROBO_PYTHON'):
        if name in os.environ:
            environment[name] = str(os.environ[name])
    planner_models = {}
    for backend, relative in (
            ('tesseract',
             'piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml'),
            ('curobo',
             'motion_planning/curobo/model/piper_collision_spheres.yaml')):
        path = root / relative
        try:
            document = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError):
            document = None
        if not isinstance(document, dict):
            planner_models[backend] = {
                'source_path': relative,
                'hardware_qualified': None,
            }
            continue
        if backend == 'curobo':
            qualification = document.get('qualification', {})
            planner_models[backend] = {
                'source_path': relative,
                'hardware_qualified': qualification.get(
                    'hardware_qualified'),
                'qualification_date': qualification.get(
                    'qualification_date'),
                'qualification_scope': qualification.get('scope'),
                'qualification_basis': qualification.get('basis'),
                'qualified_floor_profile': qualification.get(
                    'floor_profile'),
                'qualified_free_motion_speed_percent': qualification.get(
                    'free_motion_speed_percent'),
                'qualified_contact_speed_percent': qualification.get(
                    'contact_speed_percent'),
                'conservative_geometry': False,
            }
        else:
            planner_models[backend] = {
                'source_path': relative,
                'hardware_qualified': document.get(
                    'qualified_for_hardware'),
                'qualification_date': document.get('qualification_date'),
                'qualification_scope': document.get(
                    'hardware_qualification_scope'),
                'collision_profile': document.get('collision_profile'),
            }
    return {
        'captured_utc': utc_now(),
        'git': git_state(root),
        'files': files,
        'environment': environment,
        'planner_models': planner_models,
    }


def _request_value(request: object, name: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        return request.get(name, default)
    return getattr(request, name, default)


def _result_mapping(result: object) -> Dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    names = (
        'task_id', 'outcome', 'reason', 'failure_code', 'retryable',
        'safe_shutdown', 'capture_count', 'dataset_path', 'manifest_sha256',
        'mesh_job_id')
    return {name: getattr(result, name, None) for name in names}


class CampaignStore:
    """Manage generated campaign files without interacting with ROS."""

    def __init__(self, project_root: Path, campaign_id: object):
        self.project_root = Path(project_root).resolve()
        self.campaign_id = validate_campaign_id(campaign_id)
        self.root = (
            self.project_root / 'datasets' / 'experiment_campaigns'
            / self.campaign_id)
        self.definition_path = self.root / 'campaign.json'
        self.active_path = self.root / 'active_trial.json'

    def refresh_bookmarks(self) -> Dict[str, Any]:
        """Rebuild the passive campaign-wide result bookmark ledger."""
        from .bookmarks import write_campaign_bookmarks
        return write_campaign_bookmarks(self.project_root, self.root)

    def create_or_load(self) -> Dict[str, Any]:
        existing = load_json(self.definition_path)
        expected_trials = [asdict(item) for item in default_trial_schedule()]
        if existing is not None:
            if int(existing.get('schema_version', -1)) != CAMPAIGN_SCHEMA_VERSION:
                raise ValueError('unsupported campaign schema version')
            legacy_trials = [
                asdict(item) for item in alternating_trial_schedule()]
            if existing.get('trials') not in (expected_trials, legacy_trials):
                raise ValueError(
                    'existing campaign uses a different trial design; choose '
                    'a new campaign ID')
            self.refresh_bookmarks()
            return existing
        definition = {
            'schema_version': CAMPAIGN_SCHEMA_VERSION,
            'campaign_id': self.campaign_id,
            'created_utc': utc_now(),
            'evidence_class': 'MATCHED_RUNS',
            'experimental_unit': 'one physical mission',
            'design': {
                'x_m': list(DEFAULT_X_M),
                'y_m': DEFAULT_Y_M,
                'z_m': list(DEFAULT_Z_M),
                'planner_backends': list(DEFAULT_BACKENDS),
                'pairs_per_position': 1,
                'planned_missions': len(expected_trials),
                'cube_dimensions_mm': list(DEFAULT_CUBE_DIMENSIONS_MM),
                'coverage_faces': 6,
                'runtime_reference_sec': DEFAULT_RUNTIME_REFERENCE_SEC,
                'backend_order': 'curobo_block_then_tesseract_block',
                'limitation': (
                    'N=1 per backend and position; backend-block execution '
                    'confounds planner with elapsed time and environmental '
                    'drift. Per-cell variance and confidence intervals are '
                    'not supported.'),
            },
            'trials': expected_trials,
        }
        atomic_write_json(self.definition_path, definition)
        self.refresh_bookmarks()
        return definition

    def definition(self) -> Dict[str, Any]:
        value = load_json(self.definition_path)
        if value is None:
            return self.create_or_load()
        return value

    def _trial_dir(self, trial_id: str) -> Path:
        return self.root / 'trials' / str(trial_id)

    def _attempt_directories(self, trial_id: str) -> Iterable[Path]:
        directory = self._trial_dir(trial_id) / 'attempts'
        return sorted(
            (item for item in directory.glob('*') if item.is_dir()),
            key=lambda item: item.name) if directory.is_dir() else ()

    def _matched_terminal_exists(self, trial_id: str) -> bool:
        for attempt in self._attempt_directories(trial_id):
            submission = load_json(attempt / 'submission.json', {})
            if (submission.get('matches_schedule') is True
                    and (attempt / 'terminal.json').is_file()):
                return True
        return False

    def progress(self) -> Dict[str, Any]:
        definition = self.definition()
        completed = []
        for trial in definition['trials']:
            if self._matched_terminal_exists(trial['trial_id']):
                completed.append(trial['trial_id'])
        active = load_json(self.active_path, {}) or {}
        return {
            'completed': len(completed),
            'total': len(definition['trials']),
            'completed_trial_ids': completed,
            'active_trial': active,
        }

    def next_trial(self) -> Optional[Dict[str, Any]]:
        definition = self.definition()
        active = load_json(self.active_path, {}) or {}
        active_id = str(active.get('trial_id', ''))
        if active_id and not self._matched_terminal_exists(active_id):
            return next((
                dict(item) for item in definition['trials']
                if item['trial_id'] == active_id), None)
        for trial in definition['trials']:
            if not self._matched_terminal_exists(trial['trial_id']):
                return dict(trial)
        return None

    def load_next_trial(self) -> Optional[Dict[str, Any]]:
        trial = self.next_trial()
        if trial is None:
            try:
                self.active_path.unlink()
            except FileNotFoundError:
                pass
            return None
        active = dict(trial)
        active.update({
            'campaign_id': self.campaign_id,
            'loaded_utc': utc_now(),
            'task_id': '',
        })
        atomic_write_json(self.active_path, active)
        return trial

    @staticmethod
    def _schedule_match(trial: Dict[str, Any], coordinates: Sequence[Any],
                        backend: object) -> bool:
        try:
            actual = tuple(float(value) for value in coordinates)
        except (TypeError, ValueError):
            return False
        expected = (trial['x_m'], trial['y_m'], trial['z_m'])
        return bool(
            len(actual) == 3
            and all(math.isclose(a, float(b), abs_tol=1e-9)
                    for a, b in zip(actual, expected))
            and str(backend).strip().lower() == str(trial['backend']))

    def record_submission(
            self, task_id: object, request: object,
            submitted_wall_time_ns: Optional[int] = None,
            submitted_monotonic_ns: Optional[int] = None) -> Dict[str, Any]:
        trial = self.next_trial()
        if trial is None:
            raise ValueError('campaign schedule is complete')
        identifier = str(task_id).strip()
        if not identifier or '/' in identifier or identifier in ('.', '..'):
            raise ValueError('task ID is invalid')
        coordinates = tuple(_request_value(request, 'coordinates', ()))
        backend = str(_request_value(request, 'planner_backend', '')).lower()
        submission = {
            'schema_version': EVIDENCE_SCHEMA_VERSION,
            'campaign_id': self.campaign_id,
            'trial_id': trial['trial_id'],
            'task_id': identifier,
            'submitted_utc': utc_now(),
            'submitted_wall_time_ns': int(
                time.time_ns() if submitted_wall_time_ns is None
                else submitted_wall_time_ns),
            'submitted_monotonic_ns': int(
                time.monotonic_ns() if submitted_monotonic_ns is None
                else submitted_monotonic_ns),
            'requested_coordinates_m': [float(value) for value in coordinates],
            'requested_backend': backend,
            'requested_target_label': str(
                _request_value(request, 'target_label', '')),
            'expected_trial': dict(trial),
            'matches_schedule': self._schedule_match(
                trial, coordinates, backend),
            'configuration_snapshot': configuration_snapshot(
                self.project_root),
        }
        attempt = self._trial_dir(trial['trial_id']) / 'attempts' / identifier
        destination = attempt / 'submission.json'
        if destination.exists():
            previous = load_json(destination)
            if previous != submission:
                raise ValueError('task submission already exists with other data')
            return previous
        atomic_write_json(destination, submission)
        active = dict(trial)
        active.update({
            'campaign_id': self.campaign_id,
            'task_id': identifier,
            'submission_path': str(destination),
            'matches_schedule': submission['matches_schedule'],
        })
        atomic_write_json(self.active_path, active)
        self.refresh_bookmarks()
        return submission

    def _find_attempt(self, task_id: object) -> Optional[Path]:
        identifier = str(task_id)
        root = self.root / 'trials'
        if not root.is_dir():
            return None
        matches = list(root.glob('*/attempts/%s' % identifier))
        return matches[0] if len(matches) == 1 else None

    def attempt_for_task(self, task_id: object) -> Optional[Path]:
        """Return the unique generated attempt directory for one task."""
        return self._find_attempt(task_id)

    def record_terminal(self, task_id: object, result: object) -> Path:
        attempt = self._find_attempt(task_id)
        if attempt is None:
            raise ValueError('task is not registered in this campaign')
        terminal = _result_mapping(result)
        terminal.update({
            'schema_version': EVIDENCE_SCHEMA_VERSION,
            'campaign_id': self.campaign_id,
            'recorded_utc': utc_now(),
            'recorded_wall_time_ns': time.time_ns(),
            'recorded_monotonic_ns': time.monotonic_ns(),
        })
        path = attempt / 'terminal.json'
        atomic_write_json(path, terminal)
        active = load_json(self.active_path, {}) or {}
        if str(active.get('task_id', '')) == str(task_id):
            try:
                self.active_path.unlink()
            except FileNotFoundError:
                pass
        self.refresh_bookmarks()
        return path

    def record_submission_failure(self, message: object) -> Optional[Path]:
        active = load_json(self.active_path, {}) or {}
        task_id = str(active.get('task_id', ''))
        attempt = self._find_attempt(task_id) if task_id else None
        if attempt is None:
            return None
        path = attempt / 'submission_failure.json'
        atomic_write_json(path, {
            'schema_version': EVIDENCE_SCHEMA_VERSION,
            'task_id': task_id,
            'recorded_utc': utc_now(),
            'reason': str(message),
            'excluded': True,
            'exclusion_reason': 'mission action submission failed',
        })
        self.refresh_bookmarks()
        return path

    def task_attempts(self) -> List[Path]:
        return sorted(self.root.glob('trials/*/attempts/*'))
