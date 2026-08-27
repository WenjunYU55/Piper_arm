"""Guard mission-bound calibration, dataset, and process resources."""

import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml

from piper_mobile_manipulation.mission.engine import MissionFailure


def calibration_identity_for_mission(root, environment):
    """Bind capture provenance to the exact hand-eye file used by this run."""
    default_path = (
        Path(root) / 'L515_camera' / 'calibration' / 'hand_eye'
        / 'session_20260808_straight_mount' / 'calibration_result.yaml')
    path = Path(str(environment.get(
        'PIPER_HAND_EYE_CALIBRATION', default_path))).expanduser().resolve()
    if not path.is_file():
        raise MissionFailure(
            'hand-eye calibration file is missing: %s' % path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return path, digest.hexdigest()


def previous_generation_cleanup_targets(
        live_processes, all_motors_disabled):
    """Select exact stale handles that are safe to stop before admission."""
    live = tuple(dict.fromkeys(str(name) for name in live_processes))
    if 'driver' in live and not bool(all_motors_disabled):
        return ()
    return live


def discard_failed_zero_capture_dataset(
        scan_dir, dataset_root, task_id, mission_sha256):
    """Delete only an identity-matched failed dataset with no captures."""
    try:
        raw_candidate = Path(str(scan_dir))
        if raw_candidate.is_symlink():
            return False, 'scan dataset path is a symbolic link'
        root = Path(dataset_root).resolve(strict=True)
        candidate = raw_candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return False, 'scan directory or dataset root does not exist'
    if candidate.parent != root or not re.fullmatch(
            r'scan_[0-9]{8}_[0-9]{6}(?:_[A-Za-z0-9_-]+)?',
            candidate.name):
        return False, 'scan directory is outside the guarded dataset root'
    if not candidate.is_dir():
        return False, 'scan dataset is not a regular directory'
    try:
        metadata = yaml.safe_load(
            (candidate / 'metadata.yaml').read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return False, 'scan metadata is unreadable: %s' % exc
    if not isinstance(metadata, dict):
        return False, 'scan metadata is not a mapping'
    if (str(metadata.get('task_id', '')) != str(task_id)
            or str(metadata.get('mission_sha256', ''))
            != str(mission_sha256)):
        return False, 'scan metadata does not match the failed mission'
    allowed_root_files = {
        'metadata.yaml', 'manifest.json', 'coverage_envelope.yaml'}
    incomplete_frame_pattern = re.compile(
        r'view_[0-9]{3}_(?:rgb|depth|mask|native_depth|confidence|'
        r'target_depth|target_support_mask)(?:\.partial)?\.(?:png|npy)$')
    for path in candidate.rglob('*'):
        if path.is_symlink():
            return False, 'scan dataset contains a symbolic link'
        if not path.is_file():
            continue
        relative = path.relative_to(candidate)
        if relative.as_posix() in allowed_root_files:
            continue
        if relative.parent.as_posix() == 'frames':
            if re.fullmatch(r'view_[0-9]{3}_metadata\.yaml', path.name):
                return False, 'scan contains one or more completed captures'
            if incomplete_frame_pattern.fullmatch(path.name):
                continue
        return False, 'scan contains unknown or derived artifacts'
    manifest_path = candidate / 'manifest.json'
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            capture_count = int(manifest.get('capture_count', -1))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, 'scan manifest is unreadable: %s' % exc
        if capture_count != 0:
            return False, 'scan manifest records one or more captures'
    shutil.rmtree(candidate)
    return True, 'failed zero-capture scan dataset was permanently removed'


def find_failed_mission_dataset(dataset_root, task_id, mission_sha256):
    """Find one exact identity-matched mission directory."""
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, RuntimeError):
        return '', 'dataset root does not exist'
    matches = []
    for candidate in sorted(root.glob('scan_*')):
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        if not re.fullmatch(
                r'scan_[0-9]{8}_[0-9]{6}(?:_[A-Za-z0-9_-]+)?',
                candidate.name):
            continue
        try:
            metadata = yaml.safe_load(
                (candidate / 'metadata.yaml').read_text(encoding='utf-8'))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
        if (isinstance(metadata, dict)
                and str(metadata.get('task_id', '')) == str(task_id)
                and str(metadata.get('mission_sha256', ''))
                == str(mission_sha256)):
            matches.append(candidate)
    if len(matches) == 1:
        return str(matches[0]), 'found exact identity-matched mission dataset'
    if not matches:
        return '', 'no identity-matched mission dataset exists'
    return '', 'multiple identity-matched mission datasets exist'
