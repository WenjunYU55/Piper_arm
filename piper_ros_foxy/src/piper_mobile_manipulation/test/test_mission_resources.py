"""Characterize mission-bound calibration, dataset, and process resources."""

import hashlib

import pytest
import yaml

import piper_mobile_manipulation.target_scan_mission_node as mission_node
from piper_mobile_manipulation.mission.engine import MissionFailure
from piper_mobile_manipulation.mission.resources import (
    calibration_identity_for_mission,
    find_failed_mission_dataset,
    previous_generation_cleanup_targets,
)


def test_calibration_identity_uses_exact_override_bytes(tmp_path):
    """Bind the configured calibration path and byte digest unchanged."""
    calibration = tmp_path / 'calibration.yaml'
    payload = b'transform:\n  translation: [0.1, 0.2, 0.3]\n'
    calibration.write_bytes(payload)

    path, digest = calibration_identity_for_mission(
        tmp_path, {'PIPER_HAND_EYE_CALIBRATION': str(calibration)})

    assert path == calibration.resolve()
    assert digest == hashlib.sha256(payload).hexdigest()


def test_calibration_identity_fails_closed_when_file_is_missing(tmp_path):
    """Reject a missing mission provenance input before capture startup."""
    with pytest.raises(MissionFailure, match='calibration file is missing'):
        calibration_identity_for_mission(
            tmp_path,
            {'PIPER_HAND_EYE_CALIBRATION': str(tmp_path / 'missing.yaml')},
        )


def test_previous_generation_cleanup_retains_driver_disable_gate():
    """Never stop a live driver without an all-six-disabled proof."""
    live = ('vision', 'driver', 'vision')
    assert previous_generation_cleanup_targets(live, False) == ()
    assert previous_generation_cleanup_targets(live, True) == (
        'vision', 'driver')


def test_failed_dataset_discovery_requires_one_exact_identity(tmp_path):
    """Return only one task-and-mission matched scan directory."""
    first = tmp_path / 'scan_20260827_120000'
    first.mkdir()
    (first / 'metadata.yaml').write_text(yaml.safe_dump({
        'task_id': 'task-0001',
        'mission_sha256': 'digest-0001',
    }), encoding='utf-8')

    path, reason = find_failed_mission_dataset(
        tmp_path, 'task-0001', 'digest-0001')
    assert path == str(first.resolve())
    assert reason == 'found exact identity-matched mission dataset'

    second = tmp_path / 'scan_20260827_120001'
    second.mkdir()
    (second / 'metadata.yaml').write_text(
        (first / 'metadata.yaml').read_text(encoding='utf-8'),
        encoding='utf-8')
    path, reason = find_failed_mission_dataset(
        tmp_path, 'task-0001', 'digest-0001')
    assert path == ''
    assert reason == 'multiple identity-matched mission datasets exist'


def test_mission_node_keeps_resource_compatibility_objects():
    """Keep downstream imports from the established ROS module working."""
    assert mission_node.calibration_identity_for_mission is (
        calibration_identity_for_mission)
    assert mission_node.find_failed_mission_dataset is (
        find_failed_mission_dataset)
    assert mission_node.previous_generation_cleanup_targets is (
        previous_generation_cleanup_targets)
