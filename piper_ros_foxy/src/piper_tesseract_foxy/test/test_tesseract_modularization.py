"""Compatibility checks for decomposed bridge, contract, and worker code."""

from types import SimpleNamespace

import piper_tesseract_foxy.bridge_candidates as candidates
import piper_tesseract_foxy.bridge_node as bridge
import piper_tesseract_foxy.contract as contract
import piper_tesseract_foxy.contract_hashing as hashing
import piper_tesseract_foxy.contract_request as request_validation
import piper_tesseract_foxy.contract_response as response_validation
from piper_tesseract_foxy.contract_spool import Spool
from piper_tesseract_foxy.worker_components import WorkerOrchestrator


def test_bridge_candidate_helpers_keep_legacy_import_identity():
    assert bridge.target_envelope_obstacles is candidates.target_envelope_obstacles
    assert bridge.obstacle_scene_rejection_reason is (
        candidates.obstacle_scene_rejection_reason)
    assert bridge.validate_candidate_policy_batch is (
        candidates.validate_candidate_policy_batch)
    assert bridge.information_ranked_ray_candidates is (
        candidates.information_ranked_ray_candidates)


def test_contract_facade_reexports_hashing_and_spool_owners():
    assert contract.canonical_bytes is hashing.canonical_bytes
    assert contract.sha256_value is hashing.sha256_value
    assert contract.sha256_file is hashing.sha256_file
    assert contract.attach_digest is hashing.attach_digest
    assert contract.verify_digest is hashing.verify_digest
    assert contract.Spool is Spool
    assert contract.validate_request is request_validation.validate_request
    assert contract.validate_response is response_validation.validate_response


def test_canonical_bytes_fixture_is_unchanged():
    payload = {'schema_version': 5, 'finite': 1.25, 'nested': {'b': 2, 'a': 1}}
    expected = (
        b'{"finite":1.25,"nested":{"a":1,"b":2},"schema_version":5}')

    assert contract.canonical_bytes(payload) == expected
    assert contract.sha256_value(payload) == (
        'd70e3bda87ae10a9418bd907b5b89e02cd2d4bb6fb89f94b609eb627ce6f421a')


def test_worker_orchestrator_exposes_explicit_components():
    class Backend:
        last_planning_diagnostics = {'attempts': 2}

    orchestrator = WorkerOrchestrator(Backend())

    assert orchestrator.last_planning_diagnostics == {'attempts': 2}
    assert orchestrator.collision_scene.backend is orchestrator.backend
    assert orchestrator.aim_solver.backend is orchestrator.backend
    assert orchestrator.trajectory_planner.backend is orchestrator.backend


def test_worker_orchestrator_routes_live_plan_through_components():
    events = []
    backend = SimpleNamespace(
        reset_scene=lambda: events.append('scene_reset'),
        add_obstacles=lambda obstacles: events.append(
            ('obstacles', list(obstacles))),
        find_bootstrap_recovery=lambda request: (
            events.append('bootstrap_recovery') or None),
    )
    orchestrator = WorkerOrchestrator(backend)

    def plan_candidate(current, candidate, rolls, maximum_step,
                       position_limits, joint_margin,
                       bootstrap_recovery, visibility_target=None):
        events.append(('aim', int(candidate['id'])))
        return (
            dict(candidate), 0.0,
            [{'positions_rad': [0.0] * 6}],
            {'minimum_clearance_m': 0.01}, False, 0.0)

    orchestrator.aim_solver.plan_candidate_aims = plan_candidate
    request = {
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'obstacles': [],
            'candidate_views': [{
                'id': 7,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
                'required_first': True,
            }],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'max_execution_joint_step_rad': 0.1,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 5.0,
            'command_rate_hz': 20.0,
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
        },
        'start_state': {'positions_rad': [0.0] * 6},
    }

    selected, segments = orchestrator.plan(request)

    assert [item['id'] for item in selected] == [7]
    assert [item['to_viewpoint'] for item in segments] == [7]
    assert events == [
        'scene_reset', ('obstacles', []), 'bootstrap_recovery', ('aim', 7)]
