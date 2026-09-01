"""Pure tests for command-free planner benchmark bookkeeping."""

import pytest

from motion_planning.benchmarking import (
    materialize_request, numeric_summary, scenario_sha256,
    summarize_trials, trajectory_metrics)


def template():
    return {
        'schema_version': 5,
        'planner_backend': 'tesseract',
        'request_id': 'a' * 32,
        'created_at_ns': 1,
        'expires_at_ns': 2,
        'request_sha256': 'b' * 64,
        'planning': {
            'planner': 'RRTConnect',
            'pipeline': 'OMPL_ISP',
            'effective_speed_percent': 5.0,
        },
        'scene': {'candidate_views': [{'id': 7}]},
    }


def test_materialization_changes_only_backend_and_volatile_identity():
    original = template()
    first = materialize_request(original, 'tesseract', 1, now_ns=100)
    second = materialize_request(original, 'curobo', 1, now_ns=100)

    assert original == template()
    assert scenario_sha256(first) == scenario_sha256(second)
    assert first['planner_backend'] == 'tesseract'
    assert second['planner_backend'] == 'curobo'
    assert first['planning']['planner'] == 'RRTConnect'
    assert second['planning']['planner'] == 'MotionGen'
    assert first['request_id'] != second['request_id']
    assert first['request_sha256'] != second['request_sha256']


def test_materialization_is_reproducible_for_same_run_identity():
    first = materialize_request(template(), 'curobo', 3, now_ns=500)
    second = materialize_request(template(), 'curobo', 3, now_ns=500)

    assert first == second


def test_trajectory_metrics_use_the_common_scheduled_path():
    response = {
        'selected_viewpoints': [{'id': 1}],
        'segments': [{'points': [
            {'time_from_start_s': 0.0, 'positions_rad': [0.0] * 6},
            {'time_from_start_s': 1.0,
             'positions_rad': [0.1, -0.2, 0.0, 0.0, 0.0, 0.0]},
        ]}],
        'planning_diagnostics': {
            'planning_duration_sec': 0.25,
            'candidate_viewpoints_considered': 3,
            'candidate_viewpoints_rejected': 2,
            'feasible_viewpoints': 1,
        },
    }

    metrics = trajectory_metrics(response)

    assert metrics['trajectory_duration_sec'] == pytest.approx(1.0)
    assert metrics['joint_space_path_length_rad'] == pytest.approx(0.3)
    assert metrics['maximum_joint_step_rad'] == pytest.approx(0.2)
    assert metrics['trajectory_point_count'] == 2
    assert metrics['candidate_viewpoints_considered'] == 3


def test_summary_excludes_warmup_and_requires_exact_validation():
    base = {
        'backend': 'curobo', 'status': 'success', 'warmup': False,
        'expected_role': 'recorded_achieved_geometry',
        'request_wall_sec': 0.2, 'backend_reported_planning_sec': 0.1,
        'trajectory_duration_sec': 2.0,
        'joint_space_path_length_rad': 1.0,
        'trajectory_point_count': 20,
        'exact_collision_validation': 'passed',
    }
    trials = [base, dict(base, warmup=True, request_wall_sec=99.0),
              dict(base, status='failed', exact_collision_validation=(
                  'not_applicable'))]

    result = summarize_trials(trials)['curobo']

    assert result['trial_count'] == 2
    assert result['success_rate'] == pytest.approx(0.5)
    assert result['exact_validated_success_rate'] == pytest.approx(1.0)
    assert result['request_wall_sec']['maximum'] == pytest.approx(0.2)


def test_negative_control_rejection_is_not_a_planning_failure():
    base = {
        'backend': 'tesseract', 'warmup': False,
        'request_wall_sec': 0.2, 'backend_reported_planning_sec': 0.1,
        'trajectory_duration_sec': 2.0,
        'joint_space_path_length_rad': 1.0,
        'trajectory_point_count': 20,
    }
    rows = [
        dict(base, status='success',
             expected_role='recorded_achieved_geometry',
             exact_collision_validation='passed'),
        dict(base, status='failed', expected_role='negative_control',
             exact_collision_validation='not_applicable'),
    ]

    summary = summarize_trials(rows)['tesseract']

    assert summary['success_rate'] == pytest.approx(1.0)
    assert summary['negative_control_rejection_rate'] == pytest.approx(1.0)


def test_numeric_summary_ignores_unknown_negative_diagnostics():
    result = numeric_summary([-1.0, None, 0.2, 0.4])

    assert result['n'] == 2
    assert result['median'] == pytest.approx(0.3)
