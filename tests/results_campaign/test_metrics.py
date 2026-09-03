from results_campaign.metrics import (
    circular_span_deg,
    cloud_dimension_metrics,
    cube_surface_points,
    cumulative_cube_coverage,
    summarize_evidence,
    stamp_delta_sec,
)


def test_circular_span_crosses_zero_without_false_full_turn():
    assert abs(circular_span_deg([350.0, 5.0, 10.0]) - 20.0) < 1e-9


def test_capture_timestamp_delta_is_fail_closed():
    assert stamp_delta_sec(3_000_000_000, 1_000_000_000) == 2.0
    assert stamp_delta_sec(1, 2) is None


def test_cube_coverage_is_monotonic_and_complete_for_reference_points():
    cube = cube_surface_points()
    split = len(cube) // 2
    rows = cumulative_cube_coverage([cube[:split], cube[split:]], (0, 0, 0))
    assert rows[0]['cumulative_coverage_fraction'] < 1.0
    assert rows[1]['cumulative_coverage_fraction'] == 1.0
    assert rows[1]['cumulative_surface_points'] >= rows[0]['cumulative_surface_points']


def test_point_cloud_dimensions_use_declared_robust_estimator():
    cube = cube_surface_points()
    result = cloud_dimension_metrics(cube)
    assert result['qualified'] is True
    assert result['mean_absolute_dimension_error_mm'] < 1e-6


def test_summary_reports_qualification_and_ray_planning_events():
    rows = summarize_evidence({
        'campaign_id': 'campaign',
        'trial_id': 'trial',
        'task_id': 'task',
        'planner_backend': 'curobo',
        'evidence_class': 'PAIRED_PHYSICAL',
        'matches_schedule': True,
        'submission': {
            'submitted_wall_time_ns': 1_000_000_000,
            'expected_trial': {
                'pair_index': 1,
                'x_m': 0.3,
                'y_m': 0.0,
                'z_m': 0.0,
            },
            'configuration_snapshot': {
                'git': {},
                'files': [],
                'planner_models': {'curobo': {
                    'hardware_qualified': True,
                    'qualification_date': '2026-09-02',
                    'qualification_scope': (
                        'supervised_5_percent_target_scan'),
                    'qualification_basis': (
                        'operator_reported_physical_e2e'),
                    'conservative_geometry': False,
                }},
            },
        },
        'mission_result': {'capture_count': 0, 'action_summary': {}},
        'terminal': {},
        'frames': [],
        'ray_diagnostics': {'planning_events': [{
            'timestamp_ns': 2_000_000_000,
            'message': 'prequalified rays',
            'planner_revision': 1,
            'accepted_view_cycle': 0,
            'metrics': {
                'generated_ray_count': 360,
                'surviving_ray_count': 44,
                'planning_duration_sec': 1.2,
            },
        }]},
    })
    run = rows['Runs'][0]
    assert run['collision_model_hardware_qualified'] is True
    assert run['collision_model_qualification_scope'] == (
        'supervised_5_percent_target_scan')
    event = rows['Planning'][0]
    assert event['record_type'] == 'ray_event'
    assert event['generated_ray_count'] == 360
    assert event['surviving_ray_count'] == 44
