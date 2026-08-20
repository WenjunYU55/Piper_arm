from pathlib import Path


def test_every_gpu_geometry_node_has_exit_supervision():
    source = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'gpu_geometry.launch.py'
    ).read_text(encoding='utf-8')

    assert 'geometry_nodes = [' in source
    assert 'for node in geometry_nodes' in source
    assert 'OnProcessExit(' in source
    assert "reason='critical GPU geometry component exited'" in source
    assert 'LaunchDescription([*shutdown_handlers, *geometry_nodes])' in source


def test_diagnostic_landmark_cannot_start_production_semantic_refreshes():
    source = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'gpu_geometry.launch.py'
    ).read_text(encoding='utf-8')

    assert "'request_refresh_on_new_view': False" in source
    assert "'request_refresh_on_mask_disagreement': False" in source
