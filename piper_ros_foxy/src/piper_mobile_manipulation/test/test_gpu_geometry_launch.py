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
