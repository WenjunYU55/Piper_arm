import yaml

from results_campaign.collector import (
    _deduplicate_sources,
    _frame_summary,
    _resolve_frame_artifact,
    _target_mask_summary,
)


def test_frame_summary_never_mixes_native_target_with_colour_transform(tmp_path):
    path = tmp_path / 'view_000_metadata.yaml'
    path.write_text(yaml.safe_dump({
        'frame_index': 0,
        'camera_transform': {
            'available': True,
            'child_frame_id': 'camera_color_optical_frame',
            'translation_m': [0.0, 0.0, 0.0],
            'matrix_4x4': [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        'target_3d': {
            'available': True,
            'header': {'frame_id': 'camera_color_optical_frame'},
            'point': {'x': 1.0, 'y': 2.0, 'z': 3.0},
        },
        'synchronized_target_3d': {
            'available': True,
            'header': {'frame_id': 'camera_depth_optical_frame'},
            'point': {'x': 9.0, 'y': 9.0, 'z': 9.0},
        },
    }), encoding='utf-8')
    summary = _frame_summary(path)
    assert summary['target_point_camera_m'] == [1.0, 2.0, 3.0]


def test_target_summary_preserves_acquisition_prompt_context(tmp_path):
    path = tmp_path / 'result.yaml'
    path.write_text('status: ok\n', encoding='utf-8')
    summary = _target_mask_summary({
        'request_id': 'acquire-0',
        'requested_target_label': 'green cube',
        'target_profile': 'green_cube',
        'target_prompt': 'green cube .',
        'target_source': 'grounding_dino',
        'obstacle_count': 2,
        'mission_context': {'request_reason': 'rough_acquisition'},
    }, path)
    assert summary['requested_target_label'] == 'green cube'
    assert summary['target_prompt'] == 'green cube .'
    assert summary['target_source'] == 'grounding_dino'
    assert summary['obstacle_count'] == 2


def test_capture_artifacts_are_resolved_and_bookmarked_without_mutation(
        tmp_path):
    dataset = tmp_path / 'scan'
    frames = dataset / 'frames'
    frames.mkdir(parents=True)
    depth = frames / 'target_depth.png'
    depth.write_bytes(b'depth-evidence')
    before = depth.read_bytes()
    assert _resolve_frame_artifact(dataset, 'target_depth.png') == depth
    sources = _deduplicate_sources([
        {'absolute_path': str(depth), 'sha256': 'a'},
        {'absolute_path': str(depth), 'sha256': 'a'},
    ])
    assert len(sources) == 1
    assert depth.read_bytes() == before
