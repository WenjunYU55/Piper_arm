import json
from pathlib import Path

import numpy as np
import pytest

from piper_gui.ray_review_model import (
    UrdfAssembly,
    assembly_triangle_count,
    event_cycles,
    load_capability_view,
    load_diagnostic_document,
    load_optical_registration,
    revolved_envelope_mesh,
    filter_rays,
    state_at_event,
)
from piper_gui.ray_reports import RayReviewProcess


ROOT = Path(__file__).resolve().parent


def test_event_scrubbing_never_leaks_a_later_rank(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'schema_version': 2,
        'events': [{
            'event_id': 'generate', 'sequence': 0, 'timestamp_ns': 1,
            'accepted_view_cycle': 0, 'planner_revision': 0,
            'stage': 'generate',
            'ray_deltas': {'7': {'ray_id': 7, 'direction': [1, 0, 0]}},
        }, {
            'event_id': 'rank', 'sequence': 1, 'timestamp_ns': 2,
            'accepted_view_cycle': 0, 'planner_revision': 0,
            'stage': 'seed_rank',
            'ray_deltas': {'7': {'ray_id': 7, 'rank': 1}},
        }],
        'generations': [],
    }), encoding='utf-8')
    document = load_diagnostic_document(report)

    assert 'rank' not in state_at_event(document, 0)['rays'][7]
    assert state_at_event(document, 1)['rays'][7]['rank'] == 1


def test_qualified_population_event_replaces_bootstrap_ray_state(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'schema_version': 2,
        'events': [{
            'event_id': 'bootstrap', 'timestamp_ns': 1,
            'stage': 'prequalify',
            'ray_population_sha256': 'a' * 64,
            'ray_deltas': {'7': {
                'ray_id': 7, 'direction': [1, 0, 0],
                'status': 'culled', 'culled': True,
            }},
        }, {
            'event_id': 'qualified', 'timestamp_ns': 2,
            'stage': 'upgrade_population',
            'ray_population_sha256': 'b' * 64,
            'ray_population_phase': 'qualified',
            'population_reset': True,
            'target_center_m': [0.35, -0.01, 0.14],
            'ray_deltas': {'7': {
                'ray_id': 7, 'direction': [0, 1, 0],
            }},
        }],
        'generations': [],
    }), encoding='utf-8')

    state = state_at_event(load_diagnostic_document(report), 1)

    assert state['target_center_m'] == [0.35, -0.01, 0.14]
    assert state['rays'][7]['direction'] == [0, 1, 0]
    assert state['rays'][7].get('culled') is not True


def test_prequalification_scrub_preserves_requested_and_supported_bounds(
        tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'schema_version': 2,
        'events': [{
            'event_id': 'generate', 'sequence': 0, 'timestamp_ns': 1,
            'accepted_view_cycle': 0, 'planner_revision': 0,
            'stage': 'generate', 'ray_deltas': {'7': {
                'ray_id': 7, 'direction': [1, 0, 0],
                'minimum_standoff_m': 0.28,
                'maximum_standoff_m': 0.80,
            }},
        }, {
            'event_id': 'prequalify', 'sequence': 1, 'timestamp_ns': 2,
            'accepted_view_cycle': 0, 'planner_revision': 0,
            'stage': 'prequalify', 'ray_deltas': {'7': {
                'ray_id': 7, 'requested_minimum_standoff_m': 0.28,
                'requested_maximum_standoff_m': 0.80,
                'minimum_standoff_m': 0.30,
                'maximum_standoff_m': 0.54,
                'capability_intervals_m': [[0.30, 0.34], [0.50, 0.54]],
                'capability_bounded': True,
            }},
        }],
        'generations': [],
    }), encoding='utf-8')
    document = load_diagnostic_document(report)

    generated = state_at_event(document, 0)['rays'][7]
    bounded = state_at_event(document, 1)['rays'][7]
    assert generated['maximum_standoff_m'] == 0.80
    assert 'capability_intervals_m' not in generated
    assert bounded['requested_maximum_standoff_m'] == 0.80
    assert bounded['maximum_standoff_m'] == 0.54
    assert bounded['capability_intervals_m'] == [
        [0.30, 0.34], [0.50, 0.54]]


def test_target_envelope_appears_only_when_recorded_by_the_event(tmp_path):
    envelope = {'envelope_sha256': 'a' * 64, 'collision_boxes': []}
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'schema_version': 2,
        'events': [{
            'event_id': 'before', 'timestamp_ns': 1,
            'accepted_view_cycle': 0, 'stage': 'detect',
        }, {
            'event_id': 'generate', 'timestamp_ns': 2,
            'accepted_view_cycle': 0, 'stage': 'generate',
            'target_envelope': envelope,
        }],
        'generations': [],
    }), encoding='utf-8')
    document = load_diagnostic_document(report)

    assert state_at_event(document, 0)['target_envelope'] is None
    assert state_at_event(document, 1)['target_envelope'] == envelope


def test_revolved_envelope_mesh_follows_recorded_axis_and_radius():
    vertices, faces = revolved_envelope_mesh({
        'axis_origin_m': [0.1, -0.2, 0.3],
        'axis_direction': [0.0, 0.0, 1.0],
        'profile_sections': [
            {'center_s_m': -0.02, 'half_length_m': 0.01,
             'radius_m': 0.01},
            {'center_s_m': 0.0, 'half_length_m': 0.01,
             'radius_m': 0.02},
            {'center_s_m': 0.02, 'half_length_m': 0.01,
             'radius_m': 0.01},
        ],
    }, angular_segments=16)

    assert vertices.shape == (82, 3)
    assert faces.shape == (160, 3)
    assert np.all(np.isfinite(vertices))
    assert np.min(vertices[:, 2]) == pytest.approx(0.27)
    assert np.max(vertices[:, 2]) == pytest.approx(0.33)
    radial = np.linalg.norm(vertices[:-2, :2] - [0.1, -0.2], axis=1)
    assert np.max(radial) == pytest.approx(0.02)


def test_revolved_envelope_presentation_mesh_rejects_malformed_evidence():
    vertices, faces = revolved_envelope_mesh({
        'axis_origin_m': [0.0, 0.0, 0.0],
        'axis_direction': [0.0, 0.0, 0.0],
        'profile_sections': [],
    })

    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)


def test_unranked_generation_rays_remain_visible_without_rank_filter():
    state = {'rays': {
        1: {'ray_id': 1, 'direction': [1, 0, 0]},
        2: {'ray_id': 2, 'direction': [0, 1, 0]},
    }}

    assert [ray['ray_id'] for ray in filter_rays(
        state, rank_max=None)] == [1, 2]


def test_cycle_blocks_preserve_late_terminal_journal_order():
    document = {'events': [
        {'event_id': 'cycle-0', 'accepted_view_cycle': 0},
        {'event_id': 'cycle-1', 'accepted_view_cycle': 1},
        {'event_id': 'terminal', 'accepted_view_cycle': 0},
    ]}

    blocks = event_cycles(document)

    assert [block['cycle'] for block in blocks] == [0, 1, 0]
    assert [event['event_id'] for block in blocks
            for event in block['events']] == [
                'cycle-0', 'cycle-1', 'terminal']


def test_v1_loading_is_truthfully_partial(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'schema_version': 1,
        'generations': [{'session_id': 'old', 'generation': 0, 'rays': []}],
    }), encoding='utf-8')
    document = load_diagnostic_document(report)

    assert document['legacy_v1'] is True
    assert document['journal_complete'] is False
    assert [item['stage'] for item in document['events']] == ['legacy_snapshot']


def test_v1_scrubbing_recovers_its_recorded_target_center(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'schema_version': 1,
        'generations': [{
            'generation': 0,
            'target_center_m': [0.41, -0.02, 0.16],
            'rays': [],
        }],
    }), encoding='utf-8')

    state = state_at_event(load_diagnostic_document(report), 0)

    assert state['target_center_m'] == [0.41, -0.02, 0.16]


def test_viewer_source_contains_no_ros_or_motion_ownership():
    source = (ROOT / 'piper_gui/ray_review_viewer.py').read_text(encoding='utf-8')
    forbidden = ('import rclpy', 'create_publisher', 'create_client',
                 'ActionClient', 'Enable', 'MoveJ')
    assert not any(value in source for value in forbidden)


def test_viewer_navigation_and_ground_regressions_are_present():
    source = (ROOT / 'piper_gui/ray_review_viewer.py').read_text(
        encoding='utf-8')

    assert 'axes.SetTotalLength(0.07, 0.07, 0.07)' in source
    assert source.count('vtk.vtkInteractorStyleTrackballCamera()') == 2
    assert source.count('SetMouseWheelMotionFactor(1.2)') == 2
    assert 'event.type() == QtCore.QEvent.Wheel' not in source
    assert 'QtCore.Qt.Key_Q' in source
    assert 'QtCore.Qt.Key_E' in source
    assert 'vtk.vtkPlaneSource()' in source
    assert "visual.link == 'bunker_chassis_collision'" in source
    assert '_keep_camera_above_ground' in source


def test_rank_colours_and_transient_culls_are_explained_in_the_viewer():
    source = (ROOT / 'piper_gui/ray_review_viewer.py').read_text(
        encoding='utf-8')

    assert 'def _rank_color(' in source
    assert 'RANK_PALETTE' in source
    assert 'blue-to-red quality spectrum' in source
    assert 'Keep culled rays visible' in source
    assert 'self.show_culled.setChecked(True)' in source
    assert 'self.inspector.show_culled.setChecked(True)' in source
    assert 'target/ray centre' in source
    assert "' · %s population' % phase" in source
    assert 'Show standoff bounds' in source
    assert 'grey dashed = requested' in source
    assert 'cyan = capability-supported' in source
    assert 'camera meshes = post-cull survivors only' in source
    assert 'population_has_culls = any(' in source
    assert 'if not _is_culled(ray) and not ray.get' in source
    assert 'for ray in camera_rays:' in source
    assert 'target envelope' in source
    assert 'def _add_target_envelope(' in source
    assert 'Estimated revolved model' in source
    assert 'Conservative planning boxes' in source
    assert 'Original mask/depth outline' in source
    assert "QtWidgets.QGroupBox('Target layers')" in source
    assert source.index("QtWidgets.QGroupBox('Target layers')") < source.index(
        "form.addRow('Cull stage'")
    assert 'def _add_revolved_model(' in source
    assert 'def _add_source_outline(' in source
    assert 'culled now' in source
    assert 'thick line = selected' in source
    assert 'and not show_past_culled' in source
    assert "['Focused review', 'Full ray lifecycle']" in source
    assert 'Restart lifecycle' in source
    assert 'Partial historical evidence:' in source
    assert 'eliminated now:' in source
    assert "'eliminated_ray_count'" in source
    assert "parser.add_argument('--full-lifecycle'" in source


def test_complete_checked_in_urdf_visual_assembly_is_parsed():
    assembly = UrdfAssembly(ROOT / (
        'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro'))
    links = {item.link for item in assembly.visuals}

    assert {'bunker_chassis_collision', 'bunker_sensor_station_collision',
            'base_link', 'gripper_base', 'link7', 'link8',
            'camera_holder', 'l515_visual'}.issubset(links)
    assert assembly_triangle_count(assembly) == 638780


def test_camera_mesh_uses_accepted_optical_registration():
    transform = load_optical_registration(ROOT / (
        'L515_camera/calibration/hand_eye/session_20260808_straight_mount/'
        'calibration_result.yaml'))

    assert np.allclose(transform[:3, 3], [
        -0.00020008129649795418,
        0.014131228439509869,
        -0.004000160173675537,
    ])


def test_capability_map_load_is_read_only_and_decodes_committed_counts():
    path = ROOT / (
        'piper_ros_foxy/src/piper_mobile_manipulation/config/'
        'piper_camera_capability_map.npz')
    before = (path.stat().st_mtime_ns, path.read_bytes()[:64])
    view = load_capability_view(path)
    after = (path.stat().st_mtime_ns, path.read_bytes()[:64])

    assert len(view.positions_m) == 118370
    assert view.occupied_pose_direction_bins == 1479561
    assert before == after
    assert view.direction_histogram.shape == (18, 36)


class _FakeStdin:
    def __init__(self):
        self.values = []

    def write(self, value):
        self.values.append(value)

    def flush(self):
        pass


class _FakeProcess:
    def __init__(self, *args, **kwargs):
        self.stdin = _FakeStdin()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_process_manager_reuses_one_child_and_sends_stdin_json(tmp_path):
    root = tmp_path
    for name in ('one', 'two'):
        report = (root / 'datasets/active_scan/ray_diagnostics' / name)
        report.mkdir(parents=True)
        (report / 'ray_mission_diagnostics.json').write_text('{}')
    children = []

    def factory(*args, **kwargs):
        child = _FakeProcess(*args, **kwargs)
        children.append(child)
        return child

    manager = RayReviewProcess(root, process_factory=factory)

    manager.open('one')
    manager.open('two')

    assert len(children) == 1
    assert [json.loads(value)['command'] for value in children[0].stdin.values] == [
        'open', 'open']
    manager.shutdown()
    assert json.loads(children[0].stdin.values[-1])['command'] == 'shutdown'
