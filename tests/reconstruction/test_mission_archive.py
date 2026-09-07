"""Whole-scan archive bounds and lifecycle without touching a robot."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from piper_mobile_manipulation import mission_archive as archive


class StopImmediately:
    def wait(self, _delay):
        return True


@pytest.mark.parametrize('stored_count', [2, 3])
@pytest.mark.parametrize('exit_code', [0, 2])
def test_archive_closes_and_keeps_success_distinct_from_mission_outcome(
        tmp_path, monkeypatch, exit_code, stored_count):
    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = exit_code

    def start(command, **kwargs):
        bag = Path(command[command.index('--output') + 1])
        bag.mkdir()
        import sqlite3
        with sqlite3.connect(str(bag / 'bag.db3')) as connection:
            connection.execute('CREATE TABLE messages (id INTEGER)')
            connection.executemany('INSERT INTO messages VALUES (?)',
                                   [(i,) for i in range(stored_count)])
        (bag / 'metadata.yaml').write_text(
            'rosbag2_bagfile_information:\n  message_count: 3\n'
            '  topics_with_message_count: []\n  relative_file_paths: [bag.db3]\n')
        assert kwargs['start_new_session']
        assert '--all' not in command and '-a' not in command
        assert command[command.index('--max-cache-size') + 1] == '0'
        return Process()

    monkeypatch.setattr(archive.subprocess, 'Popen', start)
    monkeypatch.setattr(archive.os, 'killpg', lambda *_: None)
    final = archive.record(tmp_path, 0, 'mission-one', stop=StopImmediately())
    manifest = json.loads((final / 'recording.json').read_text())
    assert final.suffix == ('.closed' if stored_count == 3 else '.incomplete')
    assert manifest['recorded_messages'] == 3 and manifest['stored_messages'] == stored_count
    assert manifest['dropped_messages'] is None
    assert 'not mission success' in manifest['mission_outcome']
    again = archive.record(tmp_path, 0, 'mission-two', stop=StopImmediately())
    assert again != final and final.is_dir()


def test_archive_refuses_low_space_and_retains_launch_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(archive.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='free disk'):
        archive.record(tmp_path, 2, '')
    assert not list(tmp_path.iterdir())

    def fail(*_, **__):
        raise OSError('disk full')

    monkeypatch.setattr(archive.subprocess, 'Popen', fail)
    final = archive.record(tmp_path, 0, '')
    assert final.suffix == '.incomplete'
    assert 'disk full' in json.loads((final / 'recording.json').read_text())['stop_reason']


def test_archive_has_only_replay_evidence_topics_and_nonblocking_sensor_qos(tmp_path, monkeypatch):
    monkeypatch.setattr(archive.subprocess, 'Popen',
                        lambda *_, **__: (_ for _ in ()).throw(OSError('missing ros2')))
    final = archive.record(tmp_path, 0, '')
    qos = archive.yaml.safe_load((final / 'qos.yaml').read_text())
    assert '/tf_static' in archive.TOPICS
    assert '/piper/reconstruction_keyframe' in archive.TOPICS
    for name, profile in qos.items():
        if name != '/tf_static':
            assert profile['reliability'] == 'best_effort' and profile['depth'] == 8
    assert qos['/tf_static']['durability'] == 'transient_local'
    assert not any('command' in name or 'enable' in name for name in archive.TOPICS)


@pytest.mark.parametrize('live,bag,expected', [
    ('false', 'false', False), ('true', 'false', True),
    ('false', 'true', True), ('true', 'true', True),
])
def test_launch_keeps_raw_keyframes_for_archive_only_mode(live, bag, expected):
    import runpy
    from launch import LaunchContext
    from launch.actions import TimerAction
    from launch_ros.utilities import evaluate_parameters

    project = Path(__file__).resolve().parents[2]
    launch_file = (project / 'piper_ros_foxy/src/piper_mobile_manipulation/launch'
                   / 'supervised_viewpoint_execution.launch.py')
    description = runpy.run_path(str(launch_file))['generate_launch_description']()
    for timer in description.entities:
        if not isinstance(timer, TimerAction):
            continue
        for _, nodes in timer.describe_conditional_sub_entities():
            for node in nodes:
                if getattr(node, '_Node__node_executable', None) != 'scan_capture_node.py':
                    continue
                context = LaunchContext()
                context.launch_configurations.update(
                    reconstruction_recording_enabled=live, mission_archive_enabled=bag,
                    max_execution_viewpoints='8')
                parameters = evaluate_parameters(context, [node._Node__parameters[-1]])[0]
                assert parameters['reconstruction_recording_enabled'] is expected
                return
    pytest.fail('capture node absent from production launch')
