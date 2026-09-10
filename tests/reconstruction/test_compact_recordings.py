"""Exercise destructive compaction against serialized, split Foxy bags."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compact_recordings', ROOT / 'tools/bundlesdf/compact_recordings.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image
    from piper_mobile_manipulation.msg import ScanExecutionStatus
    from piper_mobile_manipulation.movement_recording import SQLiteBagWriter
    archive = tmp_path / 'archive.closed'
    archive.mkdir()
    writer = SQLiteBagWriter(archive / 'bag')
    for topic, kind in [(module.RGB, 'sensor_msgs/msg/Image'), (module.DEPTH, 'sensor_msgs/msg/Image'),
                        (module.EXECUTION, 'piper_mobile_manipulation/msg/ScanExecutionStatus'),
                        ('/tf', 'tf2_msgs/msg/TFMessage'), ('/piper/reconstruction_keyframe', 'piper_mobile_manipulation/msg/ReconstructionKeyframe')]:
        writer.create_topic(topic, kind)
    for sec, state in [(3, 'MOVING'), (4, 'SETTLING')]:
        msg = ScanExecutionStatus(state=state)
        writer.write(module.EXECUTION, serialize_message(msg), sec * 10**9)
    # Window [2,5]; a depth frame arrives outside the window but matches RGB.
    for stamp, rgb_time, depth_time in [(1, 1, 1), (2, 2, 1.9), (3, 3, 3), (5, 5, 5.1), (7, 7, 7)]:
        msg = Image()
        msg.header.stamp.sec = stamp
        msg.data = bytes(1024)
        for topic, received in [(module.RGB, rgb_time), (module.DEPTH, depth_time)]:
            writer.write(topic, serialize_message(msg), int(received * 10**9))
    msg.header.stamp.sec = 4  # Unpaired in-window RGB must not masquerade as a frameset.
    writer.write(module.RGB, serialize_message(msg), 4 * 10**9)
    writer.write('/tf', b'preserve transform bytes', 8 * 10**9)
    writer.write('/piper/reconstruction_keyframe', b'preserve entire idle accepted tuple', 9 * 10**9)
    writer.close()
    first = archive / 'bag/bag_0.db3'
    second = archive / 'bag/bag_1.db3'
    with sqlite3.connect(str(first)) as db, sqlite3.connect(str(second)) as other:
        db.backup(other)
        # Put RGB and depth in separate splits to require global stamp matching.
        db.execute('DELETE FROM messages WHERE topic_id=2')
        other.execute('DELETE FROM messages WHERE topic_id!=2')
    db.close()
    other.close()
    metadata_path = archive / 'bag/metadata.yaml'
    metadata = yaml.safe_load(metadata_path.read_text())
    metadata['rosbag2_bagfile_information']['relative_file_paths'].append('bag_1.db3')
    metadata_path.write_text(yaml.safe_dump(metadata))
    (archive / 'recording.json').write_text(json.dumps({'state': 'closed', 'task_id': 'test'}))
    return archive


def test_compaction_pairs_across_splits_and_preserves_evidence(tmp_path):
    archive = fixture(tmp_path)
    before = [p.read_bytes() for p in sorted((archive / 'bag').glob('*.db3'))]
    plan = module.compact(archive)
    assert plan['paired_frames'] == 3
    assert plan['selected_unpaired_stamps'] == 1
    assert before == [p.read_bytes() for p in sorted((archive / 'bag').glob('*.db3'))]
    report = module.compact(archive, True)
    assert report['state'] == 'complete'
    stamps = {module.RGB: [], module.DEPTH: []}
    preserved = []
    total = 0
    for path in (archive / 'bag').glob('*.db3'):
        with module.read_db(path) as db:
            for topic, data in db.execute('SELECT topics.name,messages.data FROM messages JOIN topics ON topics.id=messages.topic_id'):
                total += 1
                if topic in stamps:
                    stamps[topic].append(module.image_stamp(data[:12]))
                elif topic != module.EXECUTION:
                    preserved.append(data)
    assert sorted(stamps[module.RGB]) == sorted(stamps[module.DEPTH]) == [2*10**9, 3*10**9, 5*10**9]
    assert set(preserved) == {b'preserve transform bytes', b'preserve entire idle accepted tuple'}
    assert yaml.safe_load((archive / 'bag/metadata.yaml').read_text())['rosbag2_bagfile_information']['message_count'] == total
    assert json.loads((archive / 'recording.json').read_text())['stored_messages'] == total
    assert module.compact(archive, True)['saved_bytes'] == report['saved_bytes']


def test_interrupted_replace_resumes(tmp_path, monkeypatch):
    archive = fixture(tmp_path)
    replace = module.replace_segment
    calls = []

    def interrupted(path, segment):
        replace(path, segment)
        if not calls:
            calls.append(True)
            raise RuntimeError('simulated interruption after atomic replacement')
    monkeypatch.setattr(module, 'replace_segment', interrupted)
    with pytest.raises(RuntimeError):
        module.compact(archive, True)
    monkeypatch.setattr(module, 'replace_segment', replace)
    assert module.compact(archive, True)['state'] == 'complete'


def test_refuse_missing_movement_or_live_archive(tmp_path):
    archive = fixture(tmp_path)
    with sqlite3.connect(str(archive / 'bag/bag_0.db3')) as db:
        db.execute('DELETE FROM messages WHERE topic_id=3')
    db.close()
    with pytest.raises(ValueError, match='No execution'):
        module.compact(archive, True)
    (archive / 'recording.json').write_text('{"state":"recording"}')
    with pytest.raises(ValueError, match='Live'):
        module.compact(archive, True)


def test_validation_failure_never_replaces_original(tmp_path, monkeypatch):
    archive = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in (archive / 'bag').glob('*.db3')}
    monkeypatch.setattr(module, 'digest', lambda db: (0, 'invalid'))
    with pytest.raises(ValueError, match='payload verification'):
        module.compact(archive, True)
    assert before == {p.name: p.read_bytes() for p in (archive / 'bag').glob('*.db3')}


def test_no_scratch_space_leaves_originals(tmp_path, monkeypatch):
    from collections import namedtuple
    archive = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in (archive / 'bag').glob('*.db3')}
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: usage(100, 100, 0))
    with pytest.raises(ValueError, match='Insufficient scratch'):
        module.compact(archive, True)
    assert before == {p.name: p.read_bytes() for p in (archive / 'bag').glob('*.db3')}


def test_dense_extraction_refuses_unfinished_compaction(tmp_path):
    import subprocess
    archive = fixture(tmp_path)
    (archive / 'movement_compaction.json').write_text('{"state":"compacting"}')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/bundlesdf/extract_video.py'),
                             str(archive), str(tmp_path / 'output')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Archive compaction is unfinished' in result.stderr
    assert not (tmp_path / 'output').exists()
