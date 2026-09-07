"""Recover serialized ROS evidence with late/missing offered QoS metadata."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest


def test_bag_callback_recovery(tmp_path):
    pytest.importorskip('rclpy')
    # Use a child interpreter so generated message imports cannot be shadowed by
    # source-only packages imported during pure unit-test collection.
    root = Path(__file__).resolve().parents[2]
    script = r'''
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from rclpy.serialization import serialize_message
from piper_mobile_manipulation.msg import ReconstructionKeyframe, ScanExecutionStatus
from sensor_msgs.msg import CameraInfo
root=Path(sys.argv[1]); bag=root/'bag'; bag.mkdir()
(root/'recording.json').write_text(json.dumps(dict(state='closed',task_id='fixture',target_visibility_angle_tolerance_deg=20.)))
(bag/'metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':{'relative_file_paths':['bag_0.db3']}}))
c=sqlite3.connect(str(bag/'bag_0.db3'))
c.executescript('CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT); '
               'CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,'
               'timestamp INTEGER,data BLOB);')
topics={};bridge=CvBridge()
def add(topic,kind,msg):
 if topic not in topics:
  topics[topic]=len(topics)+1
  c.execute('INSERT INTO topics VALUES(?,?,?)',(topics[topic],topic,kind))
 c.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES(?,?,?)',
           (topics[topic],tick,serialize_message(msg)))
for i in range(6):
 tick=1_000_000_000+i*200_000_000
 checker=(np.indices((120,160)).sum(axis=0)%2*255).astype(np.uint8)
 rgb=bridge.cv2_to_imgmsg(np.repeat(checker[:,:,None],3,axis=2),'rgb8')
 depth=bridge.cv2_to_imgmsg(np.full((120,160),500,np.uint16),'16UC1')
 mask=np.zeros((120,160),np.uint8);mask[45:75,65:95]=255
 mask=bridge.cv2_to_imgmsg(mask,'mono8')
 info=CameraInfo(width=160,height=120,k=[200.,0.,80.,0.,200.,60.,0.,0.,1.],distortion_model='plumb_bob',d=[.169,-.514,-.0021,.0013,.475])
 for m in [rgb,depth,mask,info]:
  m.header.frame_id='camera_color_optical_frame';m.header.stamp.sec,m.header.stamp.nanosec=divmod(tick,1_000_000_000)
 tf=TransformStamped();tf.header.stamp=rgb.header.stamp;tf.header.frame_id='base_link';tf.child_frame_id=rgb.header.frame_id;tf.transform.rotation.w=1.
 add('/tf','tf2_msgs/msg/TFMessage',TFMessage(transforms=[tf]))
 if i==0:
  add('/piper/reconstruction_keyframe','piper_mobile_manipulation/msg/ReconstructionKeyframe',ReconstructionKeyframe(task_id='fixture',rgb=rgb,depth=depth,mask=mask,camera_info=info))
  history=dict(session_id='fixture-session',accepted_views=1,coverage_target_center=dict(x=0.,y=0.,z=.5),entries=[dict(accepted_view=1,reconstruction_stamp_ns=tick)])
  add('/piper/scan_session_history','std_msgs/msg/String',String(data=json.dumps(history)))
  add('/piper/scan_execution_status','piper_mobile_manipulation/msg/ScanExecutionStatus',ScanExecutionStatus(execution_mode='MULTIVIEW_SCAN',state='MOVING'))
 else:
  for topic,kind,m in [
    ('camera/color/camera_info','CameraInfo',info),('camera/color/image_raw','Image',rgb),
    ('camera/aligned_depth_to_color/image_raw','Image',depth),
    ('piper/sam2_target_mask','Image',mask)]:
   add('/'+topic,'sensor_msgs/msg/'+kind,m)
c.commit();c.close()
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True)
    output = tmp_path / 'recovered'
    result = subprocess.run(
        [sys.executable, str(root / 'tools/bundlesdf/import_bag.py'),
         str(tmp_path / 'bag'), str(output)],
        env=dict(os.environ, ROS_DOMAIN_ID='177', ROS_LOCALHOST_ONLY='1'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert result.returncode == 0, result.stdout
    report = json.loads((output / 'recovery.json').read_text())
    manifest = json.loads((Path(report['sessions'][0]) / 'manifest.json').read_text())
    assert manifest['accepted'] == 6
    assert manifest['mission_capture_count'] == 1
    assert not manifest['complete']  # No completion event was recorded.
    assert manifest['dropped'] == 0
    assert manifest['frames'][0]['distortion_model'] == 'plumb_bob'
    # Dense extraction must not need later live masks or a MOVING execution state.
    with sqlite3.connect(str(tmp_path / 'bag/bag_0.db3')) as db:
        db.execute("DELETE FROM messages WHERE topic_id IN (SELECT id FROM topics WHERE "
                   "name IN ('/piper/sam2_target_mask','/piper/scan_execution_status'))")
    dense = tmp_path / 'dense'
    extracted = subprocess.run(
        [sys.executable, str(root / 'tools/bundlesdf/extract_video.py'),
         str(tmp_path / 'bag'), str(dense)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=30)
    assert extracted.returncode == 0, extracted.stdout
    data = json.loads((dense / 'extraction.json').read_text())
    assert len(data['frames']) == 6
    assert data['mission_capture_count'] == 1
    assert not data['complete']
    assert data['frames'][0]['rgb_stamp_ns'] == 1_000_000_000
    assert all(f['tf_error'] is None for f in data['frames'])
    assert data['frames'][1]['tf_stamp_ns'] == 1_200_000_000

    # A nearby depth-only frameset must not be relabelled as synchronized RGB-D.
    mutate = r'''import sqlite3, sys
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Image
c=sqlite3.connect(sys.argv[1])
tid=c.execute("SELECT id FROM topics WHERE name='/camera/aligned_depth_to_color/image_raw'").fetchone()[0]
mid,blob=c.execute('SELECT id,data FROM messages WHERE topic_id=? ORDER BY id LIMIT 1',(tid,)).fetchone()
m=deserialize_message(blob,Image);m.header.stamp.nanosec+=11000000
c.execute('UPDATE messages SET data=? WHERE id=?',(serialize_message(m),mid));c.commit()
'''
    subprocess.run([sys.executable, '-c', mutate, str(next((tmp_path / 'bag').glob('*.db3')))], check=True)
    conservative = tmp_path / 'conservative'
    result = subprocess.run([sys.executable, str(root / 'tools/bundlesdf/extract_video.py'),
                             str(tmp_path / 'bag'), str(conservative)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    checked = json.loads((conservative / 'extraction.json').read_text())
    assert len(checked['frames']) == 5
    assert checked['rejected_by_reason']['ambiguous_aligned_depth_pair'] == 1
    assert all(f['rgb_stamp_ns'] == f['depth_stamp_ns'] for f in checked['frames'])

    # The accepted initial pin must obey the same rule; it has no bypass.
    mutate_seed = r'''import sqlite3, sys
from rclpy.serialization import deserialize_message, serialize_message
from piper_mobile_manipulation.msg import ReconstructionKeyframe
c=sqlite3.connect(sys.argv[1])
tid=c.execute("SELECT id FROM topics WHERE name='/piper/reconstruction_keyframe'").fetchone()[0]
mid,blob=c.execute('SELECT id,data FROM messages WHERE topic_id=? LIMIT 1',(tid,)).fetchone()
m=deserialize_message(blob,ReconstructionKeyframe);m.depth.header.stamp.nanosec+=11000000
c.execute('UPDATE messages SET data=? WHERE id=?',(serialize_message(m),mid));c.commit()
'''
    subprocess.run([sys.executable, '-c', mutate_seed,
                    str(next((tmp_path / 'bag').glob('*.db3')))], check=True)
    result = subprocess.run([sys.executable, str(root / 'tools/bundlesdf/extract_video.py'),
                             str(tmp_path / 'bag'), str(tmp_path / 'bad-seed')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Initial raw pin has ambiguous aligned-depth frameset' in result.stderr
