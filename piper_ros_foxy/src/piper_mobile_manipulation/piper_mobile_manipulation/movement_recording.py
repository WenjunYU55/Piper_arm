"""Movement windows and bounded pre-roll for command-free sensor recording."""

from collections import deque


class MovementWindow:
    """Merge movement and one-second shoulders using a single time domain."""

    def __init__(self, shoulder=1.0):
        self.shoulder = float(shoulder)
        self.moving = False
        self.until = -float('inf')

    def update(self, state, now):
        moving = state == 'MOVING'
        started = moving and not self.moving
        if self.moving and not moving:
            self.until = now + self.shoulder
        self.moving = moving
        return started

    def active(self, now):
        return self.moving or now <= self.until


class PreRoll:
    """Bound both age and bytes; a restart cannot replay already saved frames."""

    def __init__(self, seconds=1.0, maximum_bytes=128 * 1024**2):
        self.seconds = seconds
        self.maximum_bytes = maximum_bytes
        self.items = deque()
        self.bytes = 0
        self.overflow = 0

    def add(self, now, item):
        self.items.append((now, item))
        self.bytes += len(item[1])
        self.expire(now)
        while self.bytes > self.maximum_bytes and self.items:
            _, old = self.items.popleft()
            self.bytes -= len(old[1])
            self.overflow += 1

    def expire(self, now):
        while self.items and self.items[0][0] < now - self.seconds:
            _, item = self.items.popleft()
            self.bytes -= len(item[1])

    def take(self, now):
        self.expire(now)
        result = [item for _, item in self.items]
        self.items.clear()
        self.bytes = 0
        return result


def movement_intervals(events, end_ns, shoulder_ns=1_000_000_000):
    """Derive merged recorded-time intervals; unfinished movement ends at bag end."""
    intervals = []
    start = None
    for stamp, state in sorted(events):
        if state == 'MOVING' and start is None:
            start = stamp - shoulder_ns
        elif state != 'MOVING' and start is not None:
            intervals.append((start, stamp + shoulder_ns))
            start = None
    if start is not None:
        intervals.append((start, end_ns))
    merged = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


class SQLiteBagWriter:
    """Foxy SQLite/CDR bag storage without unavailable rosbag2 Python bindings."""

    def __init__(self, output):
        import sqlite3
        from pathlib import Path
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.db = sqlite3.connect(str(self.root / 'bag_0.db3'))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL,'
                        'type TEXT NOT NULL, serialization_format TEXT NOT NULL,'
                        'offered_qos_profiles TEXT NOT NULL)')
        self.db.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,'
                        'timestamp INTEGER NOT NULL, data BLOB NOT NULL)')
        self.db.execute('CREATE INDEX timestamp_idx ON messages (timestamp ASC)')
        self.topics = {}
        self.last_commit = 0.

    def create_topic(self, name, kind):
        cursor = self.db.execute('INSERT INTO topics(name,type,serialization_format,'
                                 'offered_qos_profiles) VALUES (?,?,?,?)',
                                 (name, kind, 'cdr', ''))
        self.topics[name] = cursor.lastrowid

    def write(self, topic, data, stamp):
        import time
        self.db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (?,?,?)',
                        (self.topics[topic], stamp, data))
        now = time.monotonic()
        if now - self.last_commit >= .2:
            self.db.commit()
            self.last_commit = now

    def close(self):
        import yaml
        self.db.commit()
        count, first, last = self.db.execute(
            'SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages').fetchone()
        topics = []
        for key, name, kind, serialization, qos in self.db.execute('SELECT * FROM topics'):
            n = self.db.execute('SELECT COUNT(*) FROM messages WHERE topic_id=?',
                                (key,)).fetchone()[0]
            topics.append({'topic_metadata': dict(name=name, type=kind,
                          serialization_format=serialization, offered_qos_profiles=qos),
                           'message_count': n})
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
        info = dict(version=4, storage_identifier='sqlite3', relative_file_paths=['bag_0.db3'],
                    duration={'nanoseconds': (last - first) if count else 0},
                    starting_time={'nanoseconds_since_epoch': first or 0},
                    message_count=count, topics_with_message_count=topics,
                    compression_format='', compression_mode='')
        temporary = self.root / 'metadata.yaml.tmp'
        temporary.write_text(yaml.safe_dump({'rosbag2_bagfile_information': info}))
        temporary.replace(self.root / 'metadata.yaml')
