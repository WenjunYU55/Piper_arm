"""Atomic, permission-bounded task/status/result handoff for a ROS gateway."""

import json
import os
from pathlib import Path
import tempfile

from piper_mobile_manipulation.mission_core import sha256_value


class MissionSpool:
    SUBDIRECTORIES = ('goals', 'status', 'results', 'heartbeat')

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(str(self.root), 0o700)
        for name in self.SUBDIRECTORIES:
            directory = self.root / name
            directory.mkdir(mode=0o700, exist_ok=True)
            os.chmod(str(directory), 0o700)

    @staticmethod
    def safe_id(task_id):
        value = str(task_id)
        if not value or any(character not in (
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                '0123456789_.:-') for character in value):
            raise ValueError('unsafe mission spool task ID')
        return value

    def path(self, queue, task_id):
        if queue not in self.SUBDIRECTORIES:
            raise ValueError('unsupported mission spool queue')
        return self.root / queue / (self.safe_id(task_id) + '.json')

    def write(self, queue, task_id, payload):
        value = dict(payload)
        value.pop('spool_sha256', None)
        value['spool_sha256'] = sha256_value(value)
        destination = self.path(queue, task_id)
        fd, temporary = tempfile.mkstemp(
            prefix='.%s.' % self.safe_id(task_id), suffix='.tmp',
            dir=str(destination.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, sort_keys=True, separators=(',', ':'))
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return value

    def read(self, queue, task_id):
        with open(self.path(queue, task_id), 'r', encoding='utf-8') as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError('mission spool payload is not an object')
        expected = str(value.get('spool_sha256', ''))
        unsigned = dict(value)
        unsigned.pop('spool_sha256', None)
        if expected != sha256_value(unsigned):
            raise ValueError('mission spool SHA-256 mismatch')
        return value
