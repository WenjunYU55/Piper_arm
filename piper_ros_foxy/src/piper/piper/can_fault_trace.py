"""Bounded passive CAN evidence; disk writes never run on control callbacks."""

from collections import deque
import json
import os
from pathlib import Path
import threading
import time
import uuid


TRACE_CAN_IDS = (0x2A5, 0x2A6, 0x2A7, 0x472, 0x473, 0x474,
                 0x150, 0x151, 0x152, 0x153, 0x154, 0x155, 0x156, 0x157, 0x159,
                 0x470, 0x471, 0x2A1, 0x261, 0x262, 0x263, 0x264, 0x265, 0x266)


class CanFaultTrace:
    """Retain bounded pre/post-fault traffic and persist one fault per process."""

    def __init__(self, directory, warning, capacity=12000, post_seconds=1.0):
        self.directory = Path(directory)
        self.warning = warning
        self.frames = deque(maxlen=capacity)
        self.capacity = capacity
        self.post_seconds = post_seconds
        self.lock = threading.Lock()
        self.fault = None
        self.before = None
        self.written = False
        self.writer = None

    def observe(self, message, now=None):
        """Copy actual received IDs, bytes, flags and kernel timestamps."""
        now = time.monotonic() if now is None else now
        record = {
            'monotonic': now, 'timestamp': message.timestamp,
            'can_id': message.arbitration_id, 'data': bytes(message.data).hex(),
            'dlc': message.dlc, 'is_rx': message.is_rx,
            'extended': message.is_extended_id, 'error': message.is_error_frame,
            'remote': message.is_remote_frame, 'fd': message.is_fd,
        }
        with self.lock:
            if not self.written:
                self.frames.append(record)

    def trigger(self, reason, now=None):
        """Freeze pre-fault evidence once; repeated faults cannot overwrite it."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.fault is None:
                self.fault = {'reason': str(reason), 'monotonic': now,
                              'wall_time': time.time()}
                self.before = list(self.frames)
                self.frames.clear()

    def flush(self, force=False, now=None):
        """Dispatch a bounded snapshot to a disk-only background writer."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.fault is None or self.written:
                return
            if not force and now - self.fault['monotonic'] < self.post_seconds:
                return
            self.written = True
            payload = {'schema_version': 1, 'fault': self.fault,
                       'pre_fault': self.before, 'post_fault': list(self.frames),
                       'capacity_per_window': self.capacity,
                       'limitation': 'SocketCAN observation, not independent bus '
                       'metrology; local TX does not prove controller receipt.'}
            self.frames.clear()
            self.before = None
        self.writer = threading.Thread(target=self._write, args=(payload,), daemon=True)
        self.writer.start()

    def _write(self, payload):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / ('fault-' + uuid.uuid4().hex + '.json')
            temporary = path.with_suffix('.tmp')
            with temporary.open('x') as handle:
                json.dump(payload, handle, separators=(',', ':'))
            os.replace(str(temporary), str(path))
            records = sorted(self.directory.glob('fault-*.json'),
                             key=lambda entry: entry.stat().st_mtime)
            for old in records[:-5]:
                old.unlink()
        except OSError as exc:
            self.warning('CAN fault evidence write failed: %s' % exc)
