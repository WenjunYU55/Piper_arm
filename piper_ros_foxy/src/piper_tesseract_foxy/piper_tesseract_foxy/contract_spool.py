"""Atomic filesystem queues for the private Tesseract transport."""

import json
import os
from pathlib import Path
import time

from piper_tesseract_foxy.contract_core import (
    ContractError,
    HEALTH_FILENAME,
    MAX_FILE_BYTES,
    MAX_HEALTH_BYTES,
    QUEUE_NAMES,
    SAFE_ID,
)
from piper_tesseract_foxy.contract_hashing import canonical_bytes


class Spool:
    """Private atomic queues shared by the Foxy bridge and isolated worker."""

    def __init__(self, root):
        self.root = Path(root)
        self._prepare_root()

    def _prepare_root(self):
        if self.root.exists() and self.root.is_symlink():
            raise ContractError('spool root must not be a symlink')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        stat = self.root.stat()
        if stat.st_uid != os.getuid():
            raise ContractError('spool root is not owned by the current user')
        for name in QUEUE_NAMES:
            path = self.root / name
            if path.exists() and path.is_symlink():
                raise ContractError(
                    'spool queue must not be a symlink: %s' % name)
            path.mkdir(mode=0o700, exist_ok=True)
            path.chmod(0o700)

    def path(self, queue, request_id):
        if queue not in QUEUE_NAMES:
            raise ContractError('unknown queue')
        if (
                not isinstance(request_id, str)
                or SAFE_ID.fullmatch(request_id) is None):
            raise ContractError('unsafe request identifier')
        return self.root / queue / (request_id + '.json')

    def write(self, queue, request_id, payload):
        destination = self.path(queue, request_id)
        if destination.exists():
            raise ContractError('spool destination already exists')
        temporary = destination.with_name(
            '.%s.%d.tmp' % (destination.name, os.getpid()))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(temporary), flags, 0o600)
        try:
            data = canonical_bytes(payload)
            if len(data) > MAX_FILE_BYTES:
                raise ContractError('spool payload exceeds size limit')
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(str(temporary), str(destination))
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        directory = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination

    def read(self, queue, request_id):
        path = self.path(queue, request_id)
        stat = path.lstat()
        if not path.is_file() or path.is_symlink() or stat.st_nlink != 1:
            raise ContractError(
                'spool entry is not a regular single-link file')
        if stat.st_size > MAX_FILE_BYTES:
            raise ContractError('spool entry exceeds size limit')
        with open(path, 'r', encoding='utf-8') as stream:
            return json.load(stream)

    def claim_next(self):
        for source in sorted((self.root / 'requests').glob('*.json')):
            request_id = source.stem
            if SAFE_ID.fullmatch(request_id) is None:
                continue
            destination = self.path('processing', request_id)
            if destination.exists():
                continue
            try:
                os.replace(str(source), str(destination))
            except FileNotFoundError:
                continue
            return request_id, self.read('processing', request_id)
        return None, None

    def pending(self, queue):
        if queue not in QUEUE_NAMES:
            raise ContractError('unknown queue')
        return sum(
            1 for path in (self.root / queue).glob('*.json')
            if SAFE_ID.fullmatch(path.stem))

    def write_health(self, payload):
        """Atomically replace the bounded worker liveness record."""
        destination = self.root / HEALTH_FILENAME
        if destination.exists() and (
                destination.is_symlink() or not destination.is_file()):
            raise ContractError(
                'worker health destination is not a regular file')
        temporary = self.root / (
            '.%s.%d.%d.tmp'
            % (HEALTH_FILENAME, os.getpid(), time.time_ns()))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(temporary), flags, 0o600)
        try:
            data = canonical_bytes(payload)
            if len(data) > MAX_HEALTH_BYTES:
                raise ContractError(
                    'worker health payload exceeds size limit')
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise OSError('worker health write made no progress')
                offset += written
            os.fsync(descriptor)
        except Exception:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        os.replace(str(temporary), str(destination))
        directory = os.open(str(self.root), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination

    def read_health(self):
        """Read the worker liveness record without following links."""
        path = self.root / HEALTH_FILENAME
        stat = path.lstat()
        if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
            raise ContractError(
                'worker health entry is not a regular single-link file')
        if stat.st_size <= 0 or stat.st_size > MAX_HEALTH_BYTES:
            raise ContractError('worker health entry has an invalid size')
        with open(path, 'r', encoding='utf-8') as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ContractError('worker health entry must be an object')
        return value
