"""Permission-bounded atomic filesystem queue for the cuRobo worker."""

import json
import os
from pathlib import Path
import tempfile


class Spool:
    """Claim requests and publish results without ROS dependencies."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ('requests', 'processing', 'responses', 'failed'):
            path = self.root / name
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(str(path), 0o700)

    def path(self, queue, request_id):
        return self.root / queue / ('%s.json' % request_id)

    def read(self, path):
        if path.is_symlink() or not path.is_file():
            raise ValueError('spool entry is not a regular file')
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('spool entry is too large')
        return json.loads(path.read_text(encoding='utf-8'))

    def claim_next(self):
        for source in sorted((self.root / 'requests').glob('*.json')):
            target = self.path('processing', source.stem)
            try:
                os.replace(str(source), str(target))
            except FileNotFoundError:
                continue
            return source.stem, self.read(target)
        return None, None

    @staticmethod
    def _atomic_json(path, value):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', dir=str(path.parent),
                    prefix='.%s.' % path.name, suffix='.tmp',
                    delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, sort_keys=True, separators=(',', ':'))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(str(temporary), 0o600)
            os.replace(str(temporary), str(path))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def write_response(self, request_id, value):
        self._atomic_json(self.path('responses', request_id), value)

    def write_health(self, value):
        self._atomic_json(self.root / 'worker_health.json', value)

    def write_diagnostics(self, value):
        self._atomic_json(self.root / 'worker_diagnostics.json', value)
