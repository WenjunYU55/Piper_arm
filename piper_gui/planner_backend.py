"""Persist and validate the planner selected for the next mission."""

import os
from pathlib import Path
import re
import tempfile


TESSERACT = 'tesseract'
CUROBO = 'curobo'
SUPPORTED_BACKENDS = (TESSERACT, CUROBO)
BACKEND_LABELS = {
    TESSERACT: 'Tesseract',
    CUROBO: 'cuRobo',
}
_BACKEND_LINE = re.compile(
    r'(?m)^(?P<prefix>\s*planner_backend\s*:\s*["\']?)'
    r'(?P<value>[^"\'\s#]+)(?P<suffix>["\']?\s*(?:#.*)?)$')


def default_planner_backend_path(project_root):
    """Return the GUI-owned next-mission planner setting."""
    return Path(project_root) / (
        'piper_ros_foxy/src/piper_mobile_manipulation/'
        'config/planner_backend.yaml')


def validate_planner_backend(value):
    """Normalize one persisted/selected backend or fail closed."""
    backend = str(value).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError('unsupported planner backend: %s' % value)
    return backend


def read_planner_backend(path):
    """Read the one authoritative next-mission planner value."""
    text = Path(path).read_text(encoding='utf-8')
    matches = list(_BACKEND_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            'planner configuration must contain exactly one planner_backend')
    return validate_planner_backend(matches[0].group('value'))


def write_planner_backend(path, value):
    """Atomically persist the planner used by the next mission goal."""
    backend = validate_planner_backend(value)
    target = Path(path)
    text = target.read_text(encoding='utf-8')
    matches = list(_BACKEND_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            'planner configuration must contain exactly one planner_backend')
    match = matches[0]
    updated = (
        text[:match.start()] + match.group('prefix') + backend
        + match.group('suffix') + text[match.end():])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=str(target.parent),
                prefix='.%s.' % target.name, suffix='.tmp',
                delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), target.stat().st_mode)
        os.replace(str(temporary), str(target))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return backend
