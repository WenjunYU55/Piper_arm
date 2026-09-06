"""Atomic next-mission transport settings; no ROS or robot commands."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile


STREAM_RATE_LABELS = {
    20: '20 Hz (standard)',
    100: '100 Hz (experimental)',
    200: '200 Hz (experimental)',
}


@dataclass(frozen=True)
class StreamRateSettings:
    trajectory_command_rate_hz: float
    executor_tick_rate_hz: float


def _match(text, key):
    matches = list(re.finditer(
        r'^(?P<prefix>[ \t]*' + re.escape(key) + r'[ \t]*:[ \t]*)'
        r'(?P<value>[^\s#]+)(?P<suffix>[ \t]*(?:#.*)?)$',
        text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError('configuration must contain exactly one %s' % key)
    return matches[0]


def read_stream_rate(path):
    """Read the existing authoritative execution YAML, never a GUI copy."""
    text = Path(path).read_text(encoding='utf-8')
    rate = float(_match(text, 'trajectory_command_rate_hz').group('value'))
    tick = float(_match(text, 'executor_tick_rate_hz').group('value'))
    if rate not in STREAM_RATE_LABELS:
        raise ValueError('stream rate must be 20, 100 or 200 Hz')
    if not 2 * rate <= tick < float('inf'):
        raise ValueError('executor timer must be at least twice the stream rate')
    return StreamRateSettings(rate, tick)


def write_stream_rate(path, value):
    """Save both rates together while preserving all unrelated YAML text."""
    rate = float(value)
    if rate not in STREAM_RATE_LABELS:
        raise ValueError('stream rate must be 20, 100 or 200 Hz')
    config_path = Path(path)
    text = config_path.read_text(encoding='utf-8')
    settings = StreamRateSettings(rate, max(200., 2 * rate))
    # Validate both locations before changing anything; apply in reverse order
    # so offsets stay valid. This also allows repairing invalid saved numbers.
    replacements = [(_match(text, key), number) for key, number in (
        ('trajectory_command_rate_hz', settings.trajectory_command_rate_hz),
        ('executor_tick_rate_hz', settings.executor_tick_rate_hz))]
    for match, number in sorted(replacements, key=lambda item: item[0].start(), reverse=True):
        text = text[:match.start()] + '%s%.1f%s' % (
            match.group('prefix'), number, match.group('suffix')) + text[match.end():]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=str(config_path.parent),
                prefix='.%s.' % config_path.name, suffix='.tmp', delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary_path), config_path.stat().st_mode)
        os.replace(str(temporary_path), str(config_path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return settings
