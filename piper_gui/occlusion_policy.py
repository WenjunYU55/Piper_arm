"""Persist the next mission's occlusion probe choice without runtime commands."""

from pathlib import Path

from piper_gui.scan_policy import _atomic_write
from piper_gui.stream_rate import _match


OCCLUSION_POLICY_LABELS = {
    False: 'At the start only',
    True: 'Every viewpoint',
}
KEY = 'occlusion_probe_every_viewpoint'


def default_occlusion_policy_path(project_root):
    return Path(project_root) / (
        'piper_ros_foxy/src/piper_mobile_manipulation/'
        'config/supervised_cube_workflow_params.yaml')


def read_occlusion_policy(path):
    value = _match(Path(path).read_text(encoding='utf-8'), KEY).group('value')
    if value not in ('true', 'false'):
        raise ValueError('occlusion probe choice must be true or false')
    return value == 'true'


def write_occlusion_policy(path, every_viewpoint):
    if type(every_viewpoint) is not bool:
        raise ValueError('occlusion probe choice must be a boolean')
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    match = _match(text, KEY)
    updated = text[:match.start('value')] + (
        'true' if every_viewpoint else 'false') + text[match.end('value'):]
    if updated != text:
        _atomic_write(path, updated)
    return every_viewpoint
