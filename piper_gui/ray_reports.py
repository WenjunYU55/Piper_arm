"""Command-free GUI helpers for mission ray review and historical replay.

The operator GUI deliberately owns only one small child process.  The child
has no ROS dependencies and receives validated report paths as newline JSON on
stdin.  Keeping process ownership here makes opening another report cheap and
ensures closing the operator GUI cannot affect any mission process.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import webbrowser

from piper_mobile_manipulation.ray_mission_diagnostics import (
    ARTIFACT_BASENAME,
    replay_historical_dataset,
)
from reconstruction.gui_support import validated_dataset_path


def ray_report_root(project_root):
    """Return the canonical per-mission diagnostics directory."""
    return (
        Path(project_root).resolve()
        / 'datasets' / 'active_scan' / 'ray_diagnostics'
    )


def list_ray_reports(project_root):
    """List live mission reports before command-free historical replays.

    JSON is canonical.  The HTML sibling remains a compatibility export and
    may be absent while a live writer is between its two atomic replacements.
    Each group is newest first.  A completed replay is selected explicitly by
    its worker, so prioritising live missions here prevents a later replay from
    looking like the latest full ray lifecycle after a GUI restart.
    """
    root = ray_report_root(project_root)
    if not root.is_dir():
        return []
    return sorted(
        root.glob('*/%s.json' % ARTIFACT_BASENAME),
        key=lambda path: (
            path.parent.name.startswith('replay_'),
            -path.stat().st_mtime_ns,
            path.parent.name,
        ),
    )


def validated_ray_report(project_root, selection, suffix='.json'):
    """Resolve a GUI report selection without permitting path traversal."""
    root = ray_report_root(project_root)
    if suffix not in ('.json', '.html'):
        raise ValueError('unsupported ray report representation')
    selected = root / str(selection) / (ARTIFACT_BASENAME + suffix)
    selected = selected.resolve()
    try:
        selected.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError('ray report selection escapes the report root') from exc
    if not selected.is_file():
        raise ValueError('selected ray report is missing')
    return selected


def open_ray_report(project_root, selection, opener=None):
    """Open the legacy self-contained HTML compatibility report."""
    report = validated_ray_report(project_root, selection, suffix='.html')
    open_url = opener or webbrowser.open_new_tab
    if open_url(report.as_uri()) is False:
        raise OSError('the desktop browser did not accept the ray report')
    return report


def replay_scan_dataset(project_root, selection):
    """Build a report from immutable historical capture metadata only."""
    dataset = validated_dataset_path(project_root, selection)
    return replay_historical_dataset(dataset, ray_report_root(project_root))


def ray_review_command(project_root):
    """Return the ROS-free viewer command used by the managed child."""
    return [
        sys.executable, '-m', 'piper_gui.ray_review_viewer',
        '--project-root', str(Path(project_root).resolve()),
    ]


class RayReviewProcess:
    """Own and reuse the one Ray Review child process for a GUI instance."""

    def __init__(self, project_root, process_factory=None):
        self.project_root = Path(project_root).resolve()
        self.process_factory = process_factory or subprocess.Popen
        self.process = None

    def _start(self):
        environment = os.environ.copy()
        # The module lives in the repository and must remain importable even
        # when the GUI was launched from an installed ROS entry point.
        python_path = environment.get('PYTHONPATH', '')
        entries = [str(self.project_root)]
        if python_path:
            entries.append(python_path)
        environment['PYTHONPATH'] = os.pathsep.join(entries)
        self.process = self.process_factory(
            ray_review_command(self.project_root),
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.project_root),
            env=environment,
        )

    def open(self, selection):
        report = validated_ray_report(self.project_root, selection)
        if self.process is None or self.process.poll() is not None:
            self._start()
        message = json.dumps({
            'command': 'open',
            'report': str(report),
        }, sort_keys=True) + '\n'
        try:
            self.process.stdin.write(message)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            # A desktop process can disappear between poll and write.  Retry
            # once with a new child; never spawn more than one live viewer.
            self._start()
            self.process.stdin.write(message)
            self.process.stdin.flush()
        return report

    def shutdown(self):
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.write('{"command":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=2.0)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired, ValueError):
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
