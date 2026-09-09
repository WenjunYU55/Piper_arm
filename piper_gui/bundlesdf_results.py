"""Command-free BundleSDF result selection and viewer controls for the Tk GUI."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import signal
import time
import uuid
import tkinter as tk
from tkinter import filedialog, ttk

import yaml


def job_progress(job_dir, output, stage):
    """Read bounded progress evidence; do not count pose files as successful tracks."""
    labels = dict(extract_rgbd_video='Extracting RGB-D recording',
                  track_target_masks='Tracking target masks', validate_input='Validating dataset',
                  reconstruct='Reconstructing mesh')
    label = labels.get(stage, stage)
    if stage == 'track_target_masks':
        path = Path(job_dir) / 'track_target_masks.log'
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-4096))
                tail = stream.read(4096).decode('utf-8', errors='replace')
            matches = re.findall(r'SAM2 (\d+)/(\d+) frames, (\d+) accepted, (\d+) rejected', tail)
            if matches:
                current, total, accepted, rejected = matches[-1]
                label += ': %s/%s frames (%s usable, %s rejected)' % (current, total, accepted, rejected)
    elif stage == 'reconstruct':
        poses = Path(output) / 'result/ob_in_cam'
        if poses.is_dir():
            count = sum(1 for path in poses.iterdir() if path.suffix == '.txt')
            label += ': %d pose files written; tracking quality checked at completion' % count
    return label


def input_details(selection):
    """Describe a selectable raw input without decoding large images on Tk."""
    root = Path(selection).expanduser().resolve()
    if root.name == 'bag' and (root.parent / 'recording.json').is_file():
        root = root.parent
    if (root / 'recording.json').is_file():
        data = json.loads((root / 'recording.json').read_text())
        if data.get('state') not in ('closed', 'incomplete'):
            raise ValueError('Recording is still active; select a closed recording')
        counts = data.get('topic_counts', {})
        return root, 'Recorded RGB-D: %d RGB images, %d saved masks; target tracking runs offline' % (
            counts.get('/camera/color/image_raw', 0), counts.get('/piper/sam2_target_mask', 0))
    if (root / 'manifest.json').is_file():
        data = json.loads((root / 'manifest.json').read_text())
        if data.get('dataset_type') != 'piper_raw_rgbd_reconstruction':
            raise ValueError('This is a stationary-capture dataset. Select its whole-scan '
                             'recording or a synchronized raw reconstruction session.')
        if data.get('state') == 'recording':
            raise ValueError('Session is still recording')
        return root, '%d usable raw frames; %s session' % (
            data.get('accepted', 0), 'complete' if data.get('complete') else 'partial')
    raise ValueError('Select a recorded RGB-D dataset to reconstruct, or a result to view')


def reconstruction_command(project, selection, allow_partial, stride, validate_only=False):
    """Build literal arguments and unique output paths; never start robot interfaces."""
    source, _ = input_details(selection)
    stride = int(stride)
    if stride < 1:
        raise ValueError('Frame stride must be positive')
    runtime = Path(os.environ.get(
        'PIPER_BUNDLESDF_RUNTIME', str(Path.home() / '.local/share/piper-bundlesdf')))
    name = time.strftime('%Y-%m-%d-scan-') + uuid.uuid4().hex[:12]
    output, job = runtime / 'evidence' / name, runtime / 'jobs' / name
    command = ['/usr/bin/python3', str(Path(project) / 'tools/bundlesdf/reconstruct_job.py'),
               str(source), '--output', str(output), '--job-dir', str(job),
               '--frame-stride', str(stride)]
    if allow_partial:
        command.append('--allow-incomplete')
    if validate_only:
        command.append('--validate-only')
    return command, output, job


def result_details(selection):
    """Resolve an exported run without treating unknown provenance as live evidence."""
    root = Path(selection).expanduser().resolve()
    if root.name == 'textured_mesh.obj':
        root = root.parent
    if root.name in ('result', 'base_link'):
        root = root.parent
    mesh = root / 'result/textured_mesh.obj'
    if not mesh.is_file() or not (root / 'run.json').is_file():
        raise ValueError(
            'Choose a BundleSDF output folder containing run.json '
            'and result/textured_mesh.obj.')
    oriented = root / 'base_link/textured_mesh.obj'
    if oriented.is_file() and oriented.with_name(oriented.name + '.frame.json').is_file():
        mesh = oriented
    evidence = {}
    inspection = root / 'mesh-inspection.json'
    if inspection.is_file():
        evidence = json.loads(inspection.read_text())
        if not isinstance(evidence, dict):
            raise ValueError('Invalid reconstruction evidence report')
    synthetic = 'SYNTHETIC' in str(evidence.get('evidence_class', '')).upper()
    label = ('Synthetic — not a live recording' if synthetic
             else 'Saved reconstruction — recording origin not verified')
    recovery_path = root / 'recovery.json'
    if not synthetic and recovery_path.is_file():
        recovery = json.loads(recovery_path.read_text())
        if (isinstance(recovery, dict) and recovery.get('source_bag')
                and recovery.get('evidence_class')
                in ('RECORDED_ROS_BAG_CALLBACK_RECOVERY_NOT_DDS_OR_NEW_PHYSICAL_TEST',
                    'RECORDED_RGBD_DENSE_OFFLINE_SAM2')):
            label = 'Recorded ROS RGB-D session — offline reconstruction'
    return root, mesh, label


def selection_details(selection, targets=None):
    """Cheap metadata eligibility, not a promise of successful reconstruction."""
    root = Path(selection).expanduser().resolve()
    targets = targets or {}
    try:
        if not selection:
            raise ValueError('Select a recording or saved mesh')
        if root.name == 'bag' and (root.parent / 'recording.json').is_file():
            root = root.parent
        if (root / 'recording.json').is_file() or (root / 'manifest.json').is_file():
            root, detail = input_details(root)
            archive = (root / 'recording.json').is_file()
            data = json.loads((root / ('recording.json' if archive else 'manifest.json')).read_text())
            target = targets.get(data.get('task_id'), data.get('target_label', 'Target unknown'))
            when = ''
            if archive and data.get('started_unix_ns'):
                when = datetime.fromtimestamp(
                    int(data['started_unix_ns']) / 1e9).strftime('%d %b %Y %H:%M')
            counts = data.get('topic_counts', {})
            count = counts.get('/camera/color/image_raw', 0) if archive else data.get('accepted', 0)
            reason = ''
            if archive and counts.get('/piper/reconstruction_keyframe') == 0:
                reason = 'No accepted reconstruction keyframe; this pipeline cannot reconstruct it'
            elif not count:
                reason = 'No usable frames recorded'
            label = 'Cannot reconstruct' if reason else 'Reconstruct input'
            title = ' — '.join(filter(None, [label, target, when, '{:,} {} frames'.format(
                count, 'RGB' if archive else 'usable'),
                root.name.replace('scan_recording_', '').split('.')[0][:8]]))
            detail = reason or (detail + '. Eligible input; validation still required.')
            return title, detail, not reason, False
        root, _, origin = result_details(root)
        name = root.name.replace('_', ' ').replace('-', ' ')
        context = []
        try:
            recovery = json.loads((root / 'recovery.json').read_text())
            source = Path(recovery['source_bag']).parent / 'recording.json'
            recording = json.loads(source.read_text())
            target = targets.get(recording.get('task_id'))
            if target:
                context.append(target)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        try:
            validation = json.loads((root / 'validation.json').read_text())
            if validation.get('converted_frames') is not None:
                context.append(str(validation['converted_frames']) + ' input frames')
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        if context:
            name = ' — '.join(context + [name])
        title = ('Synthetic mesh' if origin.startswith('Synthetic') else 'Saved mesh')
        return (title + ' — View only — ' + name,
                origin + '. Open this mesh to inspect it; '
                'select its source recording to reconstruct again.',
                False, True)
    except (OSError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        if (root / 'run.json').is_file() and not (root / 'result/textured_mesh.obj').is_file():
            detail = ('No exported mesh. Select the source recording to retry; '
                      'retain this run for diagnostics.')
        else:
            detail = str(exc)
        return 'Unavailable — ' + root.name, detail, False, False


def viewer_command(project_root, selection):
    """Keep mesh loading and its graphics dependencies outside ROS Python."""
    root, mesh, label = result_details(selection)
    project = Path(project_root).resolve()
    launcher = project / 'tools/bundlesdf/python_native.sh'
    python = Path(os.environ.get(
        'PIPER_BUNDLESDF_PYTHON', str(Path.home() / '.venvs/piper-bundlesdf/bin/python')))
    if not python.is_file():
        raise ValueError(
            'BundleSDF viewer environment missing. Run tools/bundlesdf/setup_native.sh.')
    return [str(launcher), str(project / 'reconstruction/view_bundlesdf.py'),
            str(mesh), '--title', 'BundleSDF: ' + label]


class BundleSDFResults(ttk.LabelFrame):
    """Browse saved results and launch a responsive, isolated textured 3D viewer."""

    def __init__(self, parent, project_root, can_reconstruct=lambda: True):
        super().__init__(parent, text='BundleSDF results', padding=10)
        self.project_root = project_root
        self.can_reconstruct = can_reconstruct
        self.job_process = None
        self.job_poll_id = None
        self.job_cancel_time = None
        self.allow_partial = tk.BooleanVar(self, True)
        self.stride = tk.StringVar(self, '1')
        self.selection = tk.StringVar(self)
        self.display_selection = tk.StringVar(self)
        self.choices = {}
        self.targets = {}
        self.status = tk.StringVar(self, 'Select a recorded RGB-D dataset or BundleSDF result.')
        self.process = None
        self.log = None
        self.poll_id = None
        self.columnconfigure(0, weight=1)
        self.combo = ttk.Combobox(self, textvariable=self.display_selection, width=70, state='readonly')
        self.combo.grid(row=0, column=0, sticky='ew')
        self.combo.bind('<<ComboboxSelected>>', self.choose_selection)
        self.combo.bind('<Return>', self.show_selection)
        ttk.Button(self, text='Browse…', command=self.browse).grid(row=0, column=1, padx=6)
        ttk.Button(self, text='Refresh', command=self.refresh).grid(row=0, column=2)
        self.open_button = ttk.Button(self, text='Open BundleSDF Model', command=self.open_model)
        self.open_button.grid(row=1, column=0, sticky='w', pady=(8, 0))
        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, columnspan=3, sticky='w', pady=(8, 0))
        self.build_button = ttk.Button(actions, text='Reconstruct', command=self.reconstruct)
        self.build_button.pack(side='left')
        self.validate_button = ttk.Button(
            actions, text='Validate dataset', command=lambda: self.reconstruct(True))
        self.validate_button.pack(side='left', padx=6)
        self.cancel_button = ttk.Button(
            actions, text='Cancel reconstruction', command=self.cancel_job, state='disabled')
        self.cancel_button.pack(side='left')
        ttk.Checkbutton(actions, text='Allow partial scan', variable=self.allow_partial).pack(
            side='left', padx=6)
        ttk.Label(actions, text='Frame stride').pack(side='left')
        ttk.Spinbox(actions, from_=1, to=30, width=4, textvariable=self.stride).pack(side='left')
        ttk.Label(self, textvariable=self.status, wraplength=780).grid(
            row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))
        ttk.Label(self, textvariable=self.selection, wraplength=780).grid(
            row=4, column=0, columnspan=3, sticky='w', pady=(4, 0))
        self.bind('<Destroy>', self.on_destroy, add='+')
        self.refresh()

    def refresh(self):
        """Discover retained local outputs; arbitrary run locations remain browsable."""
        runtime = Path(os.environ.get(
            'PIPER_BUNDLESDF_RUNTIME', str(Path.home() / '.local/share/piper-bundlesdf')))
        candidates = list((runtime / 'evidence').glob('*/run.json'))
        project = Path(self.project_root)
        candidates += list((project / 'datasets/mission_recordings').glob('*.closed/recording.json'))
        candidates += list((project / 'datasets/mission_recordings').glob('*.incomplete/recording.json'))
        for state in ('complete', 'incomplete', 'failed', 'cancelled'):
            candidates += list((project / 'datasets/reconstruction_sessions').glob(
                '*.' + state + '/manifest.json'))
        self.targets = {}
        for path in (project / 'datasets/active_scan').glob('*/metadata.yaml'):
            try:
                data = yaml.safe_load(path.read_text())
                if data.get('task_id') and data.get('target_label'):
                    self.targets[data['task_id']] = data['target_label']
            except (OSError, ValueError, AttributeError, yaml.YAMLError):
                continue
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self.choices = {}
        for path in candidates:
            label = selection_details(str(path.parent), self.targets)[0]
            # A numeric suffix disambiguates equal display names; paths remain literal.
            unique = label
            number = 2
            while unique in self.choices:
                unique = label + ' (' + str(number) + ')'
                number += 1
            self.choices[unique] = str(path.parent)
        self.combo.configure(values=list(self.choices))
        if not self.selection.get() and candidates:
            self.selection.set(str(candidates[0].parent))
        self.show_selection()

    def browse(self):
        """Select an existing output, without starting reconstruction or recording."""
        selected = filedialog.askdirectory(parent=self, title='Choose RGB-D dataset or BundleSDF result')
        if selected:
            self.selection.set(selected)
            self.show_selection()

    def choose_selection(self, _event=None):
        self.selection.set(self.choices[self.display_selection.get()])
        self.show_selection()

    def show_selection(self, _event=None):
        """Show readable identity and enable only applicable actions."""
        title, detail, reconstructable, viewable = selection_details(self.selection.get(), self.targets)
        self.display_selection.set(next((label for label, path in self.choices.items()
                                        if path == self.selection.get()), title))
        self.status.set(detail)
        self.build_button.configure(state='normal' if reconstructable and self.job_process is None
                                    else 'disabled')
        self.validate_button.configure(state='normal' if reconstructable and self.job_process is None
                                       else 'disabled')
        self.open_button.configure(state='normal' if viewable and self.process is None else 'disabled')

    def reconstruct(self, validate_only=False):
        """Launch the offline job separately and keep Tk responsive."""
        if self.job_process is not None:
            return
        try:
            if not self.can_reconstruct():
                raise ValueError('Finish the active mission before starting GPU reconstruction')
            command, self.job_output, self.job_dir = reconstruction_command(
                self.project_root, self.selection.get(), self.allow_partial.get(),
                self.stride.get(), validate_only)
            self.job_validate_only = validate_only
            self.job_log = tempfile.TemporaryFile(mode='w+')
            self.job_process = subprocess.Popen(
                command, stdout=self.job_log, stderr=subprocess.STDOUT, start_new_session=True)
            self.job_cancel_time = None
            self.build_button.configure(state='disabled')
            self.validate_button.configure(state='disabled')
            self.cancel_button.configure(state='normal')
            self.status.set('Starting offline job. Logs: ' + str(self.job_dir))
            self.job_poll_id = self.after(500, self.poll_job)
        except (OSError, ValueError, TypeError) as exc:
            if getattr(self, 'job_log', None) is not None:
                self.job_log.close()
            self.status.set(str(exc))

    def cancel_job(self):
        """Signal only the process group created for this offline job."""
        if self.job_process is not None and self.job_cancel_time is None:
            self.job_cancel_time = time.monotonic()
            try:
                os.killpg(self.job_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.status.set('Cancelling offline reconstruction; input recording is preserved')

    def poll_job(self):
        """Display atomic stage progress and select the resulting mesh on success."""
        self.job_poll_id = None
        if self.job_cancel_time and time.monotonic() - self.job_cancel_time > 10:
            try:
                os.killpg(self.job_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        code = self.job_process.poll()
        if code is None:
            if not self.job_cancel_time:
                try:
                    state = json.loads((self.job_dir / 'job.json').read_text())
                    self.status.set(job_progress(
                        self.job_dir, self.job_output, state.get('stage', 'Starting'))
                                    + ' — logs: ' + str(self.job_dir))
                except (OSError, ValueError):
                    pass
            self.job_poll_id = self.after(500, self.poll_job)
            return
        self.job_log.seek(0)
        diagnostic = self.job_log.read()[-1600:]
        self.job_log.close()
        self.job_process = None
        self.show_selection()
        self.cancel_button.configure(state='disabled')
        if self.job_cancel_time:
            self.status.set('Reconstruction cancelled; retained logs: ' + str(self.job_dir))
        elif code:
            self.status.set('Reconstruction failed: ' + diagnostic + ' Logs: ' + str(self.job_dir))
        elif self.job_validate_only:
            self.status.set('Dataset validation passed. Logs: ' + str(self.job_dir))
        else:
            self.selection.set(str(self.job_output))
            self.refresh()
            warning = ''
            try:
                report = json.loads((self.job_output / 'result_validation.json').read_text())
                failures = report.get('tracking_health', {}).get('explicit_failed_frame_ids', [])
                if failures:
                    warning = ' WARNING: %d frames reported tracking failure.' % len(failures)
            except (OSError, ValueError):
                warning = ' Tracking quality report unavailable.'
            self.status.set('Mesh exported; accuracy is unverified.' + warning
                            + ' Open BundleSDF Model to inspect. Logs: ' + str(self.job_dir))

    def open_model(self):
        """Start only an offline viewer; report child errors back on the Tk thread."""
        if self.process is not None:
            return
        try:
            command = viewer_command(self.project_root, self.selection.get())
            self.log = tempfile.TemporaryFile(mode='w+')
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            self.open_button.configure(state='disabled')
            self.show_selection()
            self.status.set(self.status.get() + ' — opening interactive viewer…')
            self.poll_id = self.after(250, self.poll_viewer)
        except (OSError, ValueError, TypeError) as exc:
            if self.log is not None:
                self.log.close()
                self.log = None
            self.status.set(str(exc))

    def poll_viewer(self):
        """Keep image loading and graphics work off the GUI event loop."""
        self.poll_id = None
        code = self.process.poll()
        if code is None:
            self.poll_id = self.after(250, self.poll_viewer)
            return
        self.log.seek(0)
        message = self.log.read()[-1600:]
        self.log.close()
        self.log = None
        self.process = None
        self.show_selection()
        if code:
            self.status.set('Viewer failed: ' + message)
        else:
            self.show_selection()

    def on_destroy(self, event):
        """Release polling and log handles; the independent viewer can stay open."""
        if event.widget is not self:
            return
        if self.poll_id is not None:
            self.after_cancel(self.poll_id)
        if self.job_poll_id is not None:
            self.after_cancel(self.job_poll_id)
        if getattr(self, 'job_log', None) is not None:
            self.job_log.close()
        if self.log is not None:
            self.log.close()
