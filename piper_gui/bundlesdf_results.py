"""Command-free BundleSDF result selection and viewer controls for the Tk GUI."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, ttk


def result_details(selection):
    """Resolve an exported run without treating unknown provenance as live evidence."""
    root = Path(selection).expanduser().resolve()
    if root.name == 'textured_mesh.obj':
        root = root.parent
    if root.name == 'result':
        root = root.parent
    mesh = root / 'result/textured_mesh.obj'
    if not mesh.is_file() or not (root / 'run.json').is_file():
        raise ValueError(
            'Choose a BundleSDF output folder containing run.json '
            'and result/textured_mesh.obj.')
    evidence = {}
    inspection = root / 'mesh-inspection.json'
    if inspection.is_file():
        evidence = json.loads(inspection.read_text())
        if not isinstance(evidence, dict):
            raise ValueError('Invalid reconstruction evidence report')
    synthetic = 'SYNTHETIC' in str(evidence.get('evidence_class', '')).upper()
    label = ('Synthetic — not a live recording' if synthetic
             else 'Saved reconstruction — recording origin not verified')
    return root, mesh, label


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

    def __init__(self, parent, project_root):
        super().__init__(parent, text='BundleSDF results', padding=10)
        self.project_root = project_root
        self.selection = tk.StringVar(self)
        self.status = tk.StringVar(self, 'Select a saved BundleSDF output folder.')
        self.process = None
        self.log = None
        self.poll_id = None
        self.columnconfigure(0, weight=1)
        self.combo = ttk.Combobox(self, textvariable=self.selection, width=48)
        self.combo.grid(row=0, column=0, sticky='ew')
        self.combo.bind('<<ComboboxSelected>>', self.show_selection)
        self.combo.bind('<Return>', self.show_selection)
        ttk.Button(self, text='Browse…', command=self.browse).grid(row=0, column=1, padx=6)
        ttk.Button(self, text='Refresh', command=self.refresh).grid(row=0, column=2)
        self.open_button = ttk.Button(self, text='Open BundleSDF Model', command=self.open_model)
        self.open_button.grid(row=1, column=0, sticky='w', pady=(8, 0))
        ttk.Label(self, textvariable=self.status, wraplength=780).grid(
            row=2, column=0, columnspan=3, sticky='w', pady=(6, 0))
        self.bind('<Destroy>', self.on_destroy, add='+')
        self.refresh()

    def refresh(self):
        """Discover retained local outputs; arbitrary run locations remain browsable."""
        runtime = Path(os.environ.get(
            'PIPER_BUNDLESDF_RUNTIME', str(Path.home() / '.local/share/piper-bundlesdf')))
        candidates = sorted((runtime / 'evidence').glob('*/run.json'), reverse=True)
        self.combo.configure(values=[str(path.parent) for path in candidates])
        if not self.selection.get() and candidates:
            self.selection.set(str(candidates[0].parent))
        self.show_selection()

    def browse(self):
        """Select an existing output, without starting reconstruction or recording."""
        selected = filedialog.askdirectory(parent=self, title='Choose BundleSDF output folder')
        if selected:
            self.selection.set(selected)
            self.show_selection()

    def show_selection(self, _event=None):
        """Display evidence origin before opening the model."""
        try:
            _, _, label = result_details(self.selection.get())
            self.status.set(label)
        except (OSError, ValueError, TypeError) as exc:
            self.status.set(str(exc))

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
        self.open_button.configure(state='normal')
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
        if self.log is not None:
            self.log.close()
