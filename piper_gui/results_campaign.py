"""Thin GUI-facing controller for passive results campaigns.

No ROS interfaces are used here.  The controller manages generated campaign
files and command-free collector/report subprocesses only.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Optional

from results_campaign.campaign import CampaignStore
from results_campaign.collector import collect_campaign, collect_task


class ResultsCampaignController:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.store: Optional[CampaignStore] = None
        self.recorder_process = None
        self.report_process = None

    def open(self, campaign_id):
        store = CampaignStore(self.project_root, campaign_id)
        store.create_or_load()
        self.store = store
        self.start_recorder()
        return store.progress()

    def start_recorder(self):
        if self.store is None:
            raise ValueError('open a campaign first')
        if self.recorder_process is not None and self.recorder_process.poll() is None:
            return self.recorder_process
        self.recorder_process = subprocess.Popen(
            [sys.executable, '-m', 'results_campaign.recorder',
             '--project-root', str(self.project_root),
             '--campaign', self.store.campaign_id],
            cwd=str(self.project_root), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return self.recorder_process

    def next_trial(self):
        if self.store is None:
            raise ValueError('open a campaign first')
        return self.store.load_next_trial()

    def record_submission(self, *args, **kwargs):
        if self.store is None:
            return None
        return self.store.record_submission(*args, **kwargs)

    def record_terminal(self, task_id, result):
        if self.store is None:
            return None
        self.store.record_terminal(task_id, result)
        attempt = self.store.attempt_for_task(task_id)
        return collect_task(self.project_root, attempt) if attempt else None

    def collect_now(self):
        if self.store is None:
            raise ValueError('open a campaign first')
        return collect_campaign(self.project_root, self.store.root)

    def build_report(self, run_reconstruction=False):
        if self.store is None:
            raise ValueError('open a campaign first')
        if self.report_process is not None and self.report_process.poll() is None:
            raise ValueError('a campaign report is already being generated')
        command = [sys.executable, '-m', 'results_campaign.report',
                   '--project-root', str(self.project_root),
                   '--campaign', self.store.campaign_id]
        if run_reconstruction:
            command.append('--run-reconstruction')
        self.report_process = subprocess.Popen(
            command, cwd=str(self.project_root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        return self.report_process

    def shutdown(self):
        for process in (self.recorder_process, self.report_process):
            if process is not None and process.poll() is None:
                process.terminate()
