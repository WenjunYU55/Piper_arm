"""
Own and supervise exact child process groups for one mission generation.

The supervisor is deliberately policy-driven.  The autonomous coordinator
uses graceful SIGINT/SIGTERM shutdown without SIGKILL, while another explicit
owner may opt into the forced-kill stage.  No process is ever discovered or
adopted by name; only handles returned by this supervisor are signal targets.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ProcessSpec:
    """Immutable startup contract for one explicitly owned process group."""

    name: str
    command: Tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'name', str(self.name))
        object.__setattr__(
            self, 'command', tuple(str(part) for part in self.command))
        object.__setattr__(
            self, 'environment', MappingProxyType(dict(self.environment)))


@dataclass
class ProcessHandle:
    """One owned group leader plus its generation-scoped log resources."""

    spec: ProcessSpec
    process: Any
    process_group_id: int
    log_stream: Any
    log_path: str
    log_offset: int


@dataclass(frozen=True)
class ShutdownReport:
    """Machine-readable result of one bounded supervisor shutdown."""

    attempted: Tuple[str, ...] = ()
    graceful_stops: Tuple[str, ...] = ()
    terminated: Tuple[str, ...] = ()
    forced_kills: Tuple[str, ...] = ()
    still_running: Tuple[str, ...] = ()
    exit_status: Tuple[Tuple[str, Optional[int]], ...] = ()
    diagnostics: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Return whether every explicitly owned target stopped."""
        return not self.still_running

    def exit_status_by_name(self):
        """Return a caller-owned mapping without exposing mutable state."""
        return dict(self.exit_status)


class ProcessSupervisor:
    """Start, monitor, and stop only explicitly owned process groups."""

    def __init__(
            self, log_root, *, popen_factory=subprocess.Popen,
            group_signaler=os.killpg, clock=time.monotonic, sleeper=time.sleep,
            graceful_timeout_sec=5.0, terminate_timeout_sec=3.0,
            force_kill=False, kill_timeout_sec=2.0):
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self._popen_factory = popen_factory
        self._group_signaler = group_signaler
        self._clock = clock
        self._sleep = sleeper
        self.graceful_timeout_sec = float(graceful_timeout_sec)
        self.terminate_timeout_sec = float(terminate_timeout_sec)
        self.force_kill = bool(force_kill)
        self.kill_timeout_sec = float(kill_timeout_sec)
        self._lock = threading.RLock()
        self.entries = {}
        self.log_offsets = {}
        self.last_shutdown_report = ShutdownReport()

    @staticmethod
    def build_environment(overrides, base_environment=None):
        """Construct the inherited process environment plus exact overrides."""
        environment = dict(
            os.environ if base_environment is None else base_environment)
        environment.update(dict(overrides))
        return environment

    @staticmethod
    def _compatibility_handle(name, entry, offset=0):
        if isinstance(entry, ProcessHandle):
            return entry
        process, log_stream, log_path = entry
        return ProcessHandle(
            spec=ProcessSpec(str(name), ()),
            process=process,
            process_group_id=int(getattr(process, 'pid', 0)),
            log_stream=log_stream,
            log_path=str(log_path),
            log_offset=int(offset),
        )

    def _handle(self, name, entry):
        return self._compatibility_handle(
            name, entry, self.log_offsets.get(name, 0))

    @staticmethod
    def _poll(handle):
        return handle.process.poll()

    @staticmethod
    def _close_log(handle):
        if not handle.log_stream.closed:
            handle.log_stream.close()

    def begin_generation(self):
        """Forget only fully stopped entries before admitting a new mission."""
        with self._lock:
            handles = {
                name: self._handle(name, entry)
                for name, entry in self.entries.items()
            }
            live = sorted(
                name for name, handle in handles.items()
                if self._poll(handle) is None)
            if live:
                return live
            for handle in handles.values():
                self._close_log(handle)
            self.entries.clear()
            self.log_offsets.clear()
            return []

    def start(self, name, command=None, environment=None):
        """Start one new-session process unless that owned name is live."""
        spec = (
            name if isinstance(name, ProcessSpec) else
            ProcessSpec(str(name), tuple(command), dict(environment)))
        with self._lock:
            existing = self.entries.get(spec.name)
            if existing is not None:
                old_handle = self._handle(spec.name, existing)
                if self._poll(old_handle) is None:
                    return old_handle
                self._close_log(old_handle)
            log_path = self.log_root / (spec.name + '.log')
            log_offset = (
                log_path.stat().st_size if log_path.exists() else 0)
            log_stream = open(log_path, 'ab', buffering=0)
            try:
                process = self._popen_factory(
                    list(spec.command), stdout=log_stream,
                    stderr=subprocess.STDOUT, env=dict(spec.environment),
                    start_new_session=True)
            except Exception:
                log_stream.close()
                raise
            handle = ProcessHandle(
                spec=spec,
                process=process,
                process_group_id=int(process.pid),
                log_stream=log_stream,
                log_path=str(log_path),
                log_offset=log_offset,
            )
            self.entries[spec.name] = handle
            self.log_offsets[spec.name] = log_offset
            return handle

    def log_since_start(self, name):
        """Read at most 256 KiB emitted by this owned startup generation."""
        with self._lock:
            entry = self.entries.get(name)
            if entry is None:
                return ''
            handle = self._handle(name, entry)
            path = Path(handle.log_path)
            offset = handle.log_offset
        try:
            with open(path, 'rb') as stream:
                stream.seek(offset)
                return stream.read(256 * 1024).decode(
                    'utf-8', errors='replace')
        except OSError:
            return ''

    def failed(self):
        """Return exit statuses for owned children that are no longer live."""
        with self._lock:
            handles = tuple(
                (name, self._handle(name, entry))
                for name, entry in self.entries.items())
        failures = {}
        for name, handle in handles:
            returncode = self._poll(handle)
            if returncode is not None:
                failures[name] = returncode
        return failures

    def health(self):
        """Return the established process-health JSON-compatible shape."""
        with self._lock:
            handles = tuple(
                (name, self._handle(name, entry))
                for name, entry in self.entries.items())
        health = {}
        for name, handle in handles:
            returncode = self._poll(handle)
            health[name] = {
                'pid': int(handle.process.pid),
                'running': returncode is None,
                'returncode': returncode,
                'log': handle.log_path,
            }
        return health

    def has_live_processes(self):
        """Return whether this supervisor still owns a live group leader."""
        with self._lock:
            handles = tuple(
                self._handle(name, entry)
                for name, entry in self.entries.items())
        return any(self._poll(handle) is None for handle in handles)

    def owned_names(self):
        """Return process names in startup order for diagnostics/tests."""
        with self._lock:
            return tuple(self.entries)

    def _signal(self, handle, signum, diagnostics):
        if self._poll(handle) is not None:
            return
        try:
            self._group_signaler(handle.process_group_id, signum)
        except ProcessLookupError:
            diagnostics.append(
                '%s process group disappeared before signal %s'
                % (handle.spec.name, int(signum)))

    def _wait_for(self, handles, timeout_sec):
        deadline = self._clock() + float(timeout_sec)
        while (
                self._clock() < deadline
                and any(self._poll(handle) is None for handle in handles)):
            self._sleep(0.05)

    def shutdown(self, names=None):
        """Stop selected owned groups using the configured escalation policy."""
        with self._lock:
            selected = set(self.entries if names is None else names)
            handles = [
                self._handle(name, entry)
                for name, entry in reversed(tuple(self.entries.items()))
                if name in selected
            ]
        attempted = tuple(handle.spec.name for handle in handles)
        diagnostics = []
        initially_live = {
            handle.spec.name for handle in handles
            if self._poll(handle) is None
        }
        for handle in handles:
            self._signal(handle, signal.SIGINT, diagnostics)
        self._wait_for(handles, self.graceful_timeout_sec)
        graceful_stops = tuple(
            handle.spec.name for handle in handles
            if handle.spec.name in initially_live
            and self._poll(handle) is not None)

        term_targets = [
            handle for handle in handles if self._poll(handle) is None]
        for handle in term_targets:
            self._signal(handle, signal.SIGTERM, diagnostics)
        self._wait_for(term_targets, self.terminate_timeout_sec)
        terminated = tuple(
            handle.spec.name for handle in term_targets
            if self._poll(handle) is not None)

        kill_targets = [
            handle for handle in term_targets if self._poll(handle) is None]
        forced_kills = ()
        if self.force_kill:
            for handle in kill_targets:
                self._signal(handle, signal.SIGKILL, diagnostics)
            self._wait_for(kill_targets, self.kill_timeout_sec)
            forced_kills = tuple(
                handle.spec.name for handle in kill_targets
                if self._poll(handle) is not None)
        elif kill_targets:
            diagnostics.extend(
                '%s remained alive after SIGTERM; SIGKILL is disabled'
                % handle.spec.name for handle in kill_targets)

        still_running = tuple(
            handle.spec.name for handle in handles
            if self._poll(handle) is None)
        exit_status = tuple(
            (handle.spec.name, self._poll(handle)) for handle in handles)
        for handle in handles:
            if self._poll(handle) is not None:
                self._close_log(handle)
        report = ShutdownReport(
            attempted=attempted,
            graceful_stops=graceful_stops,
            terminated=terminated,
            forced_kills=forced_kills,
            still_running=still_running,
            exit_status=exit_status,
            diagnostics=tuple(diagnostics),
        )
        self.last_shutdown_report = report
        return report

    def stop(self, name):
        """Stop one explicitly owned process group."""
        return self.shutdown((name,))

    def stop_all(self):
        """Preserve the coordinator's historical boolean cleanup result."""
        return self.shutdown().complete
