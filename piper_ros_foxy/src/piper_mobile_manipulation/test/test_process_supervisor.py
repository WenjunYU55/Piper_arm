"""Characterize owned process startup, monitoring, and bounded cleanup."""

import signal

import pytest

from piper_mobile_manipulation.process_supervisor import (
    ProcessSpec,
    ProcessSupervisor,
)


class FakeClock:
    """Monotonic clock advanced by the supervisor's fake sleeper."""

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += float(seconds)


class FakeProcess:
    """Popen-shaped process with signal-selectable exit behavior."""

    def __init__(self, pid, returncode=None, exits_on=None):
        self.pid = int(pid)
        self.returncode = returncode
        self.exits_on = dict(exits_on or {})

    def poll(self):
        return self.returncode

    def receive(self, signum):
        if signum in self.exits_on:
            self.returncode = self.exits_on[signum]


class FakePopenFactory:
    """Deterministic Popen factory that records exact startup arguments."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeGroupSignaler:
    """Signal only process-group IDs explicitly registered by a test."""

    def __init__(self, processes):
        self.processes = {process.pid: process for process in processes}
        self.calls = []

    def __call__(self, process_group_id, signum):
        self.calls.append((int(process_group_id), signum))
        process = self.processes.get(int(process_group_id))
        if process is None:
            raise ProcessLookupError(process_group_id)
        process.receive(signum)


def supervisor(tmp_path, processes, *, force_kill=False, outcomes=None):
    clock = FakeClock()
    factory = FakePopenFactory(
        list(processes) if outcomes is None else outcomes)
    signaler = FakeGroupSignaler(processes)
    instance = ProcessSupervisor(
        tmp_path, popen_factory=factory, group_signaler=signaler,
        clock=clock, sleeper=clock.sleep,
        graceful_timeout_sec=0.10, terminate_timeout_sec=0.10,
        force_kill=force_kill, kill_timeout_sec=0.10)
    return instance, factory, signaler


def test_successful_startup_uses_new_group_log_and_exact_environment(tmp_path):
    process = FakeProcess(101)
    owner, factory, _signaler = supervisor(tmp_path, [process])
    environment = ProcessSupervisor.build_environment(
        {'MISSION_ONLY': 'yes'}, {'INHERITED': 'kept'})

    handle = owner.start('driver', ['/safe/fake', '--ready'], environment)

    assert handle.process is process
    assert handle.process_group_id == 101
    assert owner.owned_names() == ('driver',)
    command, kwargs = factory.calls[0]
    assert command == ['/safe/fake', '--ready']
    assert kwargs['env'] == {'INHERITED': 'kept', 'MISSION_ONLY': 'yes'}
    assert kwargs['start_new_session'] is True
    assert kwargs['stdout'] is handle.log_stream
    assert kwargs['stderr'] == -2
    assert owner.health()['driver']['running'] is True
    handle.log_stream.write(b'fake child ready\n')
    assert 'fake child ready' in owner.log_since_start('driver')


def test_startup_failure_closes_log_and_records_no_ownership(tmp_path):
    failure = OSError('synthetic startup failure')
    owner, factory, _signaler = supervisor(
        tmp_path, [], outcomes=[failure])

    with pytest.raises(OSError, match='synthetic startup failure'):
        owner.start('vision', ['/safe/missing'], {'A': 'B'})

    assert owner.owned_names() == ()
    assert factory.calls[0][1]['stdout'].closed is True
    assert owner.has_live_processes() is False


def test_immediate_exit_and_later_crash_are_reported(tmp_path):
    immediate = FakeProcess(102, returncode=7)
    later = FakeProcess(103)
    owner, _factory, _signaler = supervisor(tmp_path, [immediate, later])

    owner.start('immediate', ['/safe/immediate'], {})
    owner.start('later', ['/safe/later'], {})
    assert owner.failed() == {'immediate': 7}

    later.returncode = 9
    assert owner.failed() == {'immediate': 7, 'later': 9}
    assert owner.health()['later']['returncode'] == 9


def test_graceful_termination_stops_with_sigint_only(tmp_path):
    process = FakeProcess(104, exits_on={signal.SIGINT: -2})
    owner, _factory, signaler = supervisor(tmp_path, [process])
    handle = owner.start('scan_stack', ['/safe/stack'], {})

    report = owner.shutdown()

    assert report.complete is True
    assert report.graceful_stops == ('scan_stack',)
    assert report.terminated == ()
    assert signaler.calls == [(104, signal.SIGINT)]
    assert handle.log_stream.closed is True


def test_graceful_timeout_escalates_to_sigterm(tmp_path):
    process = FakeProcess(105, exits_on={signal.SIGTERM: -15})
    owner, _factory, signaler = supervisor(tmp_path, [process])
    owner.start('vision', ['/safe/vision'], {})

    report = owner.shutdown()

    assert report.complete is True
    assert report.graceful_stops == ()
    assert report.terminated == ('vision',)
    assert signaler.calls == [
        (105, signal.SIGINT), (105, signal.SIGTERM)]


def test_autonomous_policy_reports_live_group_without_forced_kill(tmp_path):
    process = FakeProcess(106)
    owner, _factory, signaler = supervisor(tmp_path, [process])
    handle = owner.start('driver', ['/safe/driver'], {})

    report = owner.shutdown()

    assert report.complete is False
    assert report.still_running == ('driver',)
    assert report.forced_kills == ()
    assert signaler.calls == [
        (106, signal.SIGINT), (106, signal.SIGTERM)]
    assert 'SIGKILL is disabled' in report.diagnostics[0]
    assert handle.log_stream.closed is False


def test_explicit_forced_kill_policy_is_available_but_opt_in(tmp_path):
    process = FakeProcess(107, exits_on={signal.SIGKILL: -9})
    owner, _factory, signaler = supervisor(
        tmp_path, [process], force_kill=True)
    owner.start('gui_owned_stack', ['/safe/gui-stack'], {})

    report = owner.shutdown()

    assert report.complete is True
    assert report.forced_kills == ('gui_owned_stack',)
    assert signaler.calls == [
        (107, signal.SIGINT),
        (107, signal.SIGTERM),
        (107, signal.SIGKILL),
    ]


def test_multiple_cleanup_uses_reverse_startup_order(tmp_path):
    first = FakeProcess(108, exits_on={signal.SIGINT: -2})
    second = FakeProcess(109, exits_on={signal.SIGINT: -2})
    owner, _factory, signaler = supervisor(tmp_path, [first, second])
    owner.start('driver', ['/safe/driver'], {})
    owner.start('vision', ['/safe/vision'], {})

    report = owner.shutdown()

    assert report.attempted == ('vision', 'driver')
    assert signaler.calls == [
        (109, signal.SIGINT), (108, signal.SIGINT)]
    assert report.complete is True


def test_stop_one_leaves_other_owned_group_running(tmp_path):
    driver = FakeProcess(114, exits_on={signal.SIGINT: -2})
    vision = FakeProcess(115, exits_on={signal.SIGINT: -2})
    owner, _factory, signaler = supervisor(tmp_path, [driver, vision])
    owner.start('driver', ['/safe/driver'], {})
    owner.start('vision', ['/safe/vision'], {})

    report = owner.stop('vision')

    assert report.attempted == ('vision',)
    assert report.complete is True
    assert driver.returncode is None
    assert owner.health()['driver']['running'] is True
    assert signaler.calls == [(115, signal.SIGINT)]


def test_partial_startup_cleanup_stops_only_started_owner(tmp_path):
    driver = FakeProcess(110, exits_on={signal.SIGINT: -2})
    startup_failure = OSError('second child failed')
    owner, _factory, signaler = supervisor(
        tmp_path, [driver], outcomes=[driver, startup_failure])
    owner.start('driver', ['/safe/driver'], {})

    with pytest.raises(OSError, match='second child failed'):
        owner.start('vision', ['/safe/vision'], {})
    report = owner.shutdown()

    assert report.attempted == ('driver',)
    assert signaler.calls == [(110, signal.SIGINT)]
    assert report.complete is True


def test_repeated_missions_clear_stopped_generation_without_leak(tmp_path):
    first = FakeProcess(111, exits_on={signal.SIGINT: -2})
    second = FakeProcess(112, exits_on={signal.SIGINT: -2})
    owner, _factory, signaler = supervisor(tmp_path, [first, second])
    owner.start('driver', ['/safe/driver-one'], {})
    assert owner.stop_all() is True
    assert owner.begin_generation() == []

    owner.start('driver', ['/safe/driver-two'], {})
    assert owner.owned_names() == ('driver',)
    assert owner.stop_all() is True

    assert signaler.calls == [
        (111, signal.SIGINT), (112, signal.SIGINT)]


def test_unowned_process_is_never_signalled_or_adopted(tmp_path):
    owned = FakeProcess(113, exits_on={signal.SIGINT: -2})
    unrelated = FakeProcess(999, exits_on={signal.SIGINT: -2})
    owner, factory, _unused = supervisor(tmp_path, [owned])
    signaler = FakeGroupSignaler([owned, unrelated])
    owner._group_signaler = signaler
    owner.start('owned', ['/safe/owned'], {})

    missing_report = owner.stop('unrelated')
    all_report = owner.shutdown()

    assert missing_report.attempted == ()
    assert all_report.attempted == ('owned',)
    assert signaler.calls == [(113, signal.SIGINT)]
    assert unrelated.returncode is None
    assert len(factory.calls) == 1


def test_process_spec_is_immutable_and_owns_environment_copy():
    environment = {'TOKEN': 'original'}
    spec = ProcessSpec('worker', ('/safe/worker',), environment)
    environment['TOKEN'] = 'changed'

    assert spec.environment['TOKEN'] == 'original'
    with pytest.raises(TypeError):
        spec.environment['TOKEN'] = 'mutated'
