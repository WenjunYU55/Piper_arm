"""Command-free timing regressions: no cuRobo, CUDA or ROS runtime."""

import json
from types import SimpleNamespace

import pytest

from motion_planning.curobo import timing
from motion_planning.curobo.worker import CuroboBackend


def test_nested_failed_and_successful_work_is_counted_and_reset(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(timing, 'clock', lambda: now[0])

    class Backend:
        @timing.timed_stage('backend_plan', reset=True)
        def plan(self):
            self.last_planning_diagnostics = {}
            for fail in (True, False):
                try:
                    self.attempt(fail)
                except ValueError:
                    pass
            return 'same result'

        @timing.timed_stage('motiongen')
        def attempt(self, fail):
            now[0] += 2.0
            if fail:
                raise ValueError('same failure')

    backend = Backend()
    assert backend.plan() == 'same result'
    metrics = backend.last_planning_diagnostics['timing']
    assert metrics['stages']['motiongen'] == {'calls': 2, 'wall_sec': 4.0}
    assert metrics['stages']['backend_plan']['wall_sec'] == 4.0
    assert metrics['events'][0]['exception_message'] == 'same failure'
    assert metrics['events'][1]['start_offset_sec'] == 2.0
    json.dumps(metrics, allow_nan=False)
    backend.plan()
    assert backend.last_planning_diagnostics['timing']['stages']['motiongen']['calls'] == 2


def test_native_call_preserves_arguments_results_and_failures():
    backend = object.__new__(CuroboBackend)
    backend.request_timing = timing.RequestTiming()
    result = SimpleNamespace(total_time=0.3, ik_time=0.02, graph_time=0.1,
                             trajopt_time=0.18, status='SUCCESS')
    args = (object(), object(), object())

    def native(*received):
        assert received == args
        return result

    assert backend._timed_native_plan(native, *args, {'semantic_goals': 6}) is result
    event = backend.request_timing.events[0]
    assert event['native_ik_time_sec'] == 0.02
    assert 'native_finetune_time_sec' not in event  # Missing is not zero.
    error = RuntimeError('native failure')

    def fails(*_args):
        raise error

    with pytest.raises(RuntimeError) as raised:
        backend._timed_native_plan(fails, *args, {'semantic_goals': 3})
    assert raised.value is error
    assert backend.request_timing.stages['motiongen']['calls'] == 2
    assert backend.request_timing.events[-1]['exception_type'] == 'RuntimeError'


def test_clock_failure_never_changes_native_result(monkeypatch):
    monkeypatch.setattr(timing, 'clock', lambda: None)
    backend = object.__new__(CuroboBackend)
    backend.request_timing = timing.RequestTiming()
    result = object()
    assert backend._timed_native_plan(lambda *_: result, 1, 2, 3, {}) is result


def test_events_are_bounded_but_totals_include_all_calls():
    recorder = timing.RequestTiming()
    for _ in range(1026):
        with recorder.stage('check'):
            pass
    assert len(recorder.events) == 1024
    assert recorder.dropped_events == 2
    assert recorder.stages['check']['calls'] == 1026


def test_worker_persists_final_timing_on_typed_failure_without_cross_request_leak(
        monkeypatch, tmp_path, capsys):
    import motion_planning.curobo.worker as module
    from motion_planning.curobo.adapter import CuroboCandidateExhausted, verify_digest

    class Backend:
        version = 'test'

        @timing.timed_stage('backend_plan', reset=True)
        def plan(self, _request):
            self.last_planning_diagnostics = {}
            with self.request_timing.stage('motiongen'):
                raise CuroboCandidateExhausted('all failed', self.last_planning_diagnostics)

    responses = []
    worker = object.__new__(module.Worker)
    worker.backend = Backend()
    worker.spool = SimpleNamespace(
        claim_next=lambda: ('r', {'plan_kind': 'MULTIVIEW_SCAN'}),
        write_response=lambda _id, response: responses.append(response),
        path=lambda *_args: tmp_path / 'absent',
    )
    monkeypatch.setattr(module, 'validate_request', lambda _: None)
    assert worker.process_once()
    metrics = responses[-1]['planning_diagnostics']
    assert metrics['timing']['stages']['backend_plan']['calls'] == 1
    assert metrics['timing']['events'][0]['exception_type'] == 'CuroboCandidateExhausted'
    verify_digest(responses[-1], 'response_sha256')
    log = json.loads(capsys.readouterr().out.split('CUROBO_REQUEST_TIMING ', 1)[1])
    assert log['metrics']['worker_timing']['stages']['response_publication']['calls'] == 1

    def invalid(_request):
        raise ValueError('bad request')

    monkeypatch.setattr(module, 'validate_request', invalid)
    assert worker.process_once()
    assert 'timing' not in responses[-1]['planning_diagnostics']
    assert responses[-1]['diagnostic'] == 'bad request'
