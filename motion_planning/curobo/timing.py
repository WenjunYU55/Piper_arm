"""Best-effort host timings; never synchronize CUDA or influence planning."""

from contextlib import contextmanager
from functools import wraps
import math
import time


def clock():
    try:
        return time.perf_counter()
    except Exception:
        return None


class RequestTiming:
    """Bounded per-request evidence. Nested stage durations are inclusive."""

    def __init__(self):
        self.started = clock()
        self.stages = {}
        self.events = []
        self.dropped_events = 0

    @contextmanager
    def stage(self, name, context=None):
        started = clock()
        event = dict(context or {})
        event['stage'] = name
        try:
            yield event
        except BaseException as error:
            event['exception_type'] = type(error).__name__
            event['exception_message'] = str(error)
            raise
        finally:
            try:
                elapsed = clock() - started
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('invalid timing clock')
                event['wall_sec'] = elapsed
                event['start_offset_sec'] = started - self.started
                total = self.stages.setdefault(name, {'calls': 0, 'wall_sec': 0.0})
                total['calls'] += 1
                total['wall_sec'] += elapsed
                if len(self.events) < 1024:
                    self.events.append(event)
                else:
                    self.dropped_events += 1
            except Exception:
                pass  # Instrumentation cannot replace a result or exception.

    def snapshot(self):
        return {
            'schema_version': 1,
            'clock': 'perf_counter_host_no_explicit_cuda_sync',
            'stage_durations_inclusive': True,
            'stages': {key: dict(value) for key, value in self.stages.items()},
            'events': [dict(event) for event in self.events],
            'dropped_events': self.dropped_events,
        }


def timed_stage(name, reset=False):
    """Observe a backend method without changing arguments/results/errors."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if reset:
                self.request_timing = RequestTiming()
            timing = getattr(self, 'request_timing', None)
            if timing is None:
                return method(self, *args, **kwargs)
            context = {}
            if name == 'candidate' and len(args) > 1:
                candidate = args[1]
                context = {key: candidate[key] for key in ('id', 'ray_id')
                           if key in candidate}
            try:
                with timing.stage(name, context) as event:
                    result = method(self, *args, **kwargs)
                    if name in ('attached_tool_check', 'visibility_check'):
                        event['rejection_reason'] = result
                    return result
            finally:
                if reset:
                    try:
                        self.last_planning_diagnostics['timing'] = timing.snapshot()
                    except Exception:
                        pass
        return wrapped
    return decorate


def native_metrics(result, event):
    """Copy existing native scalar diagnostics, not inferred GPU kernel times."""
    try:
        for field in ('total_time', 'solve_time', 'ik_time', 'graph_time',
                      'trajopt_time', 'finetune_time'):
            value = getattr(result, field, None)
            if value is not None:
                value = float(value)
                if math.isfinite(value) and value >= 0:
                    event['native_' + field + '_sec'] = value
        event['native_status'] = str(result.status)
    except Exception:
        pass
