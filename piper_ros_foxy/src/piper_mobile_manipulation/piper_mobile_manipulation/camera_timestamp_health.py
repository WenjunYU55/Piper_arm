"""Pure camera timestamp validation used by the ROS watchdog."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimestampHealthResult:
    state: str
    healthy: bool
    offset_sec: float
    consecutive_healthy_frames: int
    monotonic: bool
    reason: str


class TimestampHealthMonitor:
    """Detect stale/future stamps, backwards jumps, and startup stability."""

    def __init__(
            self, max_offset_sec=0.5, backward_tolerance_sec=0.001,
            healthy_frames_required=15):
        self.max_offset_sec = max(0.0, float(max_offset_sec))
        self.backward_tolerance_sec = max(0.0, float(backward_tolerance_sec))
        self.healthy_frames_required = max(1, int(healthy_frames_required))
        self.last_stamp_sec = None
        self.consecutive_healthy_frames = 0

    def evaluate(self, stamp_sec, now_sec):
        stamp_sec = float(stamp_sec)
        now_sec = float(now_sec)
        offset = now_sec - stamp_sec
        finite = math.isfinite(stamp_sec) and math.isfinite(now_sec) and stamp_sec > 0.0
        monotonic = finite and (
            self.last_stamp_sec is None
            or stamp_sec >= self.last_stamp_sec - self.backward_tolerance_sec)
        if finite:
            self.last_stamp_sec = stamp_sec

        if not finite:
            return self._fault('INVALID_STAMP', offset, monotonic, 'camera timestamp is invalid')
        if not monotonic:
            return self._fault(
                'NON_MONOTONIC', offset, False,
                'camera timestamp moved backwards')
        if abs(offset) > self.max_offset_sec:
            direction = 'stale' if offset > 0.0 else 'in the future'
            return self._fault(
                'CLOCK_OFFSET', offset, True,
                'camera timestamp is %s by %.3fs (limit %.3fs)'
                % (direction, abs(offset), self.max_offset_sec))

        self.consecutive_healthy_frames += 1
        if self.consecutive_healthy_frames < self.healthy_frames_required:
            return TimestampHealthResult(
                'STARTING', False, offset, self.consecutive_healthy_frames, True,
                'waiting for %d consecutive timestamp-valid frames (%d/%d)'
                % (
                    self.healthy_frames_required,
                    self.consecutive_healthy_frames,
                    self.healthy_frames_required,
                ),
            )
        return TimestampHealthResult(
            'HEALTHY', True, offset, self.consecutive_healthy_frames, True,
            'camera timestamps are healthy')

    def no_frames(self, reason='camera frames are missing or stale'):
        return self._fault('NO_FRAMES', math.inf, True, reason)

    def _fault(self, state, offset, monotonic, reason):
        self.consecutive_healthy_frames = 0
        return TimestampHealthResult(state, False, offset, 0, monotonic, reason)
