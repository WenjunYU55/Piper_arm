"""Stabilize asynchronous controller-limit snapshots without weakening freshness."""


class MotionLimitStability:
    """Accept a changed valid hash only after it remains consistent for a window."""

    def __init__(self, confirmation_sec=7.0, minimum_samples=3):
        self.confirmation_sec = max(0.0, float(confirmation_sec))
        self.minimum_samples = max(1, int(minimum_samples))
        self.accepted = None
        self.accepted_at = None
        self.candidate = None
        self.candidate_hash = ''
        self.candidate_started_at = None
        self.candidate_samples = 0

    def observe(self, message, now):
        """Return the currently accepted message and whether it was refreshed."""
        timestamp = float(now)
        if not bool(getattr(message, 'valid', False)):
            return self.accepted, False
        digest = str(getattr(message, 'limits_sha256', ''))
        if len(digest) != 64:
            return self.accepted, False
        if self.accepted is None:
            self.accepted = message
            self.accepted_at = timestamp
            self._clear_candidate()
            return self.accepted, True
        accepted_hash = str(getattr(self.accepted, 'limits_sha256', ''))
        if digest == accepted_hash:
            self.accepted = message
            self.accepted_at = timestamp
            self._clear_candidate()
            return self.accepted, True
        if digest != self.candidate_hash:
            self.candidate = message
            self.candidate_hash = digest
            self.candidate_started_at = timestamp
            self.candidate_samples = 1
            return self.accepted, False
        self.candidate = message
        self.candidate_samples += 1
        elapsed = timestamp - float(self.candidate_started_at)
        if (
                elapsed >= self.confirmation_sec
                and self.candidate_samples >= self.minimum_samples):
            self.accepted = self.candidate
            self.accepted_at = timestamp
            self._clear_candidate()
            return self.accepted, True
        return self.accepted, False

    def _clear_candidate(self):
        self.candidate = None
        self.candidate_hash = ''
        self.candidate_started_at = None
        self.candidate_samples = 0
