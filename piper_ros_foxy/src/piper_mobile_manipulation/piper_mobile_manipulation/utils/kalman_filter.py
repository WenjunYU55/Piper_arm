import numpy as np


class ConstantVelocityKalmanFilter:
    """
    Provide a small linear filter with an optional near-static target model.

    ``process_noise`` and ``measurement_noise`` are standard deviations in
    metres.  The target tracker sets ``velocity_retention`` to zero because
    the scanned target is stationary in ``base_link``.  The six-element state
    shape is retained for compatibility with existing TrackedTarget output.
    """

    def __init__(
            self, process_noise=0.05, measurement_noise=0.02,
            velocity_retention=1.0):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.velocity_retention = float(np.clip(
            velocity_retention, 0.0, 1.0))
        self.x = np.zeros((6, 1), dtype=float)
        self.p = np.eye(6, dtype=float)
        self.initialized = False

    def reset(self):
        self.x[:] = 0.0
        self.p = np.eye(6, dtype=float)
        self.initialized = False

    def initialize(self, measurement):
        self.x[:] = 0.0
        self.x[0:3, 0] = np.asarray(measurement, dtype=float)
        measurement_variance = max(self.measurement_noise, 1.0e-6) ** 2
        self.p[:] = 0.0
        self.p[0:3, 0:3] = np.eye(3, dtype=float) * (
            0.25 * measurement_variance)
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return self.state
        dt = max(float(dt), 1e-3)
        f = np.eye(6, dtype=float)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        f[3, 3] = self.velocity_retention
        f[4, 4] = self.velocity_retention
        f[5, 5] = self.velocity_retention
        process_variance = max(self.process_noise, 0.0) ** 2 * dt
        q = np.zeros((6, 6), dtype=float)
        q[0:3, 0:3] = np.eye(3, dtype=float) * process_variance
        if self.velocity_retention > 0.0:
            q[3:6, 3:6] = np.eye(3, dtype=float) * process_variance
        self.x = f.dot(self.x)
        self.p = f.dot(self.p).dot(f.T) + q
        return self.state

    def innovation(self, measurement, measurement_noise=None):
        """Return residual, covariance and normalized innovation score."""
        z = np.asarray(measurement, dtype=float).reshape((3, 1))
        h = np.zeros((3, 6), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        h[2, 2] = 1.0
        noise = self.measurement_noise \
            if measurement_noise is None else float(measurement_noise)
        r = np.eye(3, dtype=float) * max(noise, 1.0e-6) ** 2
        y = z - h.dot(self.x)
        s = h.dot(self.p).dot(h.T) + r
        score = float(y.T.dot(np.linalg.solve(s, y))[0, 0])
        return y[:, 0].copy(), s.copy(), score

    def update(self, measurement, measurement_noise=None):
        z = np.asarray(measurement, dtype=float).reshape((3, 1))
        h = np.zeros((3, 6), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        h[2, 2] = 1.0
        noise = self.measurement_noise \
            if measurement_noise is None else float(measurement_noise)
        r = np.eye(3, dtype=float) * max(noise, 1.0e-6) ** 2
        y = z - h.dot(self.x)
        s = h.dot(self.p).dot(h.T) + r
        k = self.p.dot(h.T).dot(np.linalg.inv(s))
        self.x = self.x + k.dot(y)
        i = np.eye(6, dtype=float)
        self.p = (i - k.dot(h)).dot(self.p)
        if self.velocity_retention == 0.0:
            self.x[3:6, 0] = 0.0
        return self.state

    def step(self, measurement, dt):
        if not self.initialized:
            self.initialize(measurement)
        else:
            self.predict(dt)
            self.update(measurement)
        return self.state

    @property
    def maximum_position_stddev(self):
        if not self.initialized:
            return float('inf')
        diagonal = np.maximum(np.diag(self.p)[0:3], 0.0)
        return float(np.sqrt(np.max(diagonal)))

    @property
    def state(self):
        return self.x[:, 0].copy()
