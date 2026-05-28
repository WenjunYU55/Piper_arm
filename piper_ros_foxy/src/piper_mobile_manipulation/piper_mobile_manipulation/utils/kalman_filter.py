import numpy as np


class ConstantVelocityKalmanFilter:
    def __init__(self, process_noise=0.05, measurement_noise=0.02):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
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
        self.p = np.eye(6, dtype=float) * 0.1
        self.initialized = True

    def predict(self, dt):
        dt = max(float(dt), 1e-3)
        f = np.eye(6, dtype=float)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        q = np.eye(6, dtype=float) * self.process_noise
        self.x = f.dot(self.x)
        self.p = f.dot(self.p).dot(f.T) + q

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape((3, 1))
        h = np.zeros((3, 6), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        h[2, 2] = 1.0
        r = np.eye(3, dtype=float) * self.measurement_noise
        y = z - h.dot(self.x)
        s = h.dot(self.p).dot(h.T) + r
        k = self.p.dot(h.T).dot(np.linalg.inv(s))
        self.x = self.x + k.dot(y)
        i = np.eye(6, dtype=float)
        self.p = (i - k.dot(h)).dot(self.p)

    def step(self, measurement, dt):
        if not self.initialized:
            self.initialize(measurement)
        else:
            self.predict(dt)
            self.update(measurement)
        return self.state

    @property
    def state(self):
        return self.x[:, 0].copy()
