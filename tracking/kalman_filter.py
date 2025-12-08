import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanFilterDeepSORT:
    """
    DeepSORT Kalman filter for bounding box tracking.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R *= 1.0
        self.kf.P *= 10.0
        self.kf.Q *= 0.01

    def initiate(self, measurement):
        """Initialize track state."""
        mean = np.r_[measurement, np.zeros(3)]
        covariance = np.eye(7) * 10.
        self.kf.x = mean
        self.kf.P = covariance
        return self.kf.x.copy(), self.kf.P.copy()

    def predict(self):
        """Run Kalman Filter predict step."""
        self.kf.predict()
        return self.kf.x.copy(), self.kf.P.copy()

    def update(self, measurement):
        """Update step and return updated mean & covariance."""
        self.kf.update(measurement)
        return self.kf.x.copy(), self.kf.P.copy()
