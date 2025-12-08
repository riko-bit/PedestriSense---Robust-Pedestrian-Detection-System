import numpy as np
from .kalman_filter import KalmanFilterDeepSORT

class Track:
    """
    DeepSORT Track class
    """

    def __init__(self, measurement, track_id, n_init, max_age):
        self.track_id = track_id
        self.n_init = n_init
        self.max_age = max_age

        self.kf = KalmanFilterDeepSORT()
        self.mean, self.covariance = self.kf.initiate(measurement)

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = "Tentative"

    def predict(self):
        self.mean, self.covariance = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        measurement = detection.to_xyah()
        self.mean, self.covariance = self.kf.update(measurement)

        self.hits += 1
        self.time_since_update = 0

        if self.hits >= self.n_init:
            self.state = "Confirmed"

    def mark_missed(self):
        if self.time_since_update > self.max_age:
            self.state = "Deleted"

    def is_confirmed(self):
        return self.state == "Confirmed"

    def is_deleted(self):
        return self.state == "Deleted"

    def to_tlbr(self):
        """
        Convert [cx,cy,aspect,h] back to [x1,y1,x2,y2]
        """
        cx, cy, a, h = self.mean[:4]
        w = a * h
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2])
