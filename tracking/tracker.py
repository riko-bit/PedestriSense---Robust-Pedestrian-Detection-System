import numpy as np
from .track import Track

class Tracker:
    """
    DeepSORT Tracker: manages Track objects.
    """

    def __init__(self, metric, max_age=30, n_init=3):
        self.metric = metric
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """
        Run Kalman prediction on all tracks.
        """
        for track in self.tracks:
            track.predict()

    def update(self, detections):
        """
        Match detections to existing tracks and update tracker.
        """

        matches, unmatched_tracks, unmatched_detections = self._assign_detections_to_tracks(detections)

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            measurement = detections[det_idx].to_xyah()
            new_track = Track(
                measurement=measurement,
                track_id=self._next_id,
                n_init=self.n_init,
                max_age=self.max_age
            )
            self.tracks.append(new_track)
            self._next_id += 1

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _assign_detections_to_tracks(self, detections):
        """
        VERY SIMPLE matching logic (placeholder for cosine metric).
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        # Create a dummy cost matrix (random)
        cost_matrix = np.random.rand(len(self.tracks), len(detections))

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        threshold = 0.6

        for t in unmatched_tracks.copy():
            for d in unmatched_detections.copy():
                if cost_matrix[t, d] < threshold:
                    matches.append((t, d))
                    unmatched_tracks.remove(t)
                    unmatched_detections.remove(d)

        return matches, unmatched_tracks, unmatched_detections
