import numpy as np

class NearestNeighborDistanceMetric:
    """
    A simple cosine-distance based matching metric for DeepSORT.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget

    def distance(self, features, targets):
        """Compute cosine distance."""
        features = np.asarray(features)
        targets = np.asarray(targets)
        num = np.dot(features, targets.T)
        den = np.linalg.norm(features, axis=1)[:, None] * np.linalg.norm(targets, axis=1)
        similarity = num / np.maximum(den, 1e-6)
        return 1 - similarity
