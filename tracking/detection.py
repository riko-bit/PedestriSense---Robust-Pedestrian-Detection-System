import numpy as np

class Detection:
    """
    This class represents a bounding box detection in the image.

    Parameters:
        bbox: list -> [x, y, width, height]
        confidence: float
        feature: Optional appearance embedding (None for now)
    """
    def __init__(self, bbox, confidence, feature):
        self.bbox = np.asarray(bbox, dtype=float)
        self.confidence = float(confidence)
        self.feature = feature

    def to_tlwh(self):
        """Convert bbox to top-left width-height format."""
        return self.bbox.copy()

    def to_xyah(self):
        """Convert bbox to x,y,aspect_ratio,height format."""
        x, y, w, h = self.bbox
        return np.array([x + w / 2, y + h / 2, w / h, h])

    def to_tlbr(self):
        """Convert bbox to top-left bottom-right format."""
        x, y, w, h = self.bbox
        return np.array([x, y, x + w, y + h])
