# test_tracking.py
import cv2
import numpy as np

# Import DeepSORT modules
from tracking.tracker import Tracker
from tracking.nn_matching import NearestNeighborDistanceMetric
from tracking.detection import Detection

# Load Person detection model (MobileNet-SSD)
proto = "person detection/deploy.prototxt"
model = "person detection/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)

# Initialize DeepSORT
metric = NearestNeighborDistanceMetric("cosine", 0.4, None)
tracker = Tracker(metric)

cap = cv2.VideoCapture(0)

print("Running DeepSORT Tracking Test...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # --- Person Detection ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detection_list = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if confidence > 0.5 and idx == 15:  # 15 = person
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # DeepSORT bbox format: x, y, width, height
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detection_list.append(Detection(bbox, confidence, None))

    # --- DeepSORT Tracking ---
    tracker.predict()
    tracker.update(detection_list)

    # --- Draw Tracking ID ---
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_id = track.track_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("DeepSORT Tracking Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
