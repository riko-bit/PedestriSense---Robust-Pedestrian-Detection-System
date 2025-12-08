# person_detection.py
import cv2
import numpy as np
import os
import sys

# ---- Paths to model files (same folder as script) ----
proto_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"

# ---- Check if files exist ----
if not os.path.exists(proto_path):
    print(f"Error: '{proto_path}' not found. Please place it in the same folder as this script.")
    sys.exit(1)
if not os.path.exists(model_path):
    print(f"Error: '{model_path}' not found. Please place it in the same folder as this script.")
    sys.exit(1)

# ---- Load the model ----
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# ---- Class labels ----
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ---- Start webcam ----
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare input blob for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
