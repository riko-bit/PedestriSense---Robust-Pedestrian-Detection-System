import cv2
import numpy as np
import os
from collections import deque
from statistics import mode

# ---------------------------
# Paths to model files
# ---------------------------
base_path = os.path.join(os.path.dirname(__file__), "models")

face_proto = os.path.join(base_path, "deploy.prototxt")
face_model = os.path.join(base_path, "res10_300x300_ssd_iter_140000.caffemodel")
gender_proto = os.path.join(base_path, "deploy_gender.prototxt")
gender_model = os.path.join(base_path, "gender_net.caffemodel")
age_proto = os.path.join(base_path, "deploy_age.prototxt")
age_model = os.path.join(base_path, "age_net.caffemodel")

# ---------------------------
# Verify files exist
# ---------------------------
for file_path in [face_proto, face_model, gender_proto, gender_model, age_proto, age_model]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

# ---------------------------
# Load models
# ---------------------------
face_net = cv2.dnn.readNet(face_model, face_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)

# ---------------------------
# Define lists
# ---------------------------
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# ---------------------------
# Frame history for smoothing
# ---------------------------
gender_history = deque(maxlen=10)  # store last 10 predictions

# ---------------------------
# Start video capture
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)

            # Expand bounding box for better coverage
            padding = 20
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(w - 1, endX + padding)
            endY = min(h - 1, endY + padding)

            face = frame[startY:endY, startX:endX].copy()
            if face.size == 0:
                continue

            # Prepare blob for gender & age classification
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746],                      swapRB=True)  # RGB channels

            # Predict gender
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender_index = gender_preds[0].argmax()
            gender_history.append(gender_index)
            # Take mode of last 10 frames
            final_gender = GENDER_LIST[mode(gender_history)]

            # Predict age
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            # Draw bounding box and label
            label = f"{final_gender}, {age}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Age & Gender Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
