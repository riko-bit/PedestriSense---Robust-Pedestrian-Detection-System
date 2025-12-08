# realtime_behavior.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from statistics import mode

# Load trained model
clf = joblib.load("behavior_rf_model.pkl")

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

history = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    behavior = "Unknown"

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        features = []
        for lm in landmarks:
            features.append(lm.x)
            features.append(lm.y)
        features = np.array(features).reshape(1, -1)

        try:
            pred = clf.predict(features)[0]
            history.append(pred)
            behavior = mode(history)
        except:
            behavior = "Unknown"

    cv2.putText(frame, f"Behavior: {behavior}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Behavior Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
