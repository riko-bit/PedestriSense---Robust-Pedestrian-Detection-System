# capture_behavior_dataset.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Create folders
dataset_path = "behavior_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Classes you want to capture
classes = ["standing", "sitting", "waving"]  # add more if needed

# Number of frames to capture per class
frames_per_class = 100

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

data_X = []
data_y = []

cap = cv2.VideoCapture(0)

for label in classes:
    print(f"Get ready to capture '{label}' in 3 seconds...")
    cv2.waitKey(3000)

    count = 0
    while count < frames_per_class:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = []
            for lm in landmarks:
                features.append(lm.x)
                features.append(lm.y)
            features = np.array(features)
            data_X.append(features)
            data_y.append(label)
            count += 1

            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"Capturing '{label}': {count}/{frames_per_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Dataset", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save dataset
dataset_file = os.path.join(dataset_path, "behavior_dataset.pkl")
with open(dataset_file, "wb") as f:
    pickle.dump({"X": np.array(data_X), "y": np.array(data_y)}, f)

print(f"Dataset saved to {dataset_file}")
