import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

cap = cv2.VideoCapture(0)

fall_detected = False
fall_time = 0
previous_hip_y = None

with mp_pose.Pose() as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        label = "Person"
        color = (0, 255, 0)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # Get coordinates
            shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                                 lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])

            hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                            lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h])

            knee = np.array([lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                             lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h])

            angle = calculate_angle(shoulder, hip, knee)

            # Bounding box
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            box_width = x2 - x1
            box_height = y2 - y1

            width_height_ratio = box_width / (box_height + 1e-6)

            # Rule 1: Body angle
            rule_angle = angle < 50   # Lying angle

            # Rule 2: Sudden hip drop
            if previous_hip_y is None:
                previous_hip_y = hip[1]

            hip_drop = previous_hip_y - hip[1]
            rule_drop = hip_drop < -70   # sudden downward movement

            previous_hip_y = hip[1]

            # Rule 3: Bounding box becomes horizontal
            rule_wide = width_height_ratio > 1.2

            # If 2 out of 3 rules are true → fall detected
            rules_true = sum([rule_angle, rule_drop, rule_wide])

            if rules_true >= 2:
                fall_detected = True
                fall_time = time.time()

            if fall_detected and (time.time() - fall_time < 2.5):
                label = "FALL DETECTED"
                color = (0, 0, 255)
            else:
                fall_detected = False

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Label above box
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + 220, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            # Draw skeleton
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Fall Detection (Upgraded)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()