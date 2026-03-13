import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import os
from typing import Dict, Any, List

# -------------------- DeepSORT imports --------------------
from tracking.detection import Detection
from tracking.tracker import Tracker

# -------------------- Paths --------------------
PERSON_PROTO = "person_detection/deploy.prototxt"
PERSON_MODEL = "person_detection/mobilenet_iter_73000.caffemodel"

GENDER_PROTO = "age_gender/models/deploy_gender.prototxt"
GENDER_MODEL = "age_gender/models/gender_net.caffemodel"

AGE_PROTO = "age_gender/models/deploy_age.prototxt"
AGE_MODEL = "age_gender/models/age_net.caffemodel"

BEHAVIOR_MODEL = "pose_behavior/behavior_rf_model.pkl"


# -------------------- Helpers --------------------
def load_dnn_net(proto, model):
    return cv2.dnn.readNetFromCaffe(proto, model)


def safe_load_joblib(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


# -------------------- Load models (once) --------------------
person_net = load_dnn_net(PERSON_PROTO, PERSON_MODEL)
gender_net = load_dnn_net(GENDER_PROTO, GENDER_MODEL)
age_net = load_dnn_net(AGE_PROTO, AGE_MODEL) if os.path.exists(AGE_MODEL) else None

behavior_clf = safe_load_joblib(BEHAVIOR_MODEL)

GENDER_LIST = ["Male", "Female"]
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
            "(38-43)", "(48-53)", "(60-100)"]

# Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# DeepSORT tracker
tracker = Tracker(max_age=30, n_init=3)

# FPS smoothing
_last_time = time.time()
_last_fps = 0.0


# -------------------- Utility --------------------
def _extract_keypoints(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append([int(lm.x * w), int(lm.y * h)])
    return pts


# -------------------- MAIN PIPELINE --------------------
def run_inference(frame: np.ndarray) -> Dict[str, Any]:
    """
    FULL pipeline: detection, tracking, pose, age/gender, activity.
    Also draws boxes on the frame.
    """
    global _last_time, _last_fps

    h, w = frame.shape[:2]

    # -----------------------------------------------------
    # 1) PERSON DETECTION (MobileNet-SSD)
    # -----------------------------------------------------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )
    person_net.setInput(blob)
    dets = person_net.forward()

    detections = []
    for i in range(dets.shape[2]):
        confidence = float(dets[0, 0, i, 2])
        cls = int(dets[0, 0, i, 1])

        if confidence < 0.5 or cls != 15:
            continue

        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # DeepSORT needs tlwh (x, y, w, h)
        detections.append(
            Detection(
                tlwh=[x1, y1, x2 - x1, y2 - y1],
                confidence=confidence,
                feature=np.zeros(128)  # feature extractor skipped
            )
        )

    # -----------------------------------------------------
    # 2) TRACKING
    # -----------------------------------------------------
    tracker.predict()
    tracker.update(detections)

    # -----------------------------------------------------
    # 3) POSE + BEHAVIOR ONCE PER FRAME
    # -----------------------------------------------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_result = pose.process(frame_rgb)

    landmarks = mp_result.pose_landmarks.landmark if mp_result.pose_landmarks else None

    features = None
    if landmarks:
        feat = []
        for lm in landmarks:
            feat.append(lm.x)
            feat.append(lm.y)
        features = np.array(feat).reshape(1, -1)

    # -----------------------------------------------------
    # 4) FOR EACH TRACK → assign attributes
    # -----------------------------------------------------
    persons_out = []

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        x1, y1, x2, y2 = track.to_tlbr().astype(int)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # ------------------ AGE/GENDER ------------------
        gender = ""
        age = ""
        try:
            face = frame[y1:int(y1 + (y2 - y1) * 0.35), x1:x2]
            if face.size > 0:
                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227),
                    [78.4, 87.77, 114.9],
                    swapRB=False
                )

                gender_net.setInput(blob)
                gender = GENDER_LIST[int(np.argmax(gender_net.forward()[0]))]

                if age_net:
                    age_net.setInput(blob)
                    age = AGE_LIST[int(np.argmax(age_net.forward()[0]))]
        except:
            pass

        # ------------------ POSE → keypoints inside bbox ------------------
        keypoints = []
        if landmarks:
            all_kp = _extract_keypoints(landmarks, w, h)
            for kx, ky in all_kp:
                if x1 <= kx <= x2 and y1 <= ky <= y2:
                    keypoints.append([kx, ky])

        # ------------------ BEHAVIOR ------------------
        activity = "Unknown"
        fall_flag = False

        if features is not None and behavior_clf is not None:
            try:
                pred = behavior_clf.predict(features)[0]
                activity = str(pred)
                if "fall" in activity.lower():
                    fall_flag = True
            except:
                pass

        # Fallback fall logic using body angle:
        if landmarks and not fall_flag:
            try:
                ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                ms_x = (ls.x + rs.x) / 2
                ms_y = (ls.y + rs.y) / 2
                mh_x = (lh.x + rh.x) / 2
                mh_y = (lh.y + rh.y) / 2

                vx = mh_x - ms_x
                vy = mh_y - ms_y

                angle = abs(np.degrees(np.arctan2(vx, vy)))
                if angle > 45:
                    fall_flag = True
                    activity = "fallen"
            except:
                pass

        # ------------------ DRAW BOX ------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        persons_out.append({
            "id": int(track.track_id),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": 1.0,
            "gender": gender,
            "age": age,
            "keypoints": keypoints,
            "activity": activity,
            "fall": bool(fall_flag)
        })

    # -----------------------------------------------------
    # 5) FPS
    # -----------------------------------------------------
    now = time.time()
    fps = 1.0 / (now - _last_time)
    _last_time = now
    _last_fps = round(fps, 2)

    return {
        "fps": _last_fps,
        "resolution": [w, h],
        "persons": persons_out,
        "frame": frame  # return frame WITH boxes drawn
    }
