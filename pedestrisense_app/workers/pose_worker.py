# pedestrisense_app/workers/pose_worker.py
import os
import cv2
import numpy as np
import joblib
from collections import deque
from statistics import mode
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class PoseWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.running = True
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        # load classifier if exists
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        clf_path = os.path.join(base, "pose_behavior", "behavior_rf_model.pkl")
        self.clf = None
        self.history = deque(maxlen=6)
        if os.path.exists(clf_path):
            try:
                self.clf = joblib.load(clf_path)
            except Exception as e:
                print("PoseWorker: failed to load behavior model:", e)
                self.clf = None
        else:
            print("PoseWorker: behavior model not found; behavior will be Unknown.")

    @pyqtSlot(object)
    def receive_frame(self, frame):
        self.current_frame = frame

    def to_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def run(self):
        while self.running:
            if self.current_frame is None:
                self.msleep(10)
                continue

            frame = self.current_frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            behavior = "Unknown"
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                features = []
                for p in lm:
                    features.extend([p.x, p.y])
                features = np.array(features).reshape(1, -1)
                if self.clf:
                    try:
                        pred = self.clf.predict(features)[0]
                        self.history.append(pred)
                        behavior = mode(self.history)
                    except:
                        behavior = "Unknown"

            cv2.putText(frame, f"Behavior: {behavior}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            pix = self.to_pixmap(frame)
            self.frame_ready.emit(pix)
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait(timeout=1000)
