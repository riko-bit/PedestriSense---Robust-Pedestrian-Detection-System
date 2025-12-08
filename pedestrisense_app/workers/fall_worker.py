# pedestrisense_app/workers/fall_worker.py
import cv2
import numpy as np
import mediapipe as mp
import time
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class FallWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.running = True
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.previous_hip_y = None
        self.fall_detected = False
        self.fall_time = 0

    @pyqtSlot(object)
    def receive_frame(self, frame):
        self.current_frame = frame

    def calculate_angle(self, a,b,c):
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b; bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

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
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            label = "Safe"
            color = (0, 255, 0)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                # Try to get landmarks with safe indices
                try:
                    shoulder = [lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x*w,
                                lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y*h]
                    hip = [lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x*w,
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y*h]
                    knee = [lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x*w,
                            lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y*h]
                    angle = self.calculate_angle(shoulder, hip, knee)

                    xs = [p.x*w for p in lm]
                    ys = [p.y*h for p in lm]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))

                    box_w = x2 - x1; box_h = y2 - y1
                    ratio = box_w / (box_h + 1e-6)

                    if self.previous_hip_y is None:
                        self.previous_hip_y = hip[1]

                    hip_drop = self.previous_hip_y - hip[1]
                    self.previous_hip_y = hip[1]

                    rule1 = angle < 50
                    rule2 = hip_drop < -70
                    rule3 = ratio > 1.2

                    if (rule1 + rule2 + rule3) >= 2:
                        self.fall_detected = True
                        self.fall_time = time.time()

                    if self.fall_detected and time.time() - self.fall_time < 2.5:
                        label = "FALL DETECTED"
                        color = (0,0,255)
                    else:
                        self.fall_detected = False

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    self.mp_draw.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                except Exception:
                    pass

            pix = self.to_pixmap(frame)
            self.frame_ready.emit(pix)
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait(timeout=1000)
