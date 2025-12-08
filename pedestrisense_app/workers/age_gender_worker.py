# pedestrisense_app/workers/age_gender_worker.py
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class AgeGenderWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.running = True

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_dir = os.path.join(base, "age_gender_cnn", "models")

        # We'll attempt to use the ResNet face detector (res10) if it's there, otherwise skip detection
        self.face_detector = None
        res10_proto = os.path.join(model_dir, "deploy.prototxt")
        res10_model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(res10_proto) and os.path.exists(res10_model):
            try:
                self.face_detector = cv2.dnn.readNetFromCaffe(res10_proto, res10_model)
            except Exception as e:
                print("AgeGenderWorker: could not load res10 face detector:", e)
                self.face_detector = None
        else:
            # if res10 not present, we still proceed but won't detect faces automatically
            self.face_detector = None

        # load gender/age networks (if present)
        gender_proto = os.path.join(model_dir, "deploy_gender.prototxt")
        gender_model = os.path.join(model_dir, "gender_net.caffemodel")
        age_proto = os.path.join(model_dir, "deploy_age.prototxt")
        age_model = os.path.join(model_dir, "age_net.caffemodel")

        try:
            self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model) if os.path.exists(gender_proto) and os.path.exists(gender_model) else None
            self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model) if os.path.exists(age_proto) and os.path.exists(age_model) else None
        except Exception as e:
            print("AgeGenderWorker: failed to load gender/age nets:", e)
            self.gender_net = None
            self.age_net = None

        self.GENDER_LIST = ['Male', 'Female']
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
                         '(38-43)', '(48-53)', '(60-100)']

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
            h, w = frame.shape[:2]

            # face detection
            faces = []
            if self.face_detector is not None:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104,117,123))
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                for i in range(detections.shape[2]):
                    conf = detections[0,0,i,2]
                    if conf > 0.6:
                        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                        x1,y1,x2,y2 = box.astype(int)
                        faces.append((x1,y1,x2,y2))
            # fallback: no faces found → skip age/gender or try center crop
            if not faces:
                # try simple heuristic: center square
                cx, cy = w//2, h//3
                size = min(w,h)//4
                x1 = max(0, cx-size); y1 = max(0, cy-size)
                x2 = min(w-1, cx+size); y2 = min(h-1, cy+size)
                faces = [(x1,y1,x2,y2)]

            for (x1,y1,x2,y2) in faces:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                if self.gender_net is not None:
                    face_blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), (78.4263377603,87.7689143744,114.895847746), swapRB=False)
                    try:
                        self.gender_net.setInput(face_blob)
                        gpred = self.gender_net.forward()
                        gender = self.GENDER_LIST[int(np.argmax(gpred[0]))] if gpred is not None else ""
                    except Exception:
                        gender = ""
                else:
                    gender = ""

                if self.age_net is not None:
                    try:
                        self.age_net.setInput(face_blob)
                        apred = self.age_net.forward()
                        age = self.AGE_LIST[int(np.argmax(apred[0]))] if apred is not None else ""
                    except Exception:
                        age = ""
                else:
                    age = ""

                label = f"{gender} {age}".strip()
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,180,0), 2)
                cv2.putText(frame, label, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,0), 2)

            pix = self.to_pixmap(frame)
            self.frame_ready.emit(pix)
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait(timeout=1000)
