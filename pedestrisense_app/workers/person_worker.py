# pedestrisense_app/workers/person_worker.py
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class PersonWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.running = True
        self.net = None
        # Models (resolved relative to project root)
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        proto = os.path.join(base, "person_detection", "deploy.prototxt")
        model = os.path.join(base, "person_detection", "mobilenet_iter_73000.caffemodel")
        if os.path.exists(proto) and os.path.exists(model):
            try:
                self.net = cv2.dnn.readNetFromCaffe(proto, model)
            except Exception as e:
                print("PersonWorker: failed to load net:", e)
                self.net = None
        else:
            print("PersonWorker: person detection model missing; panel will show raw feed.")

    @pyqtSlot(object)
    def receive_frame(self, frame):
        # this slot will be called in main thread — store frame for worker thread to use
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

            if self.net is not None:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()
                for i in range(detections.shape[2]):
                    conf = float(detections[0, 0, i, 2])
                    cls = int(detections[0, 0, i, 1])
                    if conf > 0.5 and cls == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID 1", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # Emit frame pixmap
            pix = self.to_pixmap(frame)
            self.frame_ready.emit(pix)
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait(timeout=1000)
