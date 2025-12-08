import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class CameraWorker(QThread):
    new_frame = pyqtSignal(object)  # emits raw BGR frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            self.new_frame.emit(frame)
