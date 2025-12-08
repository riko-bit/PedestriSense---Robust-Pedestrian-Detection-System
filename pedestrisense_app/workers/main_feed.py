# pedestrisense_app/workers/main_feed.py
import time
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class MainFeedWorker(QThread):
    frame_available = pyqtSignal(object)  # emits BGR numpy frame

    def __init__(self, device=0, target_fps=30):
        super().__init__()
        self.device = device
        self.target_fps = target_fps
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if not cap.isOpened():
            print("ERROR: Camera not opened.")
            return

        period = 1.0 / float(self.target_fps)
        while self.running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Emit the frame to all listeners
            self.frame_available.emit(frame)

            elapsed = time.time() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(timeout=1000)