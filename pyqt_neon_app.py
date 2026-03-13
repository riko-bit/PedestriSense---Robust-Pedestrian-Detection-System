# pedestrisense_app.py
import sys
import time
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF

from main_integration import run_inference


FPS_TARGET = 30
FRAME_INTERVAL = 1.0 / FPS_TARGET


# ============================================================
#  VIDEO THREAD (reads camera + runs the integrated pipeline)
# ============================================================
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray, dict)  # frame + inference results

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera failed to start")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Run detection pipeline
            results = run_inference(frame)
            frame_out = results.get("frame", frame)

            self.frame_signal.emit(frame_out, results)

            time.sleep(FRAME_INTERVAL)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ============================================================
#  MAIN WINDOW CLASS
# ============================================================
class PedestriSenseApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PedestriSense — Desktop AI System")
        self.setGeometry(50, 30, 1350, 800)

        self._fall_alert_shown_for = set()

        self._setup_ui()
        self._setup_video_thread()

    # --------------------------------------------------------
    # UI SETUP
    # --------------------------------------------------------
    def _setup_ui(self):
        self.setStyleSheet(self._stylesheet())

        # ---------- APP TITLE ----------
        self.title_label = QLabel("PedestriSense — Robust Pedestrian Tracking System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Orbitron", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #00eaff; margin-bottom: 10px;")

        # ---------- VIDEO VIEW ----------
        self.video_frame = QLabel()
        self.video_frame.setFixedSize(960, 600)
        self.video_frame.setObjectName("videoBox")

        # Neon animation
        self._glow_strength = 12
        self._glow_timer = QTimer()
        self._glow_timer.timeout.connect(self._animate_glow)
        self._glow_timer.start(60)

        # ---------- OVERLAY TITLE ----------
        self.overlay_label = QLabel("LIVE DETECTION FEED")
        self.overlay_label.setStyleSheet("color:#00f6ff; font-size:18px; font-weight:bold;")
        self.overlay_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Stack overlay inside video box
        self.overlay_label.setParent(self.video_frame)
        self.overlay_label.move(370, 10)

        # ---------- INFO PANELS (RIGHT SIDE) ----------
        self.person_info = self._create_panel(
            "Person Info",
            ["ID: --", "Confidence: --", "Track Duration: --s"],
            color="#00ffd1"
        )

        self.pose_info = self._create_panel(
            "Pose Skeleton",
            ["Keypoints: --"],
            color="#ff00d6"
        )

        self.demo_info = self._create_panel(
            "Demographics",
            ["Age: --", "Gender: --"],
            color="#40bfff"
        )

        self.activity_info = self._create_panel(
            "Activity Recognition",
            ["Activity: --", "Fall Risk: Monitoring"],
            color="#ff9e00"
        )

        # ---------- LAYOUT ----------
        right_side = QVBoxLayout()
        right_side.addWidget(self.person_info)
        right_side.addWidget(self.pose_info)
        right_side.addWidget(self.demo_info)
        right_side.addWidget(self.activity_info)
        right_side.addStretch()

        main_layout = QVBoxLayout()
        row = QHBoxLayout()

        row.addWidget(self.video_frame)
        row.addLayout(right_side)

        main_layout.addWidget(self.title_label)
        main_layout.addLayout(row)

        self.setLayout(main_layout)

    def _setup_video_thread(self):
        self.thread = VideoThread()
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    # --------------------------------------------------------
    # PANEL CREATOR
    # --------------------------------------------------------
    def _create_panel(self, title, body_lines, color="#00ffe7"):
        frame = QFrame()
        frame.setObjectName("panel")

        v = QVBoxLayout()
        t = QLabel(title)
        t.setFont(QFont("Orbitron", 13, QFont.Bold))
        t.setStyleSheet(f"color:{color};")

        b = QLabel("\n".join(body_lines))
        b.setFont(QFont("Inter", 11))
        b.setStyleSheet("color:#d6e2f0;")

        frame.body = b  # store reference

        v.addWidget(t)
        v.addWidget(b)
        frame.setLayout(v)
        return frame

    # --------------------------------------------------------
    # UPDATE LIVE FEED
    # --------------------------------------------------------
    def update_frame(self, frame, results):
        # Draw bounding boxes directly in Qt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        persons = results.get("persons", [])
        h, w, _ = rgb.shape

        painter_image = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        painter = QPainter()
        painter.begin(painter_image)

        # Draw detection boxes + labels
        pen = QPen(QColor("#00faff"), 3)
        painter.setPen(pen)

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            text = f"ID {p['id']} | {p['confidence']:.2f}"
            painter.setPen(QColor("#00faff"))
            painter.setFont(QFont("Inter", 11, QFont.Bold))
            painter.drawText(x1, y1 - 6, text)

        painter.end()

        pixmap = QPixmap.fromImage(painter_image).scaled(
            self.video_frame.width(), self.video_frame.height(), Qt.KeepAspectRatio
        )
        self.video_frame.setPixmap(pixmap)

        self.update_panels(results)

    # --------------------------------------------------------
    # UPDATE SIDE PANELS
    # --------------------------------------------------------
    def update_panels(self, results):
        persons = results.get("persons", [])
        if not persons:
            return

        p = persons[0]

        # Update Person Info
        text = (
            f"ID: {p['id']}\n"
            f"Confidence: {p['confidence']:.2f}\n"
            f"Track Duration: --s"
        )
        self.person_info.body.setText(text)

        # Pose
        self.pose_info.body.setText(f"Keypoints: {len(p.get('keypoints', []))}")

        # Demographics
        self.demo_info.body.setText(
            f"Age: {p.get('age','--')}\nGender: {p.get('gender','--')}"
        )

        # Activity + Fall
        fall_flag = p.get("fall", False)
        activity = p.get("activity", "Unknown")

        if fall_flag:
            self.activity_info.body.setText("Activity: FALLEN\nFall Risk: ⚠️ ALERT")
            self._show_fall_popup(p["id"])
        else:
            self.activity_info.body.setText(f"Activity: {activity}\nFall Risk: Monitoring")

    # --------------------------------------------------------
    # FALL POPUP
    # --------------------------------------------------------
    def _show_fall_popup(self, pid):
        if pid in self._fall_alert_shown_for:
            return

        self._fall_alert_shown_for.add(pid)

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("⚠️ Fall Detected!")
        msg.setText(f"A fall was detected for ID: {pid}")
        msg.exec_()

    # --------------------------------------------------------
    # ANIMATED NEON BORDER AROUND CAMERA FEED
    # --------------------------------------------------------
    def _animate_glow(self):
        self._glow_strength += 1
        if self._glow_strength > 25:
            self._glow_strength = 12

        self.video_frame.setStyleSheet(
            f"""
            QLabel#videoBox {{
                border: 3px solid rgba(0,255,200,0.4);
                border-radius: 14px;
                box-shadow: 0px 0px {self._glow_strength}px #00ffe7;
                background-color: #050a0d;
            }}
            """
        )

    # --------------------------------------------------------
    # STYLESHEET FOR FULL APP
    # --------------------------------------------------------
    def _stylesheet(self):
        return """
        QWidget {
            background-color: #04070b;
        }
        QFrame#panel {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(0,255,200,0.1);
            border-radius: 12px;
            padding: 12px;
        }
        QLabel {
            color: #cfe8ff;
        }
        """

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


# ============================================================
#  MAIN
# ============================================================
def main():
    app = QApplication(sys.argv)
    window = PedestriSenseApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()