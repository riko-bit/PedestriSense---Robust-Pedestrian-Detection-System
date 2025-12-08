# pedestrisense_app/main.py
import sys
import os
import cv2

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from ui.style import APP_STYLE

# Workers
from workers.main_feed import MainFeedWorker
from workers.person_worker import PersonWorker
from workers.pose_worker import PoseWorker
from workers.fall_worker import FallWorker
from workers.age_gender_worker import AgeGenderWorker

class PedestriSenseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pedestrisense – Robust Pedestrian Detection System")
        self.setGeometry(50, 50, 1280, 800)
        self.setStyleSheet(APP_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Title
        title = QLabel("PedestriSense")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Robust Pedestrian Detection System")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # Main camera feed area
        main_cam_label = QLabel("Main Camera Feed (Raw)")
        main_cam_label.setAlignment(Qt.AlignCenter)
        main_cam_label.setFixedHeight(360)
        main_cam_label.setStyleSheet("border: 2px solid rgba(124,243,255,0.25); border-radius: 8px;")
        main_cam_label.setScaledContents(True)
        self.main_cam_label = main_cam_label

        main_layout.addWidget(main_cam_label, alignment=Qt.AlignHCenter)

        # Bottom panels grid
        grid = QGridLayout()
        grid.setSpacing(18)

        # Create placeholders for 4 panels
        self.pose_label = QLabel("Pose Detection")
        self.pose_label.setFixedSize(600, 240)
        self.pose_label.setAlignment(Qt.AlignCenter)
        self.pose_label.setStyleSheet("border: 2px solid rgba(180,180,180,0.06);")
        self.pose_label.setScaledContents(True)

        self.fall_label = QLabel("Fall Detection")
        self.fall_label.setFixedSize(600, 240)
        self.fall_label.setAlignment(Qt.AlignCenter)
        self.fall_label.setScaledContents(True)

        self.age_label = QLabel("Age / Gender")
        self.age_label.setFixedSize(600, 240)
        self.age_label.setAlignment(Qt.AlignCenter)
        self.age_label.setScaledContents(True)

        self.person_label = QLabel("Person Detection")
        self.person_label.setFixedSize(300, 160)
        self.person_label.setAlignment(Qt.AlignCenter)
        self.person_label.setScaledContents(True)

        # Layout like your mockup: three wide panels + person small box on the right
        grid.addWidget(self.pose_label, 0, 0)
        grid.addWidget(self.fall_label, 0, 1)
        grid.addWidget(self.age_label, 0, 2)
        grid.addWidget(self.person_label, 1, 1, alignment=Qt.AlignCenter)

        main_layout.addLayout(grid)

        # Initialize workers
        self.main_worker = MainFeedWorker(device=0, target_fps=30)
        self.person_worker = PersonWorker()
        self.pose_worker = PoseWorker()
        self.fall_worker = FallWorker()
        self.age_worker = AgeGenderWorker()

        # Connect main feed signal to UI main camera
        self.main_worker.frame_available.connect(self.on_main_frame)

        # Connect main feed to workers (they have receive_frame slots)
        self.main_worker.frame_available.connect(self.person_worker.receive_frame)
        self.main_worker.frame_available.connect(self.pose_worker.receive_frame)
        self.main_worker.frame_available.connect(self.fall_worker.receive_frame)
        self.main_worker.frame_available.connect(self.age_worker.receive_frame)

        # Connect worker output to GUI panels
        self.person_worker.frame_ready.connect(self.update_person_panel)
        self.pose_worker.frame_ready.connect(self.update_pose_panel)
        self.fall_worker.frame_ready.connect(self.update_fall_panel)
        self.age_worker.frame_ready.connect(self.update_age_panel)

        # Start worker threads (order: start workers first so they can receive frames)
        self.person_worker.start()
        self.pose_worker.start()
        self.fall_worker.start()
        self.age_worker.start()
        self.main_worker.start()

    def on_main_frame(self, frame):
        # Show raw camera in main panel (no overlays)
        pix = self._to_pixmap(frame)
        self.main_cam_label.setPixmap(pix.scaled(self.main_cam_label.width(), self.main_cam_label.height(), Qt.KeepAspectRatio))

    def update_person_panel(self, pix):
        self.person_label.setPixmap(pix.scaled(self.person_label.width(), self.person_label.height(), Qt.KeepAspectRatio))

    def update_pose_panel(self, pix):
        self.pose_label.setPixmap(pix.scaled(self.pose_label.width(), self.pose_label.height(), Qt.KeepAspectRatio))

    def update_fall_panel(self, pix):
        self.fall_label.setPixmap(pix.scaled(self.fall_label.width(), self.fall_label.height(), Qt.KeepAspectRatio))

    def update_age_panel(self, pix):
        self.age_label.setPixmap(pix.scaled(self.age_label.width(), self.age_label.height(), Qt.KeepAspectRatio))

    @staticmethod
    def _to_pixmap(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        from PyQt5.QtGui import QImage, QPixmap
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def closeEvent(self, event):
        # stop threads cleanly
        self.main_worker.stop()
        self.person_worker.stop()
        self.pose_worker.stop()
        self.fall_worker.stop()
        self.age_worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PedestriSenseApp()
    win.show()
    sys.exit(app.exec_())
