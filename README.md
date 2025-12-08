
<!-- Banner -->
<p align="center">
  <img src="https://img.shields.io/badge/PedestriSense-AI%20Pedestrian%20System-00eaff?style=for-the-badge&logo=ai" alt="PedestriSense Banner"/>
</p>

<h1 align="center">🚶‍♂️ PedestriSense</h1>
<h3 align="center">A Robust AI‑Powered Pedestrian Detection & Analysis System</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyQt5-Desktop%20UI-41cd52?style=flat-square&logo=qt"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-ff0000?style=flat-square&logo=opencv"/>
  <img src="https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange?style=flat-square&logo=google"/>
</p>

---

## 🌟 Overview

**PedestriSense** is a full-stack AI desktop application built with **PyQt5** and powered by advanced computer vision models.  
It provides real-time:

- 🧍 **Person Detection** (MobileNet‑SSD)  
- 🦾 **Pose Estimation** (MediaPipe Pose)  
- 🧠 **Behavior Recognition** (Random Forest Classifier)  
- ⚠️ **Fall Detection** (Geometric + Pose-based rules)  
- 👤 **Age & Gender Prediction** (Pretrained CNN models)

All modules run **independently in QThreads**, ensuring a smooth UI and real-time performance.

---

## 🎨 UI Snapshot  
> *Modern Neon-Themed Dashboard (PyQt5)*  
Multiple live video panels update simultaneously with independent model outputs.

---

## 🚀 Features

### 🔍 **1. Real‑Time Person Detection**
- Uses MobileNet‑SSD (`deploy.prototxt` + `.caffemodel`)
- Accurate pedestrian bounding boxes  
- Ideal for tracking, counting, and monitoring

### 🩻 **2. Pose Detection**
- 33-point body landmark extraction  
- High accuracy via MediaPipe Pose  
- Smooth frame‑to‑frame skeleton rendering

### 🧠 **3. Behavior Recognition**
- Custom Random Forest model using pose vectors  
- Recognizes:  
  - Standing  
  - Sitting  
  - Waving (gesture)  
- Uses temporal smoothing (mode over last N frames)

### 🛑 **4. Fall Detection**
- Rule-based logic using angle, hip drop, & bounding box ratio  
- Alerts in UI with **FALL DETECTED** status

### 👤 **5. Age & Gender Classification**
- Pretrained CNN models (Caffe)  
- Detects:  
  - Male / Female  
  - Age ranges: 0–2, 4–6, 8–12, ..., 60–100

### 🖥️ **6. PyQt5 Frontend**
- Neon-themed modern UI  
- Four synchronized live video windows  
- Optimized using threads → **No UI freezing**

---

## 📁 Project Structure

```
PedestriSense/
│
├── pedestrisense_app/
│   ├── main.py
│   ├── ui/
│   │   └── style.py
│   └── workers/
│       ├── person_worker.py
│       ├── pose_worker.py
│       ├── fall_worker.py
│       └── age_gender_worker.py
│
├── person_detection/
├── pose_behavior/
├── age_gender_cnn/
├── fall_detection/
└── tracking/
```

---

## ⚙️ Installation

### 1️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe numpy joblib PyQt5
```

### 2️⃣ Run Application

```bash
cd pedestrisense_app
python main.py
```

---

## 📌 Technologies Used

| Component | Technology |
|----------|------------|
| **UI** | PyQt5 |
| **Person Detection** | OpenCV DNN – MobileNet‑SSD |
| **Pose Extraction** | MediaPipe Pose |
| **Behavior Recognition** | Random Forest Classifier |
| **Fall Detection** | Geometry + Pose Landmark Logic |
| **Age/Gender Classification** | CNN (Caffe Models) |

---

## 🔮 Future Improvements

- DeepSORT multi-person tracking  
- YOLOv8-based detection upgrade  
- Cloud dashboard + logging  
- Fall alert notifications  
- Model optimization (TFLite / ONNX)

---

## 👨‍💻 Author

**Riko**,  
AI Developer | Computer Vision Specialist  

---

<p align="center">
  <b>⭐ If you like this project, consider starring it on GitHub!</b>
</p>
