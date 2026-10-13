# isl_helper.ai 🤟

An edge-ready, real-time Indian Sign Language (ISL) translator engineered for low latency, low resource overhead, and offline availability on standard consumer hardware.

---

## 📌 Features

- **Snapshot-Based Gesture Capture:** Avoids continuous stream frame latency and webcam hardware locks.
- **Precision ML Classification:** Extracts 21 spatial hand landmarks per hand (up to 126 features) and normalizes coordinates relative to the wrist origin.
- **Offline Sentence Engine:** Formats predicted sign gloss sequences into natural English sentences with zero external API dependencies or cloud latency.
- **Lightweight Architecture:** Runs seamlessly on standard laptops without requiring a dedicated GPU.

---

## 🛠 Tech Stack

- **Frontend / Interface:** Streamlit
- **Computer Vision:** OpenCV, MediaPipe Hands (21 3D Landmarks)
- **Machine Learning:** Random Forest Classifier
- **Programming Language:** Python 3.x

---

## 📐 Machine Learning Pipeline

[ Camera Snapshot ] ➔ [ MediaPipe Hand Tracking (21 Landmarks/Hand) ]
│
▼
[ Wrist Origin Normalization ]
│
▼
[ 126-Feature Spatial Landmark Array ]
│
▼
[ Random Forest Classifier Model ]
│
▼
[ Isolated Sign Gloss Output ]
│
▼
[ Localized Offline Sentence Engine ]
│
▼
[ Formatted English Sentence ]

## How to run 
'''bash
pip install -r requirements.txt 
streamlit run app.py
