# ISL_Helper.ai 🤟

An edge-ready, real-time Indian Sign Language (ISL) translator engineered for low latency, low resource overhead, and offline availability on standard consumer hardware.

### 🎥 Project Video (90 sec)
[YouTube Link - Add your unlisted link here]

---

## 📌 Features
- **Snapshot-Based Gesture Capture:** Avoids continuous stream frame latency and webcam hardware locks.
- **Precision ML Classification:** Extracts 21 spatial hand landmarks per hand (up to 126 features) and normalizes coordinates relative to the wrist origin.
- **Offline Sentence Engine:** Formats predicted sign gloss sequences into natural English sentences with zero external API dependencies.
- **Lightweight Architecture:** Runs seamlessly on standard laptops without requiring a dedicated GPU.

## 🛠 Tech Stack
- **Frontend:** Streamlit
- **Computer Vision:** OpenCV, MediaPipe Hands
- **Machine Learning:** Random Forest Classifier
- **Language:** Python 3.x

## 📐 Machine Learning Pipeline
`Camera Snapshot -> MediaPipe (21 Landmarks) -> Wrist Normalization -> 126-Feature Array -> Random Forest -> Sign Gloss -> Offline Sentence Engine -> English Sentence`

## 📦 Dataset & Model Note
Full `isl_dataset.csv` (~120MB) and `isl_model.pkl` (~80MB) are not uploaded to GitHub due to GitHub's 100MB file size limit.
The repository contains full code to regenerate them.

## 🚀 How to Run
1.  Install dependencies
    ```bash
    pip install -r requirements.txt
