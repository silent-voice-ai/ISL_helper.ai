import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from PIL import Image

# Load Model & Setup MediaPipe
model = joblib.load('isl_model.pkl')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("isl_helper.ai - ISL Translation")

# Session state initialization
if 'detected_words' not in st.session_state:
    st.session_state.detected_words = []
if 'gemini_sentence' not in st.session_state:
    st.session_state.gemini_sentence = "Snap a photo of your sign below to start."
if 'last_processed_file_id' not in st.session_state:
    st.session_state.last_processed_file_id = None

st.markdown("### Show your ISL sign")
image_file = st.camera_input("Take Photo")

if image_file is not None:
    if image_file.file_id != st.session_state.last_processed_file_id:
        st.session_state.last_processed_file_id = image_file.file_id
        
        bytes_data = image_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5) as hands:
            results = hands.process(rgb)
            current_prediction = "No hand detected"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z

                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                    while len(data_aux) < 126:
                        data_aux.append(0.0)

                    features = np.array(data_aux[:126]).reshape(1, -1)
                    prediction = model.predict(features)[0]
                    current_prediction = str(prediction)
                    
                    st.session_state.detected_words.append(current_prediction)

st.write(f"**Detected Glosses:** {st.session_state.detected_words}")

col1, col2 = st.columns(2)
with col1:
    if st.button("✨ Generate AI Sentence"):
        if st.session_state.detected_words:
            with st.spinner("Formatting sentence..."):
                # Smart local sentence constructor (no API key or payment needed)
                words = [w.capitalize() for w in st.session_state.detected_words]
                if len(words) >= 3 and words[0] == "Hello" and words[1] == "I":
                    # Custom polite phrasing for common demo phrases
                    st.session_state.gemini_sentence = f"Hello, I am {words[2].lower()}."
                else:
                    st.session_state.gemini_sentence = " ".join(words) + "."
        else:
            st.session_state.gemini_sentence = "No signs captured yet."

with col2:
    if st.button("Clear Buffer"):
        st.session_state.detected_words = []
        st.session_state.gemini_sentence = "Buffer cleared."
        st.session_state.last_processed_file_id = None

st.success(f"**AI Formatted Sentence:** {st.session_state.gemini_sentence}")