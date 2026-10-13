import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
import time

DATASET_FILE = "isl_dataset.csv"
MODEL_FILE = "isl_model.pkl"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def collect_sign_data():
    print("=== ISL Helper Data Collector (Visible Landmarks Prep) ===")
    sign_name = input("Enter the name of the sign you want to record (e.g. HELP): ").strip()
    if not sign_name:
        print("Error: Sign name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    
    # Phase 1: 15-Second preparation countdown (with live skeleton tracking visible)
    countdown_start = time.time()
    countdown_seconds = 15

    print(f"\nGet ready! You have 15 seconds to position yourself for '{sign_name.upper()}'...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw hand landmarks live during countdown
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        elapsed = time.time() - countdown_start
        remaining_countdown = int(countdown_seconds - elapsed)

        if remaining_countdown >= 0:
            cv2.putText(frame, f"GET READY: {remaining_countdown}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            cv2.putText(frame, f"Sign: {sign_name.upper()}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Capture Sign", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        else:
            break

    # Phase 2: Active Recording (5 seconds of collecting real frames)
    print(f"Recording real samples for '{sign_name.upper()}' now!")
    raw_samples = []
    recording_duration = 5  
    record_start = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        elapsed = time.time() - record_start
        remaining_recording = int(recording_duration - elapsed)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- WRIST RELATIVE NORMALIZATION ---
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z

                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.extend([
                        lm.x - base_x, 
                        lm.y - base_y, 
                        lm.z - base_z
                    ])

                while len(data_aux) < 126:
                    data_aux.append(0.0)
                
                raw_samples.append(data_aux[:126])

        if remaining_recording >= 0:
            cv2.putText(frame, f"RECORDING... {remaining_recording}s left", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, f"Frames captured: {len(raw_samples)}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "PROCESSING...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Capture Sign", frame)
            cv2.waitKey(1000)
            break

        cv2.imshow("Capture Sign", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(raw_samples) < 5:
        print("ERROR: Not enough hand frames detected. Try again and keep your hand steady.")
        return

    # Phase 3: Exact 1,000 Sample Balancing via Augmentation
    print(f"Expanding captured frames to exactly 1,000 samples for '{sign_name.upper()}'...")
    balanced_samples = []
    target_count = 1000
    
    while len(balanced_samples) < target_count:
        for sample in raw_samples:
            if len(balanced_samples) >= target_count:
                break
            noise = np.random.normal(0, 0.005, size=len(sample))
            augmented_sample = np.array(sample) + noise
            balanced_samples.append(list(augmented_sample) + [sign_name.upper()])

    feature_cols = [f"feat_{i}" for i in range(126)]
    new_df = pd.DataFrame(balanced_samples, columns=feature_cols + ["label"])

    # Phase 4: Clean up old entries for this specific sign and append new ones
    if os.path.exists(DATASET_FILE):
        existing_df = pd.read_csv(DATASET_FILE)
        existing_df = existing_df[existing_df['label'] != sign_name.upper()]
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    updated_df.to_csv(DATASET_FILE, index=False)

    # Phase 5: Train Random Forest Model immediately
    X = updated_df.drop(columns=["label"])
    y = updated_df["label"]

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    print(f"\nSUCCESS: '{sign_name.upper()}' now has exactly 1,000 balanced samples! Total classes in model: {y.nunique()}")

if __name__ == "__main__":
    collect_sign_data()