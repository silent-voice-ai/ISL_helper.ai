import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATASET_FILE = "isl_dataset.csv"
MODEL_FILE = "isl_model.pkl"

def train_from_single_sample(sign_name: str, single_sample_features: list):
    """
    Takes a single 126-feature landmark sample, generates 1,000 synthetic 
    variations using Gaussian noise, appends them securely via Pandas, 
    and trains the Random Forest classifier.
    """
    synthetic_rows = []
    features_array = np.array(single_sample_features)
    
    # 1. Data Augmentation (Gaussian Noise)
    for _ in range(1000):
        noise = np.random.normal(0, 0.03, size=len(features_array))
        augmented_features = features_array + noise
        row = list(augmented_features) + [sign_name.upper()]
        synthetic_rows.append(row)

    feature_cols = [f"feat_{i}" for i in range(len(features_array))]
    new_df = pd.DataFrame(synthetic_rows, columns=feature_cols + ["label"])

    # 2. Accumulative Pandas Pipeline
    if os.path.exists(DATASET_FILE):
        existing_df = pd.read_csv(DATASET_FILE)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    updated_df.to_csv(DATASET_FILE, index=False)
    print(f"Dataset updated. Total rows: {len(updated_df)}")

    # 3. Model Training
    X = updated_df.drop(columns=["label"])
    y = updated_df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Save Model
    joblib.dump(model, MODEL_FILE)
    print(f"SUCCESS: Model trained on {y.nunique()} total signs and saved as {MODEL_FILE}!")
