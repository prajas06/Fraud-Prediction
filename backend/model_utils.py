import pandas as pd
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def train_model():
    # This function is now largely superseded by backend.train_model.py
    # But kept here for legacy compatibility if called directly
    # Ideally should import from there to avoid code duplication
    # For now, let's keep it simple and just ensure paths overlap correctly
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    if not all(col in df.columns for col in FEATURES):
         raise ValueError(f"Dataset missing required columns. Expected: {FEATURES}")

    X = df[FEATURES]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print(f"Model trained.")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return pipeline

if __name__ == "__main__":
    train_model()
