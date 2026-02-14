import pandas as pd
import joblib
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

# Paths
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
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure 'creditcard.csv' is placed in 'backend/data/'.")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Basic preprocessing
    if not all(col in df.columns for col in FEATURES):
         raise ValueError(f"Dataset missing required columns. Expected: {FEATURES}")

    X = df[FEATURES]
    y = df["Class"]

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    # Using fewer trees for speed in demo, but enough for performance
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # --- EVALUATION ---
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Basic Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob))
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }

    # Curves (Downsampled for JSON size)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # Take every nth point to keep JSON small
    step = max(1, len(fpr) // 100)
    metrics["roc_curve"] = {
        "fpr": fpr[::step].tolist(),
        "tpr": tpr[::step].tolist()
    }
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    step_pr = max(1, len(precision) // 100)
    metrics["pr_curve"] = {
        "precision": precision[::step_pr].tolist(),
        "recall": recall[::step_pr].tolist()
    }

    # Threshold Analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_analysis = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        cm_t = confusion_matrix(y_test, y_pred_t)
        threshold_analysis.append({
            "threshold": float(round(t, 1)),
            "precision": float(p),
            "recall": float(r),
            "fp": int(cm_t[0, 1]), # False Positives (Cost of Friction)
            "fn": int(cm_t[1, 0]), # False Negatives (Cost of Fraud)
            "tp": int(cm_t[1, 1]),
            "tn": int(cm_t[0, 0])
        })
    metrics["threshold_analysis"] = threshold_analysis

    # Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    metrics["feature_importance"] = [
        {"feature": FEATURES[indices[i]], "importance": float(importances[indices[i]])}
        for i in range(top_n)
    ]
    
    # EDA Stats (Simple)
    metrics["eda"] = {
        "total_transactions": len(df),
        "fraud_count": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "avg_amount_legit": float(df[df["Class"] == 0]["Amount"].mean()),
        "avg_amount_fraud": float(df[df["Class"] == 1]["Amount"].mean())
    }

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    return pipeline

if __name__ == "__main__":
    train_model()
