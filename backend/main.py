from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.schemas import PaymentRequest, PredictionResponse, FeaturesInput, BatchPredictionSummary
from backend.model_utils import load_model, train_model, FEATURES, MODEL_PATH, DATA_PATH, METRICS_PATH
import pandas as pd
import numpy as np
import time
import os
import random
import io
import json

app = FastAPI(title="Credit Card Fraud Detection & Analytics System")

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
dataset_df = None 

@app.on_event("startup")
def startup_event():
    global model, dataset_df
    model = load_model()
    
    # Check if metrics exist, if not, might need to wait for training or manual trigger
    if not os.path.exists(METRICS_PATH):
        print("Metrics file not found. Recommendation: Run training to generate analytics.")
        
    if model is None:
        print("Model not found. Attempting to train...")
        try:
            model = train_model()
        except FileNotFoundError:
            print("Creditcard.csv not found. Model training skipped. Predictions will fail.")
        except Exception as e:
            print(f"Training failed: {e}")
            
    if os.path.exists(DATA_PATH):
        try:
            print("Loading dataset for sampling...")
            dataset_df = pd.read_csv(DATA_PATH)
            print("Dataset loaded.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "data_loaded": dataset_df is not None}

# --- Analytics Endpoints ---

@app.get("/analytics/metrics")
def get_model_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics not found. Please train the model first.")
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

# --- Payment Simulation Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_payment(payment: PaymentRequest):
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # Feature Generation
    is_demo_fraud = payment.amount == 9999 or payment.amount > 5000
    
    current_time_sim = time.time() % 172800 
    generated_features = { "Time": current_time_sim, "Amount": payment.amount }
    
    if is_demo_fraud:
        v_values = np.random.uniform(-10, 10, 28) 
    else:
        v_values = np.random.standard_normal(28)

    for i, val in enumerate(v_values, 1):
        generated_features[f"V{i}"] = val
        
    df = pd.DataFrame([generated_features])
    df = df[FEATURES]

    try:
        prob = model.predict_proba(df)[0][1] 
        prediction = int(model.predict(df)[0]) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    if is_demo_fraud and prediction == 0:
        prediction = 1
        prob = random.uniform(0.85, 0.99)
        notes = "SIMULATION: Suspicious amount detected (Demo Trigger)."
    elif is_demo_fraud:
        notes = "Model detected anomalies matching fraud patterns."
    elif prediction == 1:
        notes = "Model flagged this transaction based on feature analysis."
    else:
        notes = "Transaction appears normal."
    
    end_time = time.time()
    
    return {
        "label": "Transaction Flagged" if prediction == 1 else "Payment Approved",
        "probability": float(prob),
        "threshold": 0.5,
        "model": "RandomForestClassifier",
        "notes": notes,
        "processing_time_ms": (end_time - start_time) * 1000
    }

# --- ML Debugger Endpoints ---

@app.get("/random-sample")
def get_random_sample():
    if dataset_df is None:
        raise HTTPException(status_code=503, detail="Dataset not available for sampling.")
    
    row = dataset_df.sample(n=1).iloc[0]
    data = row.to_dict()
    data = {k: float(v) for k, v in data.items()}
    
    return data

@app.post("/predict-features", response_model=PredictionResponse)
def predict_features(features: FeaturesInput):
    start_time = time.time()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    data = features.dict()
    df = pd.DataFrame([data])
    df = df[FEATURES]
    
    try:
        prob = model.predict_proba(df)[0][1]
        prediction = int(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    end_time = time.time()
    
    return {
        "label": "Fraudulent" if prediction == 1 else "Legitimate",
        "probability": float(prob),
        "threshold": 0.5,
        "model": "RandomForestClassifier",
        "notes": "Based on raw feature analysis.",
        "processing_time_ms": (end_time - start_time) * 1000
    }

@app.post("/predict-batch", response_model=BatchPredictionSummary)
async def predict_batch(file: UploadFile = File(...)):
    start_time = time.time()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files allowed.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        missing_cols = [col for col in FEATURES if col not in df.columns]
        if missing_cols:
             raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
             
        X = df[FEATURES]
        predictions = model.predict(X)
        
        fraud_count = np.sum(predictions)
        total = len(predictions)
        percentage = (fraud_count / total) * 100 if total > 0 else 0
        
        end_time = time.time()
        
        return {
            "total_transactions": int(total),
            "fraud_count": int(fraud_count),
            "fraud_percentage": float(percentage),
            "processing_time_ms": (end_time - start_time) * 1000
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

frontend_path = "frontend"
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
