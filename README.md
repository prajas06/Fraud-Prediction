# Fraud Risk Analytics System

A comprehensive Fraud Detection System demonstrating end-to-end Machine Learning capabilities: form Real-time Scoring to Model Evaluation and Business Impact Analysis.

## Features

### 1. Real-Time Risk Scoring (Payment Simulation)
- Simulate a payment gateway experience.
- Input Card Details & Amount to get an instant Fraud Probability Score.
- **Demo Trigger**: Enter Amount > $5,000 to simulate a high-risk transaction.

### 2. Analytics Dashboard
- **Model Performance**: Visualize ROC Curve, Precision-Recall, and Confusion Matrix.
- **Interactive Threshold Tuning**: Adjust the decision threshold to see the trade-off between **Recall** (catching fraud) and **False Positives** (customer friction).
- **Feature Importance**: See which transaction features drive the model's decisions.

### 3. ML Debugger
- **Batch Testing**: Upload a CSV to process thousands of transactions instantly.
- **Random Sampling**: Test the model against real formatted data from the dataset.

## Data Science Context

### The Challenge
- **Imbalanced Dataset**: The dataset contains only 0.17% fraud cases. Traditional accuracy is misleading (a dummy model predicting "Legitimate" for everything would be 99.83% accurate).
- **Metric Focus**: We prioritize **Recall** (catching as much fraud as possible) and **AUC-ROC** (separability) over simple Accuracy.

### The Solution
- **Model**: RandomForestClassifier (Robust to outliers and non-linear patterns).
- **Preprocessing**: StandardScaler for feature scaling.
- **Evaluation**: The dashboard visualizes the "Business Cost" of different thresholds.

## Setup Instructions

### Prerequisites
- **Python 3.9+**
- **Data**: You must have `creditcard.csv` (the Kaggle dataset).

### Step 1: Prepare Data
Place your `creditcard.csv` file into the `backend/data/` directory.
```
scratch/
├── backend/
│   ├── data/
│   │   └── creditcard.csv  <-- Place here
```

### Step 2: Install Dependencies
Open your terminal in the project root (`scratch/`) and run:
```bash
pip install -r backend/requirements.txt
```

### Step 3: Train Model & Generate Analytics
Train the model and generate the `metrics.json` file for the dashboard:
```bash
python -m backend.train_model
```
*Note: This may take 30-60 seconds depending on your CPU.*

### Step 4: Start the Application
Launch the FastAPI server:
```bash
uvicorn backend.main:app --reload
```

### Step 5: Access the Dashboard
Open your browser and visit:
 **[http://localhost:8000](http://localhost:8000)**

## Project Structure
- `backend/`: FastAPI app, ML training logic, and data handling.
- `frontend/`: Dashboard UI (HTML/CSS/JS/Chart.js).
- `models/`: Serialized model and pre-computed analytics.

## Disclaimer
This project uses the Kaggle Credit Card Fraud Detection dataset for educational purposes.
