from pydantic import BaseModel, Field
from typing import Optional

# Base Feature Schema (Raw Features)
class FeaturesInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Payment Simulation Schema
class PaymentRequest(BaseModel):
    card_number: str = Field(..., description="16-digit card number")
    expiry_date: str = Field(..., description="MM/YY")
    cvv: str = Field(..., description="3-digit CVV")
    amount: float = Field(..., gt=0, description="Transaction amount")

# Response Schema
class PredictionResponse(BaseModel):
    label: str
    probability: float
    threshold: float = 0.5
    model: str
    notes: str
    processing_time_ms: float
    
class BatchPredictionSummary(BaseModel):
    total_transactions: int
    fraud_count: int
    fraud_percentage: float
    processing_time_ms: float
    # Maybe return list of results? For simplicity, just summary for now as requested.
