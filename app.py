import os
import pickle
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# MongoDB setup
MONGO_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URL)
db = client.fraud_detection
transactions_collection = db.transactions

# Load model
MODEL_PATH = "fraud_rf_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found: fraud_rf_model.pkl")



model = joblib.load(MODEL_PATH)


# FastAPI app
app = FastAPI(title="Fraud Detection API with MongoDB")

# Pydantic schemas
class TransactionIn(BaseModel):
    user_id: str
    amount: float
    haversine_km: float
    new_device: int
    login_txn_gap_min: float
    timestamp: str

class TransactionOut(TransactionIn):
    id: Optional[str]
    fraud_probability: float
    is_fraud: bool

@app.on_event("startup")
async def startup_event():
    # Optional: create indexes if needed
    await transactions_collection.create_index("user_id")

@app.post("/transaction/", response_model=TransactionOut)
async def predict_and_store(txn: TransactionIn):
    # Prepare data for prediction
    df = pd.DataFrame([txn.dict()])
    features = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]

    try:
        prob = model.predict_proba(df[features])[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    is_fraud = bool(prob >= 0.5)


    # Prepare document to save
    txn_dict = txn.dict()
    txn_dict.update({
    "fraud_probability": float(prob),
    "is_fraud": is_fraud
    })


    result = await transactions_collection.insert_one(txn_dict)
    txn_dict["id"] = str(result.inserted_id)
    return txn_dict

@app.get("/transactions/", response_model=List[TransactionOut])
async def get_all_transactions():
    cursor = transactions_collection.find({})
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        results.append(doc)
    return results
