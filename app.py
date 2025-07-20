import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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

class TransactionsBatchIn(BaseModel):
    transactions: List[TransactionIn]

@app.on_event("startup")
async def startup_event():
    await transactions_collection.create_index("user_id")


@app.post("/transaction/", response_model=TransactionOut)
async def predict_and_store(txn: TransactionIn):
    df = pd.DataFrame([txn.dict()])
    features = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]
    try:
        prob = model.predict_proba(df[features])[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    is_fraud = bool(prob >= 0.5)

    txn_dict = txn.dict()
    txn_dict.update({
        "fraud_probability": float(prob),
        "is_fraud": is_fraud
    })

    result = await transactions_collection.insert_one(txn_dict)
    txn_dict["id"] = str(result.inserted_id)
    return txn_dict


@app.post("/transactions/batch/", response_model=List[TransactionOut])
async def batch_predict_and_store(batch: TransactionsBatchIn):
    df = pd.DataFrame([txn.dict() for txn in batch.transactions])
    features = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]

    try:
        probs = model.predict_proba(df[features])[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    docs = []
    for i, txn in enumerate(batch.transactions):
        doc = txn.dict()
        doc["fraud_probability"] = float(probs[i])
        doc["is_fraud"] = bool(probs[i] >= 0.5)
        docs.append(doc)

    result = await transactions_collection.insert_many(docs)

    for doc, inserted_id in zip(docs, result.inserted_ids):
        doc["id"] = str(inserted_id)

    return docs


@app.get("/transactions/", response_model=List[TransactionOut])
async def get_transactions(
    user_id: Optional[str] = None,
    is_fraud: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    query = {}
    if user_id:
        query["user_id"] = user_id
    if is_fraud is not None:
        query["is_fraud"] = is_fraud

    cursor = transactions_collection.find(query).skip(skip).limit(limit)
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        results.append(doc)

    return results
