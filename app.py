import json
import queue
from datetime import datetime
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pymongo import MongoClient

# ── CONFIG ───────────────────────────────────────
MODEL_PATH   = "fraud_model.pkl"
THRESHOLD    = 0.23
MONGO_URI    = "mongodb://localhost:27017/"
DB_NAME      = "fraud_detection"
COLL_PREDICT = "api_predictions"
COLL_ALERTS  = "fraud_alert_log"

FEATURES = ["amount", "hour", "weekday", "is_debit",
            "total_txns", "lat_rounded", "lon_rounded"]

# ── INIT ─────────────────────────────────────────
MODEL = joblib.load(MODEL_PATH)
db = MongoClient(MONGO_URI)[DB_NAME]

# simple pub-sub for Server-Sent Events
alert_queue: "queue.Queue[str]" = queue.Queue()
app = FastAPI(title="Fraud-Detection API", version="1.0.0")

# ── HELPERS ──────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["txn_time"] = pd.to_datetime(df["txn_time"])
    df["hour"] = df["txn_time"].dt.hour
    df["weekday"] = df["txn_time"].dt.dayofweek
    df["is_debit"] = (df["txn_type"].str.lower() == "debit").astype(int)
    df["lat_rounded"] = df["location_lat"].round(2)
    df["lon_rounded"] = df["location_long"].round(2)
    df["total_txns"] = df.groupby("user_id").cumcount() + 1
    return df

def push_alert(doc: dict):
    """Insert alert only if txn_id not already present."""
    db[COLL_ALERTS].update_one(
        {"txn_id": doc["txn_id"]},
        {"$setOnInsert": doc},
        upsert=True
    )
    alert_queue.put(json.dumps(doc, default=str))

# ── ROUTES ───────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "docs": "/docs", "events": "/events"}

@app.post("/predict_fraud")
def predict_fraud(txn: dict):
    try:
        df = prepare_features(pd.DataFrame([txn]))
        prob = float(MODEL.predict_proba(df[FEATURES])[0][1])
    except Exception as e:
        raise HTTPException(400, f"Bad input: {e}")

    is_fraud = prob >= THRESHOLD
    result = {
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud,
        "alert_message": "🚨 Fraud suspected!" if is_fraud else "✅ Transaction looks normal."
    }

    # Save prediction
    db[COLL_PREDICT].insert_one({**txn, **result})

    # Push alert only if fraud
    if is_fraud:
        alert_doc = {
            "alert_time": datetime.utcnow(),
            "user_id": txn["user_id"],
            "txn_id": txn["txn_id"],
            "amount": txn["amount"],
            "fraud_prob": prob,
            "location": [txn["location_lat"], txn["location_long"]],
            "read": False
        }
        push_alert(alert_doc)
        

    return result

@app.post("/predict_batch")
def predict_batch(txns: List[dict]):
    if not txns:
        raise HTTPException(400, "Empty batch")

    df_in = pd.DataFrame(txns)
    try:
        X = prepare_features(df_in)[FEATURES]
    except Exception as e:
        raise HTTPException(400, f"Feature prep error: {e}")

    probs = MODEL.predict_proba(X)[:, 1]
    df_in["fraud_prob"] = probs.round(4)
    df_in["is_fraud"] = probs >= THRESHOLD

    # Save predictions
    db[COLL_PREDICT].insert_many(df_in.to_dict("records"))

    # Save fraud alerts uniquely
    flagged = df_in[df_in.is_fraud]
    if not flagged.empty:
        flagged = flagged.assign(
            alert_time=datetime.utcnow(),
            read=False
        )
        for rec in flagged.to_dict("records"):
            db[COLL_ALERTS].update_one(
                {"txn_id": rec["txn_id"]},
                {"$setOnInsert": rec},
                upsert=True
            )
            alert_queue.put(json.dumps(rec, default=str))

    return df_in[["txn_id", "fraud_prob", "is_fraud"]].to_dict("records")




@app.get("/events")
def sse(request: Request):
    def event_stream():
        yield "retry: 3000\n\n"
        while True:
            if await_disconnect(request):
                break
            try:
                data = alert_queue.get(timeout=1)
                yield f"data: {data}\n\n"
            except queue.Empty:
                continue
    return StreamingResponse(event_stream(), media_type="text/event-stream")

def await_disconnect(req: Request) -> bool:
    try:
        return req.is_disconnected()
    except RuntimeError:
        return True
