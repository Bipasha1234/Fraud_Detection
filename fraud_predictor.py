import random
from datetime import datetime

import pymongo

# --- Setup MongoDB ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]
prediction_col = db["api_predictions"]
alert_col = db["fraud_alert_log"]
high_risk_col = db["high_risk_txns"]

# --- Simulate a Transaction ---
txn = {
    "user_id": 8,
    "txn_id": 5071,
    "txn_time": datetime.now().isoformat(),
    "amount": 22141.66,
    "txn_type": "payment",
    "location_lat": 27.69845,
    "location_long": 85.30165,
    "device_id": "dev858",
    "phone": "+9779814459951"
}

# --- Dummy model prediction (simulate) ---
fraud_probability = round(random.uniform(0.1, 0.2), 4)
is_fraud = fraud_probability > 0.12

# --- Store in api_predictions ---
api_doc = {
    **txn,
    "fraud_probability": fraud_probability,
    "is_fraud": is_fraud,
    "alert_message": "⚠️ Suspicious transaction detected!" if is_fraud else "✅ Transaction looks normal."
}
prediction_col.insert_one(api_doc)

# --- Print feedback for dev log ---
print("Fraud prob =", fraud_probability)

# --- Log if Fraud Detected ---
if is_fraud:
    message_admin = f"[ADMIN] User {txn['user_id']} • Txn {txn['txn_id']} • Rs {txn['amount']} • fraud_prob={round(fraud_probability*100, 2)}%"
    message_user = f"⚠️ Suspicious txn #{txn['txn_id']} of Rs {txn['amount']}. If this isn’t you, please contact the bank."

    alert_doc = {
        "alert_time": datetime.now(),
        "user_id": txn["user_id"],
        "txn_id": txn["txn_id"],
        "amount": txn["amount"],
        "fraud_prob": fraud_probability,
        "message_user": message_user,
        "message_admin": message_admin,
        "account_flagged": True
    }
    alert_col.insert_one(alert_doc)
    high_risk_col.insert_one({
        **txn,
        "is_fraud": 1,
        "fraud_prob": fraud_probability,
        "fraud_pred": True,
        "lat_rounded": round(txn["location_lat"], 1),
        "long_rounded": round(txn["location_long"], 1),
        "total_txns": 14,  # can be dynamic
        "hour": datetime.now().hour,
        "day": datetime.now().day,
        "weekday": datetime.now().weekday()
    })

    print("🚨 ALERT: Fraud detected. Admin and User notified.")
else:
    print("✅ Normal transaction logged.")
