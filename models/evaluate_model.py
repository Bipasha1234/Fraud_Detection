import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODEL_PATH = "models/fraud_rf_model.pkl"
NEW_DATA_PATH = "data/new_transactions.csv"
METRICS_PATH = "metrics/latest_metrics.json"

# Load model
model = joblib.load(MODEL_PATH)

# Load new data
df = pd.read_csv(NEW_DATA_PATH)
FEATURES = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]

X_test = df[FEATURES]
y_test = df["is_fraud"]

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall
}

# Save metrics to JSON
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation metrics saved:", metrics)
