import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Paths
MODEL_PATH = "models/fraud_rf_model.pkl"
NEW_DATA_PATH = "data/new_transactions.csv"
OLD_DATA_PATH = "data/merged_anomaly_features.csv"  # Your original training data
METRICS_PATH = "metrics/latest_metrics.json"

FEATURES = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]

def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_metrics(metrics):
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0)
    }

def balance_data(X, y):
    df = X.copy()
    df["is_fraud"] = y
    fraud = df[df["is_fraud"] == 1]
    non_fraud = df[df["is_fraud"] == 0]

    fraud_upsampled = resample(fraud,
                               replace=True,
                               n_samples=len(non_fraud),
                               random_state=42)
    balanced_df = pd.concat([non_fraud, fraud_upsampled])
    return balanced_df.drop("is_fraud", axis=1), balanced_df["is_fraud"]

def retrain_model(old_data_path, new_data_path):
    # Load old and new data
    old_df = pd.read_csv(old_data_path)
    new_df = pd.read_csv(new_data_path)

    combined = pd.concat([old_df, new_df]).drop_duplicates()

    X = combined[FEATURES]
    y = combined["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    metrics = evaluate_model(model, X_test, y_test)

    # Save retrained model
    joblib.dump(model, MODEL_PATH)
    save_metrics(metrics)

    print("Model retrained and saved.")
    print("New evaluation metrics:", metrics)
    return model, metrics

def monitor():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading new data...")
    new_df = pd.read_csv(NEW_DATA_PATH)
    X_new = new_df[FEATURES]
    y_new = new_df["is_fraud"]

    print("Evaluating on new data...")
    new_metrics = evaluate_model(model, X_new, y_new)

    print("New metrics:", new_metrics)
    old_metrics = load_metrics()

    if old_metrics is None:
        print("No previous metrics found, saving current metrics.")
        save_metrics(new_metrics)
    else:
        print("Previous metrics:", old_metrics)
        # Define a threshold for acceptable metric drop, e.g. 5%
        threshold = 0.05

        # Check if accuracy dropped significantly
        if new_metrics["accuracy"] < old_metrics["accuracy"] - threshold:
            print("Performance dropped. Retraining model...")
            model, new_metrics = retrain_model(OLD_DATA_PATH, NEW_DATA_PATH)
        else:
            print("Performance acceptable. Updating metrics.")
            save_metrics(new_metrics)

    print("Monitoring complete.")

if __name__ == "__main__":
    monitor()
