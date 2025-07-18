import pandas as pd
from joblib import load

# ------------------------------------------------------------------
# CONFIG — edit if your model/threshold file names change
# ------------------------------------------------------------------
MODEL_PATH = "fraud_model.pkl"      # saved with joblib.dump(...)
BEST_TH    = 0.30                   # ← replace with your own best_th value
FEATURES   = [
    "amount",
    "hour",
    "weekday",
    "is_debit",
    "total_txns",
    "lat_rounded",
    "lon_rounded",
]
# ------------------------------------------------------------------

# load the trained model once (when the module is imported)
_model = load(MODEL_PATH)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates training-time feature engineering on fresh transaction rows.
    Expects columns: txn_time, txn_type, amount, location_lat, location_long …
    """
    df = df.copy()

    df["txn_time"]    = pd.to_datetime(df["txn_time"])
    df["hour"]        = df["txn_time"].dt.hour
    df["weekday"]     = df["txn_time"].dt.dayofweek
    df["is_debit"]    = (df["txn_type"].str.lower() == "debit").astype(int)
    df["lat_rounded"] = df["location_lat"].round(2)
    df["lon_rounded"] = df["location_long"].round(2)

    # running count of txns per user within the batch
    df["total_txns"]  = df.groupby("user_id").cumcount() + 1
    return df


def score_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns original dataframe with two extra columns:
    • fraud_prob  – probability from the model
    • fraud_pred  – True/False using BEST_TH
    """
    feats = prepare_features(raw_df)
    X     = feats[FEATURES]

    raw_df = raw_df.copy()
    raw_df["fraud_prob"] = _model.predict_proba(X)[:, 1]
    raw_df["fraud_pred"] = raw_df["fraud_prob"] >= BEST_TH
    return raw_df
