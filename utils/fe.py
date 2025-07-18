# utils/fe.py
import pandas as pd

FEATURES = [
    "amount", "hour", "weekday", "is_debit",
    "total_txns", "lat_rounded", "lon_rounded"
]

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates training-time transformations on new data."""
    df = df.copy()
    df["txn_time"]   = pd.to_datetime(df["txn_time"])
    df["hour"]       = df["txn_time"].dt.hour
    df["weekday"]    = df["txn_time"].dt.dayofweek
    df["is_debit"]   = (df["txn_type"].str.lower()=="debit").astype(int)
    df["lat_rounded"] = df["location_lat"].round(2)
    df["lon_rounded"] = df["location_long"].round(2)

    # If caller doesn’t supply total_txns, default to 1
    if "total_txns" not in df.columns:
        df["total_txns"] = 1

    return df[FEATURES]
