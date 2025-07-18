import datetime as dt
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- MongoDB Setup ---
client = MongoClient("mongodb://localhost:27017")
db = client["fraud_detection"]
batch_collection = db["uploaded_batches"]
alert_collection = db["fraud_alerts"]

# --- Load data and model ---
@st.cache_data
def load_data():
    df = pd.read_csv("merged_anomaly_features.csv", parse_dates=["txn_time", "login_time"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("fraud_rf_model.pkl")

# --- Haversine function ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(pd.to_numeric(lat1, errors='coerce'))
    lon1 = np.radians(pd.to_numeric(lon1, errors='coerce'))
    lat2 = np.radians(pd.to_numeric(lat2, errors='coerce'))
    lon2 = np.radians(pd.to_numeric(lon2, errors='coerce'))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- Feature engineering ---
def apply_feature_engineering(df):
    df["txn_time"] = pd.to_datetime(df["txn_time"])
    df["login_time"] = pd.to_datetime(df["login_time"])
    if "haversine_km" not in df.columns:
        df["haversine_km"] = haversine(df["location_lat"], df["location_long"], df["login_lat"], df["login_long"])
    if "new_device" not in df.columns:
        df["new_device"] = (df["device_id"] != df["login_device"]).astype(int)
    if "login_txn_gap_min" not in df.columns:
        df["login_txn_gap_min"] = (df["txn_time"] - df["login_time"]).dt.total_seconds() / 60.0
    return df

# --- Tabs ---
tabs = st.tabs(["🛡️ Dashboard", "📤 Uploaded Batches", "⚠️ Alerts", "👤 User-wise Fraud History"])

# === Dashboard Tab ===
with tabs[0]:
    st.sidebar.header("📤 Batch Upload")
    uploaded_file = st.sidebar.file_uploader("Upload new transaction CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            df = apply_feature_engineering(df)
            batch_name = f"batch_{dt.datetime.now():%Y%m%d_%H%M%S}"
            st.success(f"✅ Uploaded `{uploaded_file.name}` as `{batch_name}`")
            st.toast("🚀 Feature engineering applied!", icon="✨")
            batch_records = df.to_dict(orient="records")
            batch_collection.insert_many([{**rec, "batch_name": batch_name} for rec in batch_records])
        except Exception as e:
            st.error(f"❌ Error in uploaded file: {e}")
            st.stop()
    else:
        df = load_data()

    model = load_model()

    st.sidebar.header("📊 Filters")
    user_filter = st.sidebar.selectbox("Select User (optional):", ["All"] + sorted(df["user_id"].unique()))
    threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.05)

    if user_filter != "All":
        df = df[df["user_id"] == user_filter]

    features = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]
    df["fraud_prob"] = model.predict_proba(df[features])[:, 1]
    df["fraud_pred"] = (df["fraud_prob"] >= threshold).astype(int)

    st.title("🛡️ Fraud Detection Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Flagged as Fraud", df["fraud_pred"].sum())
    col3.metric("Avg Haversine Distance", round(df["haversine_km"].mean(), 2))

    st.subheader("🚨 High-Risk Alerts")
    alerts = df[df["fraud_pred"] == 1]
    if not alerts.empty:
        for _, row in alerts.head(5).iterrows():
            st.error(f"User {row['user_id']} → Rs {row['amount']:.2f} | Prob: {row['fraud_prob']:.2%}")

        alerts_to_save = alerts.copy()
        alerts_to_save["batch_name"] = batch_name if "batch_name" in locals() else "base_data"
        alert_records = alerts_to_save[["user_id", "txn_time", "amount", "fraud_prob", "batch_name"]].to_dict(orient="records")
        alert_collection.insert_many(alert_records)
    else:
        st.info("✅ No high-risk transactions detected at this threshold.")

    st.subheader("📍 Haversine Distance vs Fraud")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df, x="fraud_pred", y="haversine_km", ax=ax1)
    st.pyplot(fig1)

    st.subheader("💻 New Device Usage")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="new_device", hue="fraud_pred", ax=ax2)
    st.pyplot(fig2)

    st.subheader("⏱️ Login-Transaction Gap")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="fraud_pred", y="login_txn_gap_min", ax=ax3)
    st.pyplot(fig3)

    st.subheader("📋 Transactions Table")
    st.dataframe(df[["user_id", "txn_time", "amount", "fraud_prob", "fraud_pred"]])

# === Uploaded Batches Tab ===
with tabs[1]:
    st.header("📦 Uploaded Batches from MongoDB")
    batch_names = batch_collection.distinct("batch_name")
    selected_batch = st.selectbox("Select batch to view:", batch_names[::-1])

    if selected_batch:
        records = list(batch_collection.find({"batch_name": selected_batch}, {"_id": 0}))
        if records:
            batch_df = pd.DataFrame(records)
            st.success(f"Showing batch: {selected_batch} ({len(batch_df)} records)")
            st.dataframe(batch_df.head(100))
        else:
            st.warning("No records found for this batch.")

# === Alerts Tab ===
with tabs[2]:
    st.header("⚠️ High-Risk Fraud Alerts (MongoDB)")
    
    all_alerts = list(alert_collection.find({}, {"_id": 0}))
    st.write("🚨 Alerts Detected:", len(alerts))
    if all_alerts:
        alert_df = pd.DataFrame(all_alerts).sort_values("fraud_prob", ascending=False)
        st.dataframe(alert_df)
    else:
        st.info("✅ No alerts saved in the database yet.")

# === User-wise Fraud History ===
with tabs[3]:
    st.subheader("👤 User-wise Historical Fraud Analysis")
    user_ids = sorted(df["user_id"].unique())
    selected_user = st.selectbox("Select a user to view history:", user_ids)
    user_df = df[df["user_id"] == selected_user]

    if not user_df.empty:
        st.metric("Total Transactions", len(user_df))
        st.metric("Fraudulent Transactions", user_df["fraud_pred"].sum())
        st.line_chart(user_df.sort_values("txn_time")[["txn_time", "fraud_prob"]].set_index("txn_time"))
        st.dataframe(user_df[["txn_time", "amount", "fraud_prob", "fraud_pred"]])
    else:
        st.warning("No transactions found for this user.")
