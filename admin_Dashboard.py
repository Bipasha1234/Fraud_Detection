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

    # Check if file changed
    if uploaded_file:
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            # New file uploaded, reset flag
            st.session_state.uploaded_done = False
            st.session_state.uploaded_file_name = uploaded_file.name

    if uploaded_file and not st.session_state.get("uploaded_done", False):
        df = pd.read_csv(uploaded_file)
        try:
            df = apply_feature_engineering(df)
            batch_name = f"batch_{dt.datetime.now():%Y%m%d_%H%M%S}"
            st.success(f"✅ Uploaded `{uploaded_file.name}` as `{batch_name}`")
            st.toast("🚀 Feature engineering applied!", icon="✨")
            batch_records = df.to_dict(orient="records")
            batch_collection.insert_many([{**rec, "batch_name": batch_name} for rec in batch_records])
            st.session_state.uploaded_done = True  # mark upload done
        except Exception as e:
            st.error(f"❌ Error in uploaded file: {e}")
            st.stop()
    elif not uploaded_file:
        # Reset if no file uploaded
        st.session_state.uploaded_done = False
        st.session_state.uploaded_file_name = None
        df = load_data()
    else:
        # File uploaded and already processed, load base data or keep last upload as you prefer
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
            batch_df = apply_feature_engineering(batch_df)
            model = load_model()
            features = ["amount", "haversine_km", "new_device", "login_txn_gap_min"]
            batch_df["fraud_prob"] = model.predict_proba(batch_df[features])[:, 1]
            threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

            batch_df["fraud_pred"] = (batch_df["fraud_prob"] >= threshold).astype(int)
            batch_df["fraud_label"] = batch_df["fraud_pred"].map({1: "🚨 RED FLAG", 0: "✅ Normal"})

            batch_df["action"] = batch_df["fraud_label"].map({
                "🚨 RED FLAG": "Freeze account & investigate",
                "✅ Normal": "No action"
            })

            batch_df["action"] = batch_df["fraud_prob"].apply(
                lambda x: "Freeze account & investigate" if x >= 0.95 else (
                    "Review transaction" if x >= 0.5 else "No action")
            )
            def generate_summary(row):
                summary_parts = []
                if row.get("login_time") and row.get("txn_time"):
                    txn_hour = pd.to_datetime(row["txn_time"]).hour
                    if 0 <= txn_hour < 6:
                         summary_parts.append(f"Unusual hour: {txn_hour}AM")

                if row.get("haversine_km") is not None and row["haversine_km"] > 50:
                    summary_parts.append(f"Unusual location: {row['haversine_km']:.1f} km away")
                if row.get("new_device") == 1:
                    summary_parts.append("Used new device")
                return "; ".join(summary_parts) if summary_parts else "No anomalies"

            batch_df["summary"] = batch_df.apply(generate_summary, axis=1)

            styled_df = batch_df[[
                "user_id", "txn_time", "amount", "fraud_prob", "fraud_label", "action", "summary"
            ]]

            def highlight_red(row):
                if row["fraud_label"] == "🚨 RED FLAG":
                    return ["background-color: #ffe6e6"] * len(row)
                return [""] * len(row)

            st.success(f"Showing batch: {selected_batch} ({len(batch_df)} records)")
            st.dataframe(styled_df.style.apply(highlight_red, axis=1))




            st.subheader("📊 Fraud Probability Distribution")
            # List of visualization options
            visualization_options = [
                "Fraud Probability Distribution",
                "Fraud vs Normal Counts",
                "Transaction Amount Distribution",
                "Fraud Transactions Over Time",
                "Haversine Distance by Fraud Label",
                "New Device Usage by Fraud Label"
            ]

            selected_visualizations = st.multiselect(
                "Select Visualizations to display",
                visualization_options,
                default=[]
            )

            if "Fraud Probability Distribution" in selected_visualizations:
                st.subheader("📊 Fraud Probability Distribution")
                fig_prob, ax_prob = plt.subplots()
                sns.histplot(batch_df["fraud_prob"], bins=30, kde=True, ax=ax_prob)
                ax_prob.set_xlabel("Fraud Probability")
                st.pyplot(fig_prob)

            if "Fraud vs Normal Counts" in selected_visualizations:
                st.subheader("🚩 Fraud vs Normal Counts")
                fig_count, ax_count = plt.subplots()
                sns.countplot(x="fraud_label", data=batch_df, ax=ax_count)
                ax_count.set_ylabel("Number of Transactions")
                st.pyplot(fig_count)

            if "Transaction Amount Distribution" in selected_visualizations:
                st.subheader("💰 Transaction Amount Distribution")
                fig_amt, ax_amt = plt.subplots()
                sns.boxplot(x="fraud_label", y="amount", data=batch_df, ax=ax_amt)
                ax_amt.set_ylabel("Transaction Amount")
                st.pyplot(fig_amt)

            if "Fraud Transactions Over Time" in selected_visualizations:
                st.subheader("🕒 Fraud Transactions Over Time")
                batch_df["txn_date"] = pd.to_datetime(batch_df["txn_time"]).dt.date
                fraud_over_time = batch_df.groupby(["txn_date", "fraud_label"]).size().unstack(fill_value=0)
                fig_time, ax_time = plt.subplots()
                fraud_over_time.plot(kind="line", ax=ax_time)
                ax_time.set_ylabel("Number of Transactions")
                ax_time.set_xlabel("Date")
                st.pyplot(fig_time)

            if "Haversine Distance by Fraud Label" in selected_visualizations:
                st.subheader("📍 Haversine Distance by Fraud Label")
                fig_dist, ax_dist = plt.subplots()
                sns.boxplot(x="fraud_label", y="haversine_km", data=batch_df, ax=ax_dist)
                st.pyplot(fig_dist)

            if "New Device Usage by Fraud Label" in selected_visualizations:
                st.subheader("💻 New Device Usage by Fraud Label")
                fig_device, ax_device = plt.subplots()
                sns.countplot(x="new_device", hue="fraud_label", data=batch_df, ax=ax_device)
                ax_device.set_xlabel("New Device (0=No, 1=Yes)")
                st.pyplot(fig_device)


        else:
            st.warning("No records found for this batch.")

# === Alerts Tab ===
# === Alerts Tab ===
with tabs[2]:
    st.header("⚠️ High-Risk Fraud Alerts (MongoDB)")

    # Only include alerts from uploaded batches (exclude base_data)
    batch_alerts = list(alert_collection.find(
        {"batch_name": {"$ne": "base_data"}},  # filter out base_data
        {"_id": 0}
    ))

    st.write("🚨 Alerts from Uploaded Batches:", len(batch_alerts))
    if batch_alerts:
        alert_df = pd.DataFrame(batch_alerts).sort_values("fraud_prob", ascending=False)
        st.dataframe(alert_df)
    else:
        st.info("✅ No alerts from batch uploads yet.")


with tabs[3]:
    st.header("👤 User-wise Fraud History")

    # Get user transaction data from uploaded_batches collection (excluding base_data)
    user_records = list(batch_collection.find(
        {"batch_name": {"$ne": "base_data"}},  # exclude default/base data
        {"_id": 0}
    ))

    if user_records:
        user_df = pd.DataFrame(user_records)
        
        # Group and aggregate to show user-wise summary
        grouped = user_df.groupby("user_id").agg({
            "txn_id": "count",
            "amount": ["sum", "mean"],
            "haversine_km": "mean",
            "new_device": "sum",
            "login_txn_gap_min": "mean"
        }).reset_index()

        # Flatten MultiIndex columns
        grouped.columns = [
            "user_id", 
            "total_transactions", 
            "total_amount", 
            "avg_amount", 
            "avg_haversine_km", 
            "new_devices_used", 
            "avg_login_txn_gap_min"
        ]

        grouped = grouped.sort_values("total_transactions", ascending=False)

        st.dataframe(grouped)

    else:
        st.info("✅ No user transactions found in uploaded batches.")
