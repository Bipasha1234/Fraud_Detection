import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]

# Export predictions
df = pd.DataFrame(list(db["api_predictions"].find()))
df.to_csv("fraud_predictions.csv", index=False)

# Export fraud alerts
df_alerts = pd.DataFrame(list(db["fraud_alert_log"].find()))
df_alerts.to_csv("fraud_alerts.csv", index=False)

print("✅ Exported both CSV files successfully.")
