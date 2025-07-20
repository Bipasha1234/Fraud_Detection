import pandas as pd
from pymongo import MongoClient

# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["fraud_detection"]
collection = db["uploaded_batches"]

# 2. Query all documents
data = list(collection.find())

# 3. Convert to DataFrame (remove MongoDB’s ObjectId if needed)
for doc in data:
    doc['_id'] = str(doc['_id'])  # Convert ObjectId to string

df = pd.DataFrame(data)

# 4. Save to CSV
df.to_csv("fraud_transactions.csv", index=False)

print("Exported successfully to fraud_transactions.csv")
