import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load event-level dataset
df = pd.read_csv("data/event_level_dataset.csv")

# Encode event_type
le = LabelEncoder()
df["event_type_encoded"] = le.fit_transform(df["event_type"])

# Sort by user and timestamp
df = df.sort_values(["user_id", "timestamp"])

# Build sequences per user
sequences = []
user_ids = []

for user_id, group in df.groupby("user_id"):
    seq = group[["event_type_encoded", "is_off_hours"]].values
    sequences.append(seq)
    user_ids.append(user_id)

print("Total user sequences:", len(sequences))
