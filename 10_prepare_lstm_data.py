import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "../data/event_level_dataset.csv"
OUT_X = "../data/lstm_X.npy"
OUT_USERS = "../data/lstm_user_ids.csv"

# Load
df = pd.read_csv(INPUT_PATH)

# Parse timestamp (CERT format is dayfirst)
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

# Drop bad timestamp rows (rare, but happens)
df = df.dropna(subset=["timestamp"])

# Create off-hours flag (customize if you want)
# Here: off-hours = before 9am OR after 6pm OR weekend
hour = df["timestamp"].dt.hour
weekday = df["timestamp"].dt.weekday  # 0=Mon ... 6=Sun
df["is_off_hours"] = ((hour < 9) | (hour >= 18) | (weekday >= 5)).astype(int)

# Encode event_type
le = LabelEncoder()
df["event_type_encoded"] = le.fit_transform(df["event_type"])

# Sort
df = df.sort_values(["user_id", "timestamp"])

# Build sequences per user
sequences = []
user_ids = []

for user_id, group in df.groupby("user_id"):
    seq = group[["event_type_encoded", "is_off_hours"]].to_numpy()
    sequences.append(seq)
    user_ids.append(user_id)

# Pad sequences to same length (LSTM needs fixed-length)
max_len = max(len(s) for s in sequences)
X = np.zeros((len(sequences), max_len, 2), dtype=np.float32)

for i, s in enumerate(sequences):
    X[i, :len(s), :] = s

np.save(OUT_X, X)
pd.DataFrame({"user_id": user_ids}).to_csv(OUT_USERS, index=False)

print("Total user sequences:", len(user_ids))
print("Max sequence length:", max_len)
print("Saved:", OUT_X, "and", OUT_USERS)
