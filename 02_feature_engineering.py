import pandas as pd

# Load
user = pd.read_csv("user_activity_logs.csv")
network = pd.read_csv("network_logs.csv")
hr = pd.read_csv("hr_context_data.csv")

# Parse time + off-hours
user["timestamp"] = pd.to_datetime(user["timestamp"])
user["hour"] = user["timestamp"].dt.hour
user["is_off_hours"] = ((user["hour"] < 8) | (user["hour"] > 18)).astype(int)

# User features
user_features = (
    user.groupby("user_id")
    .agg(
        total_events=("action_type", "count"),
        unique_actions=("action_type", "nunique"),
        unique_resources=("resource", "nunique"),
        off_hours_events=("is_off_hours", "sum"),
        device_switches=("device", "nunique"),
    )
    .reset_index()
)

# Network features
network_features = (
    network.groupby("user_id")
    .agg(
        total_sessions=("session_id", "count"),
        total_bytes_sent=("bytes_sent", "sum"),
        avg_bytes_sent=("bytes_sent", "mean"),
        risky_sessions=("risk_flag", "sum"),
    )
    .reset_index()
)

# Merge
df = (
    user_features.merge(network_features, on="user_id", how="left")
    .merge(hr, on="user_id", how="left")
)

# Fill missing network values
for col in ["total_sessions", "total_bytes_sent", "avg_bytes_sent", "risky_sessions"]:
    df[col] = df[col].fillna(0)

# Encode
df["employment_status"] = df["employment_status"].map({"permanent": 0, "contract": 1})
df["privilege_level"] = df["privilege_level"].map({"low": 0, "medium": 1, "high": 2})

# Label per user (max severity)
label_map = {"benign": 0, "anomalous": 1, "malicious": 2}
user["label_num"] = user["label"].map(label_map)
labels = user.groupby("user_id")["label_num"].max().reset_index().rename(columns={"label_num": "label"})

df = df.merge(labels, on="user_id", how="left").fillna({"label": 0})

# Save master file
df.to_csv("master_dataset.csv", index=False)

print("Saved: master_dataset.csv; There we have a processed data file in which ALL NOISE have been REMOVED")
print(df)
