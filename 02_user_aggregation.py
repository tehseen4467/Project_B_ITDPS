import pandas as pd

# ---- Load unified event stream ----
events = pd.read_csv("data/event_level_dataset.csv")

# Parse timestamp (again, just to be safe)
events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
events = events.dropna(subset=["timestamp"])

# ---- Basic time features ----
events["hour"] = events["timestamp"].dt.hour
events["date_only"] = events["timestamp"].dt.date

# Define "off-hours" (customize if you want)
# Off-hours = before 6 AM OR after 8 PM
events["is_off_hours"] = ((events["hour"] < 6) | (events["hour"] >= 20)).astype(int)

# ---- Event type counts per user ----
type_counts = (
    events.pivot_table(
        index="user_id",
        columns="event_type",
        values="timestamp",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

# Ensure expected columns exist even if some types missing
for col in ["logon", "file", "device"]:
    if col not in type_counts.columns:
        type_counts[col] = 0

# ---- User-level aggregates ----
agg = events.groupby("user_id").agg(
    total_events=("event_type", "count"),
    active_days=("date_only", "nunique"),
    off_hours_events=("is_off_hours", "sum"),
).reset_index()

# Off-hours ratio
agg["off_hours_ratio"] = agg["off_hours_events"] / agg["total_events"]

# Average events per active day
agg["avg_events_per_day"] = agg["total_events"] / agg["active_days"].replace(0, 1)

# ---- Merge type counts into aggregates ----
user_features = agg.merge(type_counts[["user_id", "logon", "file", "device"]], on="user_id", how="left")

# Optional: ratios by type
user_features["logon_ratio"] = user_features["logon"] / user_features["total_events"]
user_features["file_ratio"] = user_features["file"] / user_features["total_events"]
user_features["device_ratio"] = user_features["device"] / user_features["total_events"]

# ---- Save output ----
user_features.to_csv("data/user_features_if.csv", index=False)

print("Saved user-level feature table:", user_features.shape)
print(user_features.head(5))
