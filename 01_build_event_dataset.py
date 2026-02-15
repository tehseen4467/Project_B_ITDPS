import pandas as pd

# Load data
logon = pd.read_csv("data/logon_subset.csv")
file = pd.read_csv("data/file_subset.csv")
device = pd.read_csv("data/device_subset.csv")

# Rename
logon = logon.rename(columns={"user": "user_id", "date": "timestamp"})
file = file.rename(columns={"user": "user_id", "date": "timestamp"})
device = device.rename(columns={"user": "user_id", "date": "timestamp"})

# Add event_type
logon["event_type"] = "logon"
file["event_type"] = "file"
device["event_type"] = "device"

# Convert timestamp
logon["timestamp"] = pd.to_datetime(logon["timestamp"], errors="coerce")

file["timestamp"] = pd.to_datetime(file["timestamp"], errors="coerce")
device["timestamp"] = pd.to_datetime(device["timestamp"], errors="coerce")


# Keep minimal columns
logon = logon[["user_id", "timestamp", "event_type"]]
file = file[["user_id", "timestamp", "event_type"]]
device = device[["user_id", "timestamp", "event_type"]]

# Combine
events = pd.concat([logon, file, device])

# Sort
events = events.sort_values(["user_id", "timestamp"])

# Save
events.to_csv("data/event_level_dataset.csv", index=False)

print("Final dataset shape:", events.shape)
