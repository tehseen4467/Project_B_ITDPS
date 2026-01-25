import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load master dataset
df = pd.read_csv("master_dataset.csv")

# Select features for anomaly detection
features = [
    "total_events",
    "unique_actions",
    "unique_resources",
    "off_hours_events",
    "device_switches",
    "total_sessions",
    "total_bytes_sent",
    "avg_bytes_sent",
    "risky_sessions",
    "employment_status",
    "privilege_level"
]

X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso = IsolationForest(
    n_estimators=200,
    contamination=0.2,
    random_state=42
)

iso.fit(X_scaled)

# Predict anomalies
df["anomaly_flag"] = iso.predict(X_scaled)
df["anomaly_score"] = iso.decision_function(X_scaled)

# Convert anomaly score to risk score (0–100)
df["risk_score"] = (
    1 - (df["anomaly_score"] - df["anomaly_score"].min()) /
    (df["anomaly_score"].max() - df["anomaly_score"].min())
) * 100

df["risk_score"] = df["risk_score"].round(2)

# Save output
df.to_csv("master_with_anomaly_scores.csv", index=False)

print(" Isolation Forest completed i.e ANOMLY DETECTION ALGORITHM WOKRING PERFECTLY FINE.")
print(df[["user_id", "risk_score", "anomaly_flag", "label"]].sort_values("risk_score", ascending=False))
