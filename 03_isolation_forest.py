import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---- Load user features ----
df = pd.read_csv("data/user_features_if.csv")

user_ids = df["user_id"]
X = df.drop(columns=["user_id"])

# ---- Scale features ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Train Isolation Forest ----
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # assume 5% insiders
    random_state=42
)

model.fit(X_scaled)

# ---- Get anomaly scores ----
scores = model.decision_function(X_scaled)
labels = model.predict(X_scaled)   # -1 = anomaly, 1 = normal

# ---- Save results ----
results = df.copy()
results["anomaly_score"] = scores
results["anomaly_flag"] = labels

results.to_csv("data/isolation_forest_results.csv", index=False)

print("Anomalies detected:", (labels == -1).sum())
print("Results shape:", results.shape)
