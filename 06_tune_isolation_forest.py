import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load user features
df = pd.read_csv("data/user_features_if.csv")

# Separate features
feature_cols = df.columns.drop("user_id")
X = df[feature_cols]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tuned Isolation Forest (lower contamination = fewer flags)
model = IsolationForest(
    n_estimators=200,
    contamination=0.15,
    random_state=42
)

model.fit(X_scaled)

# Predict anomaly labels (-1 anomaly, 1 normal)
labels = model.predict(X_scaled)

# Add outputs
df_out = df.copy()
df_out["anomaly_flag"] = labels

# Optional but useful: add anomaly_score for ranking (higher = more normal)
scores = model.decision_function(X_scaled)
df_out["anomaly_score"] = scores

# Save tuned results
OUT_PATH = "data/isolation_forest_results_tuned_003.csv"
df_out.to_csv(OUT_PATH, index=False)

# Print summary
anomalies = (labels == -1).sum()
print("Saved:", OUT_PATH)
print("Total users:", len(df_out))
print("Anomalies detected:", anomalies)
print("Anomaly ratio:", round(anomalies / len(df_out), 4))

# Show top suspicious users (lowest scores)
top = df_out.sort_values("anomaly_score").head(10)[["user_id", "anomaly_score", "anomaly_flag"]]
print("\nTop suspicious users:")
print(top.to_string(index=False))
