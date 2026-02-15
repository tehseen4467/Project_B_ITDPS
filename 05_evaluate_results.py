import pandas as pd

# Load results
df = pd.read_csv("data/isolation_forest_results.csv")

total_users = len(df)
anomalies = (df["anomaly_flag"] == -1).sum()
normal = (df["anomaly_flag"] == 1).sum()

anomaly_ratio = anomalies / total_users

print("\n==== Isolation Forest Evaluation ====")
print("Total users:", total_users)
print("Normal users:", normal)
print("Anomalous users:", anomalies)
print("Anomaly ratio:", round(anomaly_ratio, 4))

# Top anomalous users
top_anomalies = df[df["anomaly_flag"] == -1].sort_values(
    by="anomaly_score"
).head(10)

print("\nTop suspicious users:")
print(top_anomalies[["user_id", "anomaly_score"]])
