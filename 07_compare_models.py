import pandas as pd

default_df = pd.read_csv("data/isolation_forest_results.csv")
tuned_df = pd.read_csv("data/isolation_forest_results_tuned_003.csv")

default_anomalies = (default_df["anomaly_flag"] == -1).sum()
tuned_anomalies = (tuned_df["anomaly_flag"] == -1).sum()
total_users = len(default_df)

reduction = (default_anomalies - tuned_anomalies) / default_anomalies

summary = pd.DataFrame([{
    "total_users": total_users,
    "default_contamination": 0.05,
    "default_anomalies": default_anomalies,
    "default_ratio": round(default_anomalies/total_users, 4),
    "tuned_contamination": 0.03,
    "tuned_anomalies": tuned_anomalies,
    "tuned_ratio": round(tuned_anomalies/total_users, 4),
    "alert_reduction_percent": round(reduction*100, 2)
}])

summary.to_csv("outputs/model_comparison_summary.csv", index=False)

print(summary.to_string(index=False))
