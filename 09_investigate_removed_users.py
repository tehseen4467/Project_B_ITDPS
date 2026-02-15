import os
import pandas as pd

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load default and tuned results
default_df = pd.read_csv("data/isolation_forest_results.csv")
tuned_df = pd.read_csv("data/isolation_forest_results_tuned_003.csv")

# Load full feature table
features_df = pd.read_csv("data/user_features_if.csv")

# Merge default and tuned flags
merged = default_df[["user_id", "anomaly_flag"]].merge(
    tuned_df[["user_id", "anomaly_flag"]],
    on="user_id",
    suffixes=("_default", "_tuned")
)

# Users flagged before but cleared after tuning
removed_users = merged[
    (merged["anomaly_flag_default"] == -1) &
    (merged["anomaly_flag_tuned"] == 1)
]

# Attach full feature data
removed_with_features = removed_users.merge(
    features_df,
    on="user_id",
    how="left"
)

# Save to CSV
output_path = "outputs/removed_users_feature_analysis.csv"
removed_with_features.to_csv(output_path, index=False)

print("\nRemoved users saved to:", output_path)
print("Rows:", len(removed_with_features))
print("\nPreview:")
print(removed_with_features.head())
