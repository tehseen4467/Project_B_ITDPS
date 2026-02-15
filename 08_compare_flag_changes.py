import pandas as pd

default_df = pd.read_csv("data/isolation_forest_results.csv")[["user_id","anomaly_flag"]]
tuned_df   = pd.read_csv("data/isolation_forest_results_tuned_003.csv")[["user_id","anomaly_flag"]]

m = default_df.merge(tuned_df, on="user_id", suffixes=("_default","_tuned"))

# Users flagged before but not after = noise reduced
reduced = m[(m["anomaly_flag_default"] == -1) & (m["anomaly_flag_tuned"] == 1)]

# Users not flagged before but flagged after (rare, but good to show)
new_flags = m[(m["anomaly_flag_default"] == 1) & (m["anomaly_flag_tuned"] == -1)]

print("\nFlagged before, cleared after (reduced alerts):", len(reduced))
print(reduced.head(10).to_string(index=False))

print("\nNewly flagged after tuning:", len(new_flags))
print(new_flags.head(10).to_string(index=False))

reduced.to_csv("outputs/alerts_reduced_users.csv", index=False)
new_flags.to_csv("outputs/newly_flagged_users.csv", index=False)
