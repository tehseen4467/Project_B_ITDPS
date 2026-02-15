import pandas as pd

# Choose which model output you want to evaluate
PRED_PATH = "../data/isolation_forest_results_tuned_003.csv"
TRUTH_PATH = "../data/silver_truth_labels.csv"


TOP_K = 8  # change to 6/8/10 as needed

df = pd.read_csv(PRED_PATH)

# Most anomalous = smallest anomaly_score (more negative)
df = df.sort_values("anomaly_score", ascending=True)

silver = df.head(TOP_K)[["user_id"]].copy()
silver.to_csv(TRUTH_PATH, index=False)

print("Saved:", TRUTH_PATH)
print("Top suspicious users:", TOP_K)
print(silver.to_string(index=False))