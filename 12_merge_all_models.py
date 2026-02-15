import pandas as pd

# Load results
if_default = pd.read_csv("../data/isolation_forest_results.csv")
if_tuned = pd.read_csv("../data/isolation_forest_results_tuned_003.csv")
lstm = pd.read_csv("../outputs/lstm_anomaly_scores.csv")

# Rename columns to avoid collisions
if_default = if_default.rename(columns={
    "anomaly_flag": "if_flag_default",
    "anomaly_score": "if_score_default"
})

if_tuned = if_tuned.rename(columns={
    "anomaly_flag": "if_flag_tuned",
    "anomaly_score": "if_score_tuned"
})

# Merge
merged = if_default.merge(
    if_tuned[["user_id", "if_flag_tuned", "if_score_tuned"]],
    on="user_id"
).merge(
    lstm,
    on="user_id"
)

# Create combined ranking score
merged["combined_score"] = (
    (merged["if_score_tuned"].rank(pct=True)) +
    (merged["lstm_score_norm"])
) / 2

merged = merged.sort_values("combined_score", ascending=False)

merged.to_csv("../outputs/final_model_comparison.csv", index=False)

print("Saved: outputs/final_model_comparison.csv")
print(merged.head(10).to_string(index=False))
