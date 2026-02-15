import pandas as pd

# ---------- Paths ----------
PRED_PATH = "data/isolation_forest_results.csv"          # or tuned file (see below)
TRUTH_PATH = "data/selected_users.csv"
OUT_PATH  = "outputs/metrics_if_tuned_real.csv"


# ---------- Load data ----------
pred = pd.read_csv(PRED_PATH)
truth = pd.read_csv(TRUTH_PATH)

# Make sure column name is correct
truth.columns = ["user_id"]

# Convert anomaly_flag to binary (1 = malicious, 0 = normal)
pred["pred_label"] = pred["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

# Create true label column
pred["true_label"] = pred["user_id"].isin(truth["user_id"]).astype(int)

# ---------- Confusion matrix ----------
TP = ((pred["pred_label"] == 1) & (pred["true_label"] == 1)).sum()
FP = ((pred["pred_label"] == 1) & (pred["true_label"] == 0)).sum()
TN = ((pred["pred_label"] == 0) & (pred["true_label"] == 0)).sum()
FN = ((pred["pred_label"] == 0) & (pred["true_label"] == 1)).sum()

# ---------- Metrics ----------
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
F1 = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0

# Save results
metrics = pd.DataFrame([{
    "TP": TP,
    "FP": FP,
    "TN": TN,
    "FN": FN,
    "FPR": round(FPR, 4),
    "TPR": round(TPR, 4),
    "Precision": round(Precision, 4),
    "F1": round(F1, 4)
}])

metrics.to_csv(OUT_PATH, index=False)

print("\n=== Real Evaluation Metrics (Isolation Forest Tuned) ===")
print(metrics)
