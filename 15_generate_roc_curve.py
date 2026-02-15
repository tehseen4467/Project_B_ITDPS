import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Paths (RUN FROM PROJECT_CERT ROOT)
PRED_PATH = "data/isolation_forest_results_tuned_003.csv"
TRUTH_PATH = "data/silver_truth_labels.csv"

# Load predictions
pred = pd.read_csv(PRED_PATH)
truth = pd.read_csv(TRUTH_PATH)

# Build true labels
truth_users = set(truth["user_id"].astype(str))
pred["true_label"] = pred["user_id"].astype(str).isin(truth_users).astype(int)

# IMPORTANT:
# Isolation Forest anomaly_score:
# More negative = more anomalous
# For ROC, higher score should mean more likely malicious
# So we invert it

pred["roc_score"] = -pred["anomaly_score"]

y_true = pred["true_label"]
y_scores = pred["roc_score"]

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Isolation Forest")
plt.legend()
plt.grid(True)

# Save
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/roc_curve_if_tuned.png", dpi=300)
plt.show()

print("AUC:", roc_auc)
