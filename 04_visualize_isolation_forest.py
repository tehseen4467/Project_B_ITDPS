import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
INPUT_PATH = "data/isolation_forest_results.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load ----------

df = pd.read_csv(INPUT_PATH)

# Basic checks
required_cols = ["user_id", "anomaly_score", "anomaly_flag"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Normalize labels: -1 anomaly, 1 normal
df["is_anomaly"] = (df["anomaly_flag"] == -1).astype(int)

# ---------- Plot 1: Histogram of anomaly scores ----------
plt.figure()
plt.hist(df["anomaly_score"].dropna(), bins=30)
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("anomaly_score (lower = more anomalous)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_score_histogram.png"), dpi=200)
plt.close()

# ---------- Plot 2: Off-hours ratio vs anomaly score (if available) ----------
if "off_hours_ratio" in df.columns:
    plt.figure()
    plt.scatter(df["off_hours_ratio"], df["anomaly_score"])
    plt.title("Off-hours Ratio vs Anomaly Score")
    plt.xlabel("off_hours_ratio")
    plt.ylabel("anomaly_score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_offhours_vs_score.png"), dpi=200)
    plt.close()

# ---------- Plot 3: Compare normal vs anomaly (boxplots) ----------
# Pick a few features that usually matter
candidate_features = [
    "off_hours_ratio",
    "avg_events_per_day",
    "file_ratio",
    "device_ratio",
    "total_events",
    "active_days",
]
features = [f for f in candidate_features if f in df.columns]

if features:
    # Boxplot: two groups (normal vs anomaly) for each feature
    # We do one plot per feature to keep it clean (and not unreadable).
    for f in features:
        normal = df[df["is_anomaly"] == 0][f].dropna()
        anomaly = df[df["is_anomaly"] == 1][f].dropna()

        plt.figure()
        plt.boxplot([normal, anomaly], tick_labels=["Normal", "Anomaly"])
        plt.title(f"Normal vs Anomaly: {f}")
        plt.ylabel(f)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"03_box_{f}.png"), dpi=200)
        plt.close()

# ---------- Plot 4: Top 10 anomalies (bar chart) ----------
top = df.sort_values("anomaly_score").head(10)  # lowest scores = most anomalous
plt.figure()
plt.bar(top["user_id"].astype(str), top["anomaly_score"])
plt.title("Top 10 Most Anomalous Users (Lowest Scores)")
plt.xlabel("user_id")
plt.ylabel("anomaly_score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_top10_anomalies.png"), dpi=200)
plt.close()

print("Saved plots to:", OUT_DIR)
print("Generated files:")
for fn in sorted(os.listdir(OUT_DIR)):
    if fn.endswith(".png"):
        print(" -", fn)
        