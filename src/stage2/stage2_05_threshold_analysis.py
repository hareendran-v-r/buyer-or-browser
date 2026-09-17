"""Experiment 5: select a threshold from training OOF predictions only."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR, RANDOM_STATE, evaluate_model, load_data,
    make_forest_pipeline, split_data,
)

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
model = make_forest_pipeline(X.columns)

# Each training observation is scored by a model that did not train on it.
oof_probabilities = cross_val_predict(
    model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]
thresholds = np.arange(0.10, 0.91, 0.05)
rows = []
for threshold in thresholds:
    predictions = (oof_probabilities >= threshold).astype(int)
    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(y_train, predictions, zero_division=0),
        "Recall": recall_score(y_train, predictions, zero_division=0),
        "F1": f1_score(y_train, predictions, zero_division=0),
    })
curve = pd.DataFrame(rows)
selected_threshold = float(curve.loc[curve["F1"].idxmax(), "Threshold"])

# Fit once on all training data, then compare both fixed thresholds on the test set.
model.fit(X_train, y_train)
test_results = pd.DataFrame([
    {"Threshold_Type": "Default", "Threshold": 0.5,
     **evaluate_model(model, X_test, y_test, threshold=0.5)},
    {"Threshold_Type": "Selected from training OOF", "Threshold": selected_threshold,
     **evaluate_model(model, X_test, y_test, threshold=selected_threshold)},
])
curve.to_csv(OUTPUT_DIR / "05_training_oof_threshold_curve.csv", index=False)
test_results.to_csv(OUTPUT_DIR / "05_threshold_test_comparison.csv", index=False)

print("\nDECISION THRESHOLD ANALYSIS")
print(f"Threshold selected from training OOF F1: {selected_threshold:.2f}")
print(test_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

plt.figure(figsize=(9, 6))
for metric in ["Precision", "Recall", "F1"]:
    plt.plot(curve["Threshold"], curve[metric], marker="o", label=metric)
plt.axvline(0.5, color="gray", linestyle="--", label="Default = 0.50")
plt.axvline(selected_threshold, color="black", linestyle=":",
            label=f"Selected = {selected_threshold:.2f}")
plt.xlabel("Decision threshold")
plt.ylabel("Training out-of-fold score")
plt.title("Training-only Threshold Selection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_threshold_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
