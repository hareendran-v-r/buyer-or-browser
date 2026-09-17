"""Experiment 3: five-fold stratified CV with fold-local preprocessing."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR, RANDOM_STATE, load_data, make_forest_pipeline,
    make_logistic_pipeline,
)

X, y = load_data()
models = {
    "Logistic Regression": make_logistic_pipeline(X.columns),
    "Random Forest": make_forest_pipeline(X.columns),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
    "F1": "f1", "ROC_AUC": "roc_auc",
}
fold_rows = []
summary_rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    for fold in range(cv.n_splits):
        fold_rows.append({
            "Model": name, "Fold": fold + 1,
            **{metric: scores[f"test_{metric}"][fold] for metric in scoring},
        })
    summary = {"Model": name}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        summary[f"{metric}_Mean"] = values.mean()
        summary[f"{metric}_Std"] = values.std(ddof=1)
    summary_rows.append(summary)

folds = pd.DataFrame(fold_rows)
summary = pd.DataFrame(summary_rows)
folds.to_csv(OUTPUT_DIR / "03_cross_validation_folds.csv", index=False)
summary.to_csv(OUTPUT_DIR / "03_cross_validation_summary.csv", index=False)
print("\nSTRATIFIED CROSS-VALIDATION SUMMARY")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

data = [folds.loc[folds["Model"] == name, "F1"] for name in models]
plt.figure(figsize=(8, 5))
plt.boxplot(data, tick_labels=list(models))
plt.ylabel("Purchase-class F1")
plt.title("Five-fold Stratified Cross-Validation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_cross_validation_f1.png", dpi=300, bbox_inches="tight")
plt.close()
